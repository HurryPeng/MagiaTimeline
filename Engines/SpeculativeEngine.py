from IR import *
from Strategies.AbstractStrategy import *
from Strategies.AbstractStrategy import AbstractStrategy
from Util import *
from Engines.AbstractEngine import *

import av.container
import av.container.input
import av.video
import av.video.stream
import av.frame
import typing
import fractions
import math

class IntervalGrower(IIR):
    def __init__(
        self,
        flagIndexType: typing.Type[AbstractFlagIndex],
        fps: fractions.Fraction,
        timeBase: fractions.Fraction,
        mainFlagIndex: AbstractFlagIndex,
        featureFlagIndex: AbstractFlagIndex,
        aggregateFeatures: typing.Callable[[typing.List[typing.Any]], typing.Any],
        aggregateAndMoveFeatureToIntervalOnHook: bool = False,
        ocrFrameFlagIndex: typing.Optional[AbstractFlagIndex] = None,
        verbose: bool = False
    ) -> None:
        super().__init__(flagIndexType, fps, timeBase)
        self.mainFlagIndex: AbstractFlagIndex = mainFlagIndex
        self.featureFlagIndex: AbstractFlagIndex = featureFlagIndex
        self.aggregateFeatures: typing.Callable[[typing.List[typing.Any]], typing.Any] = aggregateFeatures
        self.aggregateAndMoveFeatureToIntervalOnHook: bool = aggregateAndMoveFeatureToIntervalOnHook
        self.ocrFrameFlagIndex: typing.Optional[AbstractFlagIndex] = ocrFrameFlagIndex

        self.proposalStride: fractions.Fraction = fractions.Fraction(1, 2)

        self.verbose = verbose

    def propose(self) -> typing.Tuple[int, typing.Optional[Interval], typing.Optional[Interval]]:
        # returns: proposed timestampC (-1 for next I-frame), previous interval, next interval

        if len(self.intervals) == 0:
            # initial proposal
            return 0, None, None
        if len(self.intervals) == 1:
            # second proposal
            return -1, self.intervals[0], None

        # scan self.intervals from the back
        # find the earliest interval that does not touch its next interval
        # propose the mid point of the gap and the two intervals

        cur: int = 0
        for i in range(len(self.intervals) - 2, -1, -1):
            prev = self.intervals[i]
            next = self.intervals[i + 1]
            if prev.touches(next):
                cur = i + 1
                break

        if cur == len(self.intervals) - 1:
            return -1, self.intervals[cur], None
        
        prev = self.intervals[cur]
        next = self.intervals[cur + 1]
        propose: int = math.floor(prev.end + (next.begin - prev.end) * self.proposalStride)
        return propose, prev, next
    
    def insertInterval(self, framePoint: FramePoint, frame: typing.Optional[av.frame.Frame]) -> Interval:
        interval = Interval(self.flagIndexType, self.mainFlagIndex, framePoint.timestamp, framePoint.timestamp, [framePoint])
        if self.ocrFrameFlagIndex is not None:
            assert frame is not None
            interval.setFlag(self.ocrFrameFlagIndex, frame)
        self.intervals.append(interval)
        self.sort()
        if self.verbose:
            print("insertInterval       ", f"({interval.begin}, {interval.end}) {(interval.end - interval.begin)}", formatTimestamp(self.timeBase, interval.begin))
        return interval

    def extendInterval(self, interval: Interval, framePoint: FramePoint) -> None:
        if interval.begin > framePoint.timestamp:
            interval.begin = framePoint.timestamp
            if self.verbose:
                print("extendIntervalToLeft ", f"<{interval.begin}, {interval.end}] {(interval.end - interval.begin)}", formatTimestamp(self.timeBase, interval.begin))
        elif interval.end < framePoint.timestamp:
            interval.end = framePoint.timestamp
            if self.verbose:
                print("extendIntervalToRight", f"[{interval.begin}, {interval.end}> {(interval.end - interval.begin)}", formatTimestamp(self.timeBase, interval.end))
        interval.framePoints.append(framePoint)
        self.sort()

    def hookInterval(self, intervalL: Interval, intervalR: Interval):
        assert intervalL.end < intervalR.begin
        intervalL.end = intervalR.begin
        if self.aggregateAndMoveFeatureToIntervalOnHook:
            features = [framePoint.getFlag(self.featureFlagIndex) for framePoint in intervalL.framePoints]
            featuresAggregated = self.aggregateFeatures(features)
            intervalL.setFlag(self.featureFlagIndex, featuresAggregated, inDiskCache=True)
            for framePoint in intervalL.framePoints:
                framePoint.setFlag(self.featureFlagIndex, None)
        if self.verbose:
            print("hookInterval         ", f"[{intervalL.begin}, {intervalL.end}}} {(intervalL.end - intervalL.begin)}", formatTimestamp(self.timeBase, intervalL.end))


class FrameCache:
    def __init__(self, container: "av.container.InputContainer", stream: "av.video.stream.VideoStream") -> None:
        self.container: "av.container.InputContainer" = container
        self.stream: "av.video.stream.VideoStream" = stream
        self.cache: typing.List[av.frame.Frame] = []
        self.begin: int = 0
        self.end: int = 0
        self.nextI: int = 0

        self.fps: fractions.Fraction = self.stream.average_rate
        self.timeBase: fractions.Fraction = self.stream.time_base
        # For variable frame rate videos, the unit timestamp is only the average frame duration
        # Do not use this for calculating the precise timestamp of the next I-frame
        # Use for approximation only
        self.unitTimestamp: int = int(1 / self.timeBase / self.fps)

        self.statDecodedFrames: int = 0

    # Get the frame that is closest to C within the [L, R) range
    # May return None if there is no frame in the range
    # If no R is specified, then R is the next I-frame
    def getFrame(self, timestampC: int, timestampL: int, timestampR: int = -1) -> typing.Optional[av.frame.Frame]:
        if timestampR == -1:
            assert self.nextI != 0
            timestampR = self.nextI
        assert timestampR <= self.nextI
        assert timestampC >= timestampL and timestampC <= timestampR
        assert timestampL >= self.begin

        # Make sure that now the cache contains at least one more frame after timestampC
        self.proceedTo(timestampC + 1)

        minDist: int = -1
        minFrame: typing.Optional[av.frame.Frame] = None

        for frame in self.cache:
            if frame.pts < timestampL or frame.pts >= timestampR:
                continue
            dist: int = abs(frame.pts - timestampC)
            if minFrame is None or dist < minDist:
                minDist = dist
                minFrame = frame
    
        return minFrame

    # Proceed until the latest frame timestamp is greater than or equal to tgtTimestamp
    # If tgtTimestamp == -1, then proceed until the end of the video
    def proceedTo(self, tgtTimestamp: int) -> None:
        if tgtTimestamp <= self.end and not tgtTimestamp == -1:
            return
        try:
            for frame in self.container.decode(self.stream):
                if frame is None:
                    break
                self.statDecodedFrames += 1
                self.cache.append(frame)
                if frame.pts >= tgtTimestamp and not tgtTimestamp == -1:
                    break
        except av.EOFError:
            pass
        self.end = self.cache[-1].pts
    
    def leap(self) -> typing.Optional[av.frame.Frame]:
        # leap to the next next I-frame (using seek) and take down that frame
        # return to the next I-frame and build an empty cache that is large enough for all frames in between
        # return the frame

        del self.cache[:]

        frameI2: typing.Optional[av.frame.Frame] = None
        try:
            # Sometimes the seek returns the same frame even if the offset is increased by 1
            # So we increase the offset by the unit timestamp instead
            # This might be a bug in the pyav library
            self.container.seek(self.nextI + self.unitTimestamp, stream=self.stream, any_frame=False, backward=False)
            frameI2 = next(self.container.decode(self.stream))
            assert frameI2.pts > self.nextI
            self.statDecodedFrames += 1
        except av.PermissionError:
            frameI2 = None
        self.container.seek(self.nextI, stream=self.stream, any_frame=False)
        frameI1: av.frame.Frame = next(self.container.decode(self.stream))
        self.statDecodedFrames += 1

        if frameI2 is not None and frameI2.pts == frameI1.pts:
            # This is init and the first frame's pts is not 0
            self.container.seek(frameI1.pts + self.unitTimestamp, stream=self.stream, any_frame=False, backward=False)
            frameI2 = next(self.container.decode(self.stream))
            assert frameI2.pts > frameI1.pts
            self.statDecodedFrames += 1
            self.container.seek(0, stream=self.stream, any_frame=False, backward=False)
            frameI1 = next(self.container.decode(self.stream))
            self.statDecodedFrames += 1

        self.cache = [frameI1]
        self.begin = frameI1.pts
        self.end = frameI1.pts

        if frameI2 is not None:
            self.nextI = frameI2.pts
        else:
            # last segment
            self.proceedTo(-1)
            self.nextI = self.end + 1
        return frameI2

class SpeculativeEngine(AbstractEngine):
    def __init__(self, config: dict) -> None:
        self.config: dict = config
        self.emptyGapForceCheck: int = config["emptyGapForceCheck"]
        self.debug: bool = config["debug"]

    def getRequiredAbstractStrategyType(self) -> type[AbstractStrategy]:
        return AbstractSpeculativeStrategy

    def run(self, strategy: AbstractSpeculativeStrategy, container: "av.container.InputContainer", stream: "av.video.stream.VideoStream") -> IIR:
        timeBase: fractions.Fraction = stream.time_base
        fps: fractions.Fraction = stream.average_rate
        frameCount: float = stream.frames

        mainFlagIndex: AbstractFlagIndex = strategy.getMainFlagIndex()
        featureFlagIndex: AbstractFlagIndex = strategy.getFeatureFlagIndex()
        ocrFrameFlagIndex: typing.Optional[AbstractFlagIndex] = None
        if isinstance(strategy, AbstractOcrStrategy):
            ocrFrameFlagIndex = strategy.getOcrFrameFlagIndex()

        self.emptyFeatureMaxTimestamp: int = ms2Timestamp(self.emptyGapForceCheck, timeBase)

        intervalGrower: IntervalGrower = IntervalGrower(
            strategy.getFlagIndexType(),
            fps,
            timeBase,
            mainFlagIndex,
            featureFlagIndex,
            strategy.aggregateFeatures,
            strategy.aggregateAndMoveFeatureToIntervalOnHook(),
            ocrFrameFlagIndex,
            self.debug)
        frameCache: FrameCache = FrameCache(container, stream)

        print("==== IIR Building ====")

        # let the interval grower propose and insert intervals until the end of the video
        # if it proposes a timestamp, request the frame from the frame cache
        # compare the proposed timestamp with the timestamps of the previous and next intervals
        # if it is mergable with the previous or next intervals, merge them
        # if it is different from the previous and next intervals, insert it as a new interval
        # if it proposes -1, proceed to the next I-frame and insert it as if it were proposed by the interval grower
        # if the next I-frame is None, then let the interval grower propose the last timestamp in the video
        # when it again proposes -1, break the loop and end the building process

        lastSegment: bool = False
        while True:
            proposeC, prev, next = intervalGrower.propose()
            if lastSegment and proposeC == -1:
                break
            if proposeC == 0: # Init
                assert prev is None and next is None
                frameI2 = frameCache.leap()
                if frameI2 is None: # Last segment
                    lastSegment = True
                    frameI2 = frameCache.getFrame(frameCache.end, frameCache.begin, frameCache.nextI)
                frameI1 = frameCache.getFrame(frameCache.begin, frameCache.begin, frameCache.nextI)
                framePoint1 = strategy.genFramePoint(avFrame2CvMat(frameI1), frameI1.pts)
                interval1 = intervalGrower.insertInterval(framePoint1, frameI1)
                prev = interval1
                framePoint2 = strategy.genFramePoint(avFrame2CvMat(frameI2), frameI2.pts)
                merge = strategy.decideFeatureMerge([framePoint.getFlag(featureFlagIndex) for framePoint in interval1.framePoints], [framePoint2.getFlag(featureFlagIndex)])
                if merge:
                    intervalGrower.extendInterval(interval1, framePoint2)
                else:
                    intervalGrower.insertInterval(framePoint2, frameI2)
            elif proposeC == -1: # leap to the next next I-frame
                assert prev is not None and next is None
                frameI2 = frameCache.leap()
                if frameI2 is None: # Last segment
                    lastSegment = True
                    frameI2 = frameCache.getFrame(frameCache.end, frameCache.begin, frameCache.nextI)
                framePoint2 = strategy.genFramePoint(avFrame2CvMat(frameI2), frameI2.pts)
                print(framePoint2.toString(timeBase))
                merge = strategy.decideFeatureMerge([framePoint.getFlag(featureFlagIndex) for framePoint in prev.framePoints], [framePoint2.getFlag(featureFlagIndex)])
                dist = prev.distFramePoint(framePoint2)
                if merge and not dist > self.emptyFeatureMaxTimestamp:
                    intervalGrower.extendInterval(prev, framePoint2)
                else:
                    intervalGrower.insertInterval(framePoint2, frameI2)
            else: # reasonable proposal
                assert prev is not None and next is not None
                frame = frameCache.getFrame(proposeC, prev.end + 1, next.begin)
                if frame is None:
                    # The two intervals have no more frames in between
                    intervalGrower.hookInterval(prev, next)
                    continue
                framePoint = strategy.genFramePoint(avFrame2CvMat(frame), frame.pts)
                isEmptyFeature = strategy.isEmptyFeature(framePoint.getFlag(featureFlagIndex))

                mergeLeft = strategy.decideFeatureMerge([framePoint.getFlag(featureFlagIndex) for framePoint in prev.framePoints], [framePoint.getFlag(featureFlagIndex)])
                distLeft = prev.distFramePoint(framePoint)
                if mergeLeft and not (isEmptyFeature and distLeft > self.emptyFeatureMaxTimestamp):
                    intervalGrower.extendInterval(prev, framePoint)
                else:
                    mergeRight = strategy.decideFeatureMerge([framePoint.getFlag(featureFlagIndex)], [framePoint.getFlag(featureFlagIndex) for framePoint in next.framePoints])
                    distRight = next.distFramePoint(framePoint)
                    if mergeRight and not (isEmptyFeature and distRight > self.emptyFeatureMaxTimestamp):
                        intervalGrower.extendInterval(next, framePoint)
                    else:
                        intervalGrower.insertInterval(framePoint, frame)
        
        print("==== IIR Passes ====")

        print("iirPassSuppressNonMain")
        iirPassSuppressNonMain = IIRPassRemovePredicate(lambda interval: not interval.framePoints[0].getFlag(mainFlagIndex))
        iirPassSuppressNonMain.apply(intervalGrower)
        for name, iirPass in (strategy.getSpecIirPasses()).items():
            print(name)
            iirPass.apply(intervalGrower)

        print("totalFrames/decoded/analyzed", frameCount, frameCache.statDecodedFrames, strategy.statAnalyzedFrames)

        if hasattr(strategy, "statDecideFeatureMerge"):
            print("statDecideFeatureMerge", strategy.statDecideFeatureMerge)
            print("statDecideFeatureMergeDiff", strategy.statDecideFeatureMergeDiff)
            print("statDecideFeatureMergeComputeECC", strategy.statDecideFeatureMergeComputeECC)
            print("statDecideFeatureMergeFindTransformECC", strategy.statDecideFeatureMergeFindTransformECC)
            print("statDecideFeatureMergeOCR", strategy.statDecideFeatureMergeOCR)

        return intervalGrower
