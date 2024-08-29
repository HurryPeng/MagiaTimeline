import av
import av.container
from IR import *
from Strategies.AbstractStrategy import *
import av.container
import av.container.input
import av.video
import av.video.stream
import av.frame
from Util import *

class IntervalGrower(IIR):
    def __init__(
        self,
        flagIndexType: typing.Type[AbstractFlagIndex],
        fps: fractions.Fraction,
        unitTimestamp: int,
        mainFlagIndex: AbstractFlagIndex,
        ocrFrameFlagIndex: typing.Optional[AbstractFlagIndex] = None
    ) -> None:
        super().__init__(flagIndexType, fps, unitTimestamp)
        self.mainFlag: AbstractFlagIndex = mainFlagIndex
        self.ocrFrameFlagIndex: typing.Optional[AbstractFlagIndex] = ocrFrameFlagIndex

        self.proposalStride: fractions.Fraction = fractions.Fraction(1, 2)

        self.processedFrames: typing.Set[int] = set()

        self.verbose = True

    def propose(self) -> typing.Tuple[int, typing.Optional[Interval], typing.Optional[Interval]]:
        # returns: proposed timestamp (-1 for no proposal), previous interval, next interval

        if len(self.intervals) == 0:
            return 0, None, None
        if len(self.intervals) == 1:
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
        return int(prev.end * (1 - self.proposalStride) + next.begin * self.proposalStride), prev, next
    
    def insertInterval(self, framePoint: FramePoint, frame: typing.Optional[av.frame.Frame] = None) -> Interval:
        interval = Interval(self.flagIndexType, self.mainFlag, framePoint.timestamp, framePoint.timestamp + self.unitTimestamp, [framePoint])
        if self.ocrFrameFlagIndex is not None and frame is not None:
            interval.flags[self.ocrFrameFlagIndex] = frame
        self.intervals.append(interval)
        self.sort()
        self.processedFrames.add(framePoint.timestamp)
        if self.verbose:
            print("insertInterval       ", f"({interval.begin // self.unitTimestamp}, {interval.end // self.unitTimestamp}) {(interval.end - interval.begin) // self.unitTimestamp}", framePoint.toString(1 / self.unitTimestamp / self.fps, 1))
        return interval

    def extendInterval(self, interval: Interval, framePoint: FramePoint):
        if interval.begin > framePoint.timestamp:
            interval.begin = framePoint.timestamp
            if self.verbose:
                print("extendIntervalToLeft ", f"<{interval.begin // self.unitTimestamp}, {interval.end // self.unitTimestamp}] {(interval.end - interval.begin) // self.unitTimestamp}", framePoint.toString(1 / self.unitTimestamp / self.fps, 1))
        elif interval.end < framePoint.timestamp + self.unitTimestamp:
            interval.end = framePoint.timestamp + self.unitTimestamp
            if self.verbose:
                print("extendIntervalToRight", f"[{interval.begin // self.unitTimestamp}, {interval.end // self.unitTimestamp}> {(interval.end - interval.begin) // self.unitTimestamp}", framePoint.toString(1 / self.unitTimestamp / self.fps, 1))
        interval.framePoints.append(framePoint)
        self.processedFrames.add(framePoint.timestamp)
        self.sort()

    def isProcessed(self, timestamp: int) -> bool:
        return timestamp in self.processedFrames

class FrameCache:
    def __init__(self, container: av.container.InputContainer, stream: av.video.stream.VideoStream) -> None:
        self.container: av.container.InputContainer = container
        self.stream: av.video.stream.VideoStream = stream
        self.cache: typing.List[av.frame.Frame] = []
        self.indexBegin: int = 0
        self.indexEnd: int = 0
        self.indexNextI: int = 0

        self.fps: fractions.Fraction = self.stream.average_rate
        self.timeBase: fractions.Fraction = self.stream.time_base
        self.unitTimestamp: int = int(1 / self.timeBase / self.fps)

        self.statDecodedFrames: int = 0

    def canGetFrame(self, timestamp: int) -> bool:
        index: int = int(timestamp // self.unitTimestamp)
        return index >= self.indexBegin and index < self.indexNextI

    def getFrame(self, timestamp: int) -> av.frame.Frame:
        index: int = int(timestamp // self.unitTimestamp)
        if index < self.indexBegin:
            raise Exception("FrameCache: timestamp earlier than the previous I-frame")
        if index > self.indexNextI and self.indexNextI != -1:
            raise Exception("FrameCache: timestamp later than the next I-frame")
        if index >= self.indexEnd:
            self.proceedTo(timestamp)
        assert index >= self.indexBegin and index < self.indexEnd
        return self.cache[index - self.indexBegin]
    
    def proceedTo(self, timestamp: int) -> None:
        tgtIndex: int = int(timestamp // self.unitTimestamp)
        # print("proceedTo", formatTimestamp(self.timeBase, timestamp), timestamp, tgtIndex)
        if tgtIndex < self.indexEnd:
            return
        # print("proceedTo", formatTimestamp(self.timeBase, timestamp), timestamp, tgtIndex)
        
        tgtTimestamp: int = tgtIndex * self.unitTimestamp
        for frame in self.container.decode(self.stream):
            self.statDecodedFrames += 1
            curIndex: int = int(frame.pts // self.unitTimestamp)
            self.cache[curIndex - self.indexBegin] = frame
            if frame.pts >= tgtTimestamp:
                break
        self.indexEnd = tgtIndex + 1
    
    def leap(self) -> typing.Optional[av.frame.Frame]:
        # leap to the next next I-frame (using seek) and take down that frame
        # return to the next I-frame and build an empty cache that is large enough for all frames in between
        # return the frame

        del self.cache[:]

        frameI2: typing.Optional[av.frame.Frame] = None
        try:
            self.container.seek((self.indexNextI + 1) * self.unitTimestamp, stream=self.stream, any_frame=False, backward=False)
            frameI2 = next(self.container.decode(self.stream))
            self.statDecodedFrames += 1
        except Exception:
            frameI2 = None
        self.container.seek(self.indexNextI * self.unitTimestamp, stream=self.stream, any_frame=False)
        frameI1: av.frame.Frame = next(self.container.decode(self.stream))
        self.statDecodedFrames += 1

        self.indexBegin = int(frameI1.pts // self.unitTimestamp)
        self.indexEnd = self.indexBegin + 1
        if frameI2 is not None:
            self.indexNextI = int(frameI2.pts // self.unitTimestamp)
            self.cache = [frameI1] + [None] * (self.indexNextI - self.indexBegin - 1)
        else:
            self.indexNextI = -1
            self.cache = [frameI1] + [None] * (self.stream.frames - self.indexBegin - 1)
        return frameI2

class SpeculativeEngine:
    def __init__(self) -> None:
        self.emptyFeatureMaxTime: int = 1000

    def run(self, strategy: SpeculativeStrategy, container: av.container.InputContainer, stream: av.video.stream.VideoStream) -> IIR:
        strategy: SpeculativeStrategy = strategy
        container: av.container.InputContainer = container
        stream: av.video.stream.VideoStream = stream

        timeBase: fractions.Fraction = stream.time_base
        fps: fractions.Fraction = stream.average_rate
        frameCount: float = stream.frames
        unitTimestamp: int = int(1 / timeBase / fps)

        mainFlagIndex: AbstractFlagIndex = strategy.getMainFlagIndex()
        featureFlagIndex: AbstractFlagIndex = strategy.getFeatureFlagIndex()
        ocrFrameFlagIndex: typing.Optional[AbstractFlagIndex] = None
        if isinstance(strategy, OcrStrategy):
            ocrFrameFlagIndex = strategy.getOcrFrameFlagIndex()

        self.emptyFeatureMaxTimestamp: int = ms2Timestamp(self.emptyFeatureMaxTime, fps, unitTimestamp)

        intervalGrower: IntervalGrower = IntervalGrower(strategy.getFlagIndexType(), fps, unitTimestamp, mainFlagIndex, ocrFrameFlagIndex)
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
            timestamp, prev, next = intervalGrower.propose()
            if lastSegment and timestamp == -1:
                break
            if timestamp == 0: # Init
                frameI2 = frameCache.leap()
                if frameI2 is None:
                    lastSegment = True
                    frameI2 = frameCache.getFrame((stream.frames - 1) * unitTimestamp)
                frameI1 = frameCache.getFrame(0)
                framePoint1 = strategy.genFramePoint(avFrame2CvMat(frameI1), int(frameI1.pts // unitTimestamp), frameI1.pts)
                interval1 = intervalGrower.insertInterval(framePoint1, frameI1)
                prev = interval1
                framePoint2 = strategy.genFramePoint(avFrame2CvMat(frameI2), int(frameI2.pts // unitTimestamp), frameI2.pts)
                merge = strategy.decideFeatureMerge([framePoint.getFlag(featureFlagIndex) for framePoint in interval1.framePoints], framePoint2.getFlag(featureFlagIndex))
                if merge:
                    intervalGrower.extendInterval(interval1, framePoint2)
                else:
                    intervalGrower.insertInterval(framePoint2, frameI2)
            elif timestamp == -1: # leap to the next next I-frame
                frameI2 = frameCache.leap()
                if frameI2 is None:
                    lastSegment = True
                    frameI2 = frameCache.getFrame((stream.frames - 1) * unitTimestamp)
                framePoint2 = strategy.genFramePoint(avFrame2CvMat(frameI2), int(frameI2.pts // unitTimestamp), frameI2.pts)
                print(framePoint2.toString(timeBase, 1))
                merge = strategy.decideFeatureMerge([framePoint.getFlag(featureFlagIndex) for framePoint in prev.framePoints], framePoint2.getFlag(featureFlagIndex))
                dist = prev.distFramePoint(framePoint2)
                if merge and not (strategy.isEmptyFeature(framePoint2.getFlag(featureFlagIndex)) and dist > self.emptyFeatureMaxTimestamp):
                    intervalGrower.extendInterval(prev, framePoint2)
                else:
                    intervalGrower.insertInterval(framePoint2, frameI2)
            else: # reasonable proposal
                frame = frameCache.getFrame(timestamp)
                framePoint = strategy.genFramePoint(avFrame2CvMat(frame), int(frame.pts // unitTimestamp), frame.pts)
                isEmptyFeature = strategy.isEmptyFeature(framePoint.getFlag(featureFlagIndex))

                mergeLeft = strategy.decideFeatureMerge([framePoint.getFlag(featureFlagIndex) for framePoint in prev.framePoints], framePoint.getFlag(featureFlagIndex))
                distLeft = prev.distFramePoint(framePoint)
                if mergeLeft and not (isEmptyFeature and distLeft > self.emptyFeatureMaxTimestamp):
                    intervalGrower.extendInterval(prev, framePoint)
                else:
                    mergeRight = strategy.decideFeatureMerge([framePoint.getFlag(featureFlagIndex) for framePoint in next.framePoints], framePoint.getFlag(featureFlagIndex))
                    distRight = next.distFramePoint(framePoint)
                    if mergeRight and not (isEmptyFeature and distRight > self.emptyFeatureMaxTimestamp):
                        intervalGrower.extendInterval(next, framePoint)
                    else:
                        intervalGrower.insertInterval(framePoint, frame)
        
        print("==== IIR Passes ====")

        print("iirPassSuppressNonMain")
        iirPassSuppressNonMain = IIRPassRemovePredicate(lambda interval: not interval.framePoints[0].getFlag(mainFlagIndex))
        iirPassSuppressNonMain.apply(intervalGrower)
        for name, iirPass in (strategy.getIirPasses()).items():
            print(name)
            iirPass.apply(intervalGrower)

        print("totalFrames/decoded/analyzed", frameCount, frameCache.statDecodedFrames, strategy.statAnalyzedFrames)

        return intervalGrower
