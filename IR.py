from __future__ import annotations
import abc
import typing
import fractions

from Util import *
from AbstractFlagIndex import *

class FramePoint:
    def __init__(self, flagIndexType: typing.Type[AbstractFlagIndex], timestamp: int, timeBase: fractions.Fraction):
        self.flagIndexType: typing.Type[AbstractFlagIndex] = flagIndexType
        self.timestamp: int = timestamp
        self.timeBase: fractions.Fraction = timeBase
        self.flags: typing.List[typing.Any] = self.flagIndexType.getDefaultFlags()
        self.debugFrame: cv.Mat | None = None

    def setFlag(self, index: AbstractFlagIndex, val: typing.Any, inDiskCache: bool = False):
        if inDiskCache and val is not None:
            val = DiskCacheHandle(val)
        self.flags[index] = val

    def getFlag(self, index: AbstractFlagIndex) -> typing.Any:
        val = self.flags[index]
        if isinstance(val, DiskCacheHandle):
            return val.get()
        return val

    def setDebugFlag(self, *val: typing.Any):
        self.flags[self.flagIndexType.Debug()] = val

    def getDebugFlag(self) -> typing.Any:
        return self.flags[self.flagIndexType.Debug()]
    
    def setDebugFrame(self, debugFrame):
        self.debugFrame = debugFrame

    def setDebugFrameHSV(self, debugFrame: cv.Mat):
        self.setDebugFrame(cv.cvtColor(debugFrame, cv.COLOR_HSV2BGR))
    
    def clearDebugFrame(self):
        self.debugFrame = None

    def getDebugFrame(self) -> cv.Mat | None:
        return self.debugFrame
    
    def timeString(self) -> str:
        return formatTimestamp(self.timeBase, self.timestamp)

    def toString(self) -> str:
        return "frame {}".format(formatTimestamp(self.timeBase, self.timestamp))

    def toStringFull(self) -> str:
        return "frame {} {}".format(formatTimestamp(self.timeBase, self.timestamp), self.flags)

class FPIR: # Frame Point Intermediate Representation
    def __init__(self, flagIndexType: typing.Type[AbstractFlagIndex], sampleRate: int, timeBase: fractions.Fraction):
        self.flagIndexType: typing.Type[AbstractFlagIndex] = flagIndexType
        self.framePoints: typing.List[FramePoint] = []
        self.sampleRate: int = sampleRate
        self.timeBase: fractions.Fraction = timeBase

    def genVirtualEnd(self) -> FramePoint:
        index: int = len(self.framePoints)
        timestamp: int = self.framePoints[-1].timestamp
        return FramePoint(self.flagIndexType, timestamp, self.timeBase)

    def getFramePointsWithVirtualEnd(self, length: int = 1) -> typing.List[FramePoint]:
        return self.framePoints + [self.genVirtualEnd()] * length

    def toStringFull(self) -> str:
        lines = []
        for framePoint in self.framePoints:
            lines.append(framePoint.toStringFull() + "\n")
        return "".join(lines)

class FPIRPass(abc.ABC):
    @abc.abstractmethod
    def apply(self, fpir: FPIR):
        # returns anything or nothing
        pass

class FPIRPassBooleanRemoveNoise(FPIRPass):
    def __init__(self, flag: AbstractFlagIndex, trueToFalse: bool = True, minLength: int = 10):
        self.flag: AbstractFlagIndex = flag
        self.trueToFalse: bool = trueToFalse
        self.minLength: int = minLength

    def apply(self, fpir: FPIR):
        for id, framePoint in enumerate(fpir.framePoints):
            if framePoint.getFlag(self.flag) != self.trueToFalse:
                continue
            l = id - self.minLength
            r = id + self.minLength
            if l < 0 or r > len(fpir.framePoints) - 1:
                continue
            length = 1
            for i in range(id - 1, l - 1, -1):
                if fpir.framePoints[i].getFlag(self.flag) != framePoint.getFlag(self.flag):
                    break
                length += 1
            for i in range(id + 1, r + 1):
                if fpir.framePoints[i].getFlag(self.flag) != framePoint.getFlag(self.flag):
                    break
                length += 1
            if length < self.minLength: # flip
                framePoint.setFlag(self.flag, not framePoint.getFlag(self.flag))

class FPIRPassDetectFeatureJump(FPIRPass):
    def __init__(self, featFlag: AbstractFlagIndex, dstFlag: AbstractFlagIndex, \
            featOpMean: typing.Callable[[typing.List[typing.Any]], typing.Any] = lambda feats : np.mean(feats, axis=0), \
            featOpDist: typing.Callable[[typing.Any, typing.Any], float] = lambda lhs, rhs : np.linalg.norm(lhs - rhs), \
            threshDist: float = 0.5, \
            featOpStd: typing.Callable[[typing.List[typing.Any]], float] = lambda feats: np.mean(np.std(feats, axis=0)), \
            threshStd: typing.Optional[float] = 0.0, \
            windowSize: int = 3, inverse: bool = False
            ):
        self.featFlag: AbstractFlagIndex = featFlag
        self.dstFlag: AbstractFlagIndex = dstFlag
        self.featOpMean: typing.Callable[[typing.List[typing.Any]], typing.Any] = featOpMean
        self.featOpDist: typing.Callable[[typing.Any, typing.Any], float] = featOpDist
        self.threshDist: float = threshDist
        self.windowSize: int = windowSize
        self.inverse: bool = inverse
        self.featOpStd: typing.Callable[[typing.List[typing.Any]], float] = featOpStd
        self.threshStd: float = threshStd

    def apply(self, fpir: FPIR):
        framePointsExt = fpir.getFramePointsWithVirtualEnd(self.windowSize)
        for id, framePoint in enumerate(fpir.framePoints):
            featsToBeMeant = []
            for i in range(id + 1, id + 1 + self.windowSize):
                featsToBeMeant.append(framePointsExt[i].getFlag(self.featFlag))

            meanFeat = self.featOpMean(featsToBeMeant)
            dist = self.featOpDist(framePoint.getFlag(self.featFlag), meanFeat)

            if self.threshStd > 0.0:
                stdFeat: float = self.featOpStd(featsToBeMeant)
                if stdFeat > self.threshStd:
                    continue

            if dist >= self.threshDist:
                framePoint.setFlag(self.dstFlag, not self.inverse)
            else:
                framePoint.setFlag(self.dstFlag, self.inverse)

class FPIRPassShift(FPIRPass):
    def __init__(self, tgtFlag: AbstractFlagIndex, refFlag: AbstractFlagIndex, shift: int, padding: typing.Any):
        self.tgtFlag: AbstractFlagIndex = tgtFlag
        self.refFlag: AbstractFlagIndex = refFlag
        self.shift: int = shift
        self.padding: typing.Any = padding

    def apply(self, fpir: FPIR):
        for iTgt, framePoint in enumerate(fpir.framePoints):
            iSrc = iTgt - self.shift
            if iSrc < 0 or iSrc >= len(fpir.framePoints):
                framePoint.setFlag(self.tgtFlag, self.padding)
            else:
                framePoint.setFlag(self.tgtFlag, fpir.framePoints[iSrc].getFlag(self.refFlag))

class FPIRPassFunctional(FPIRPass):
    def __init__(self, func: typing.Callable[[FPIR], typing.Any]):
        self.func = func

    def apply(self, fpir: FPIR):
        return self.func(fpir)

class FPIRPassFramewiseFunctional(FPIRPass):
    def __init__(self, func: typing.Callable[[FramePoint], typing.Any]):
        self.func = func

    def apply(self, fpir: FPIR):
        for id, framePoint in enumerate(fpir.framePoints):
            self.func(framePoint)

class FPIRPassBuildIntervals(FPIRPass):
    @abc.abstractmethod
    def apply(self, fpir: FPIR) -> typing.List[Interval]:
        pass

class FPIRPassBooleanBuildIntervals(FPIRPassBuildIntervals):
    def __init__(self, *flags: AbstractFlagIndex):
        self.flags: typing.Tuple[AbstractFlagIndex, ...] = flags

    def apply(self, fpir: FPIR) -> typing.List[Interval]:
        intervals: typing.List[Interval] = []
        lastBegin: typing.List[int] = [0] * len(self.flags)
        state: typing.List[bool] = [False] * len(self.flags)
        for i, framePoint in enumerate(fpir.getFramePointsWithVirtualEnd()):
            for s in range(len(state)):
                if not state[s]: # off -> on
                    if framePoint.getFlag(self.flags[s]):
                        state[s] = True
                        lastBegin[s] = i
                else: # on - > off
                    if not framePoint.getFlag(self.flags[s]):
                        state[s] = False
                        intervals.append(Interval(fpir.flagIndexType, self.flags[s], fpir.framePoints[lastBegin[s]].timestamp, framePoint.timestamp, framePoint.timeBase, fpir.framePoints[lastBegin[s] : i]))
        return intervals

class Interval:
    def __init__(
            self,
            flagIndexType: typing.Type[AbstractFlagIndex],
            mainFlag: AbstractFlagIndex,
            begin: int,
            end: int,
            timeBase: fractions.Fraction,
            framePoints: typing.List[FramePoint] = [],
            flags: typing.List[typing.Any] = []
        ):
        self.flagIndexType: typing.Type[AbstractFlagIndex] = flagIndexType
        self.mainFlag: AbstractFlagIndex = mainFlag
        self.framePoints: typing.List[FramePoint] = framePoints
        # begin and end are not promised to align with underlying framePoints after applying IIRPass
        self.begin: int = begin # timestamp
        self.end: int = end # timestamp
        self.timeBase: fractions.Fraction = timeBase
        self.style: str = "Default"
        self.text: str = ""
        self.flags: typing.List[typing.Any] = flags
        if len(self.flags) == 0:
            self.flags = self.flagIndexType.getDefaultFlags()

    def getName(self, id: int = -1) -> str:
        return f"Subtitle_{self.mainFlag.name}_{id}"
    
    def setFlag(self, index: AbstractFlagIndex, val: typing.Any, inDiskCache: bool = False):
        if inDiskCache and val is not None:
            val = DiskCacheHandle(val)
        self.flags[index] = val

    def getFlag(self, index: AbstractFlagIndex) -> typing.Any:
        val = self.flags[index]
        if isinstance(val, DiskCacheHandle):
            return val.get()
        return val

    def toAss(self, id: int = -1) -> str:
        template = "Dialogue: 0,{},{},{},,0,0,0,,{}"
        sBegin = formatTimestamp(self.timeBase, self.begin)
        sEnd = formatTimestamp(self.timeBase, self.end)
        text = self.text
        if text == "":
            text = self.getName(id)
        return template.format(sBegin, sEnd, self.style, text)
    
    def timeString(self) -> str:
        return "[{}, {})".format(formatTimestamp(self.timeBase, self.begin), formatTimestamp(self.timeBase, self.end))
    
    def timeStringBegin(self) -> str:
        return formatTimestamp(self.timeBase, self.begin)
    
    def timeStringEnd(self) -> str:
        return formatTimestamp(self.timeBase, self.end)

    def dist(self, other: Interval) -> int:
        l = self
        r = other
        if self.begin > other.begin:
            l = other
            r = self
        return r.begin - l.end
    
    def distFramePoint(self, framePoint: FramePoint) -> int:
        if framePoint.timestamp < self.begin:
            return self.begin - framePoint.timestamp
        if framePoint.timestamp > self.end:
            return framePoint.timestamp - self.end
        return 0

    def intersects(self, other: Interval) -> bool:
        return self.dist(other) < 0

    def touches(self, other: Interval) -> bool:
        return self.dist(other) == 0
    
    def getMidPoint(self) -> int:
        return int((self.begin + self.end) // 2)
    
    def merge(self, other: Interval) -> Interval:
        return Interval(self.flagIndexType, self.mainFlag, min(self.begin, other.begin), max(self.end, other.end), self.timeBase, self.framePoints + other.framePoints, self.flags)

class IIR: # Interval Intermediate Representation
    def __init__(self, flagIndexType: typing.Type[AbstractFlagIndex], fps: fractions.Fraction, timeBase: fractions.Fraction):
        self.flagIndexType: typing.Type[AbstractFlagIndex] = flagIndexType
        self.fps: fractions.Fraction = fps
        self.timeBase: fractions.Fraction = timeBase
        self.intervals: typing.List[Interval] = []

    def appendFromFpir(self, fpir: FPIR, fpirPassBuildIntervals: FPIRPassBuildIntervals):
        # does not guarantee that intervals are in order after appending
        self.intervals += fpirPassBuildIntervals.apply(fpir)

    def sort(self):
        self.intervals.sort(key=lambda interval : interval.begin)

    def toAss(self) -> str:
        lines: typing.List[str] = []
        mainFlagCounter: typing.Dict[int, int] = {}
        for _, interval in enumerate(self.intervals):
            id = mainFlagCounter.get(interval.mainFlag, 0)
            mainFlagCounter[interval.mainFlag] = id + 1
            lines.append(interval.toAss(id) + "\n")
        return "".join(lines)
    
    def getMidpoints(self) -> typing.List[typing.Tuple[str, int]]:
        midpoints: typing.List[typing.Tuple[str, int]] = []
        mainFlagCounter: typing.Dict[int, int] = {}
        for _, interval in enumerate(self.intervals):
            id = mainFlagCounter.get(interval.mainFlag, 0)
            mainFlagCounter[interval.mainFlag] = id + 1
            midpoints.append((interval.getName(id), interval.getMidPoint()))
        return midpoints
    
    def collectIfMainFlag(self, mainFlag: AbstractFlagIndex) -> typing.List[Interval]:
        return [interval for interval in self.intervals if interval.mainFlag == mainFlag]
    
    def ms2Timestamp(self, ms: int) -> int:
        return ms2Timestamp(ms, self.timeBase)

class IIRPass(abc.ABC):
    @abc.abstractmethod
    def apply(self, iir: IIR) -> typing.Any:
        # returns anything
        pass

class IIRPassFillGap(IIRPass):
    def __init__(self, flag: AbstractFlagIndex, maxGap: int = 300, meetPoint: float = 0.5):
        self.flag: AbstractFlagIndex = flag
        self.maxGap: int = maxGap # in millisecs
        self.meetPoint: float = meetPoint
    
    def apply(self, iir: IIR):
        for id, interval in enumerate(iir.intervals):
            if interval.mainFlag != self.flag:
                continue
            otherId = id + 1
            while otherId < len(iir.intervals):
                otherInterval = iir.intervals[otherId]
                if otherInterval.mainFlag != self.flag:
                    otherId += 1
                    continue
                if interval.dist(otherInterval) > iir.ms2Timestamp(self.maxGap):
                    break
                if interval.dist(otherInterval) <= 0:
                    otherId += 1
                    continue
                mid = int(interval.end * (1.0 - self.meetPoint) + otherInterval.begin * self.meetPoint)
                interval.end = mid
                otherInterval.begin = mid
                break
        iir.sort()

class IIRPassExtend(IIRPass):
    def __init__(self, flag: AbstractFlagIndex, front: int = 0, back: int = 0):
        self.flag: AbstractFlagIndex = flag
        self.front: int = front # in millisecs
        self.back: int = back # in millisecs

    def apply(self, iir: IIR):
        # Assert sorted
        for id, interval in enumerate(iir.intervals):
            if interval.mainFlag != self.flag:
                continue
            if id == 0:
                interval.begin = max(interval.begin - iir.ms2Timestamp(self.front), 0)
            else:
                interval.begin = max(interval.begin - iir.ms2Timestamp(self.front), iir.intervals[id - 1].end)
            if id == len(iir.intervals) - 1:
                interval.end += iir.ms2Timestamp(self.back)
            else:
                interval.end = min(interval.end + iir.ms2Timestamp(self.back), iir.intervals[id + 1].begin)

class IIRPassAlign(IIRPass):
    def __init__(self, tgtFlag: AbstractFlagIndex, refFlag: AbstractFlagIndex, maxGap: int = 300):
        self.tgtFlag: AbstractFlagIndex = tgtFlag
        self.refFlag: AbstractFlagIndex = refFlag
        self.maxGap: int = maxGap # in millisecs
    
    def apply(self, iir: IIR):
        refPoints: typing.List[int] = []
        for _, interval in enumerate(iir.intervals):
            if interval.mainFlag != self.refFlag:
                continue
            refPoints.append(interval.begin)
            refPoints.append(interval.end)
        refPoints.sort()
        if len(refPoints) == 0:
            return

        for _, interval in enumerate(iir.intervals):
            if interval.mainFlag != self.tgtFlag:
                continue
            
            r = 0
            while r < len(refPoints) and refPoints[r] < interval.begin:
                r = r + 1
            l = max(0, r - 1)
            r = min(r, len(refPoints) - 1)
            lDist = refPoints[l] - interval.begin # <= 0
            rDist = refPoints[r] - interval.begin # >= 0
            dist = lDist
            if rDist < -lDist:
                dist = rDist
            if abs(dist) <= iir.ms2Timestamp(self.maxGap):
                interval.begin += dist
            
            r = 0
            while r < len(refPoints) and refPoints[r] < interval.end:
                r = r + 1
            l = max(0, r - 1)
            r = min(r, len(refPoints) - 1)
            lDist = refPoints[l] - interval.end # <= 0
            rDist = refPoints[r] - interval.end # >= 0
            dist = lDist
            if rDist < -lDist:
                dist = rDist
            if abs(dist) <= iir.ms2Timestamp(self.maxGap):
                interval.end += dist
            
        iir.sort()

class IIRPassFunctional(IIRPass):
    def __init__(self, func: typing.Callable[[IIR], typing.Any]):
        self.func = func

    def apply(self, iir: IIR):
        return self.func(iir)

class IIRPassIntervalwiseFunctional(IIRPass):
    def __init__(self, func: typing.Callable[[Interval], typing.Any]):
        self.func = func

    def apply(self, iir: IIR):
        for id, interval in enumerate(iir.intervals):
            self.func(interval)

class IIRPassOffset(IIRPass):
    def __init__(self, offset: int):
        self.offset: int = offset

    def apply(self, iir: IIR):
        for id, interval in enumerate(iir.intervals):
            interval.begin += self.offset
            interval.end += self.offset

class IIRPassRemovePredicate(IIRPass):
    def __init__(self, pred: typing.Callable[[Interval], bool]):
        self.pred = pred

    def apply(self, iir: IIR):
        iir.intervals = [interval for interval in iir.intervals if not self.pred(interval)]

class IIRPassDenoise(IIRPass):
    def __init__(self, flag: AbstractFlagIndex, minTime: int):
        self.flag: AbstractFlagIndex = flag
        self.minTime: int = minTime

    def apply(self, iir: IIR):
        iir.intervals = [interval for interval in iir.intervals if not (interval.mainFlag == self.flag and interval.end - interval.begin < iir.ms2Timestamp(self.minTime))]

class IIRPassMerge(IIRPass):
    def __init__(self, pred: typing.Callable[[IIR, Interval, Interval], bool], debug: bool = False):
        self.debug: bool = debug
        self.pred = pred

    def apply(self, iir: IIR):
        newIntervals: typing.List[Interval] = []
        for i in range(len(iir.intervals)):
            if len(newIntervals) == 0:
                newIntervals.append(iir.intervals[i])
                continue
            if self.pred(iir, newIntervals[-1], iir.intervals[i]):
                if self.debug:
                    print(f"Merging {newIntervals[-1].timeString()} and {iir.intervals[i].timeString()}")
                newIntervals[-1] = newIntervals[-1].merge(iir.intervals[i])
            else:
                newIntervals.append(iir.intervals[i])
        iir.intervals = newIntervals