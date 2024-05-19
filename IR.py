from __future__ import annotations
import abc
import typing

from Util import *
from AbstractFlagIndex import *

class FramePoint:
    def __init__(self, flagIndexType: typing.Type[AbstractFlagIndex], index: int, timestamp: int):
        self.flagIndexType: typing.Type[AbstractFlagIndex] = flagIndexType
        self.index: int = index
        self.timestamp: int = timestamp
        self.flags: typing.List[typing.Any] = self.flagIndexType.getDefaultFlags()
        self.debugFrame: cv.Mat | None = None

    def setFlag(self, index: AbstractFlagIndex, val: typing.Any):
        self.flags[index] = val

    def getFlag(self, index: AbstractFlagIndex) -> typing.Any:
        return self.flags[index]

    def setFlags(self, map: typing.Dict[AbstractFlagIndex, typing.Any]):
        for k, v in map.items():
            self.flags[k] = v

    def setDebugFlag(self, *val: typing.Any):
        self.flags[self.flagIndexType.Debug] = val

    def getDebugFlag(self) -> typing.Any:
        return self.flags[self.flagIndexType.Debug]
    
    def setDebugFrame(self, debugFrame):
        self.debugFrame = debugFrame

    def setDebugFrameHSV(self, debugFrame: cv.Mat):
        self.setDebugFrame(cv.cvtColor(debugFrame, cv.COLOR_HSV2BGR))
    
    def clearDebugFrame(self):
        self.debugFrame = None

    def getDebugFrame(self) -> cv.Mat | None:
        return self.debugFrame

    def toString(self) -> str:
        return "frame {} {}".format(self.index, formatTimestamp(self.timestamp))

    def toStringFull(self) -> str:
        return "frame {} {} {}".format(self.index, formatTimestamp(self.timestamp), self.flags)

class FPIR: # Frame Point Intermediate Representation
    def __init__(self, flagIndexType: typing.Type[AbstractFlagIndex]):
        self.flagIndexType: typing.Type[AbstractFlagIndex] = flagIndexType
        self.framePoints: typing.List[FramePoint] = []

    def genVirtualEnd(self) -> FramePoint:
        index: int = len(self.framePoints)
        timestamp: int = self.framePoints[-1].timestamp
        return FramePoint(self.flagIndexType, index, timestamp)

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
            featOpMean: typing.Callable[[typing.List[typing.Any]], typing.Any], \
            featOpDist: typing.Callable[[typing.Any, typing.Any], float], \
            threshDist: float = 0.5, windowSize: int = 3, inverse: bool = False):
        self.featFlag: AbstractFlagIndex = featFlag
        self.dstFlag: AbstractFlagIndex = dstFlag
        self.featOpMean: typing.Callable[[typing.List[typing.Any]], typing.Any] = featOpMean
        self.featOpDist: typing.Callable[[typing.Any, typing.Any], float] = featOpDist
        self.threshDist: float = threshDist
        self.windowSize: int = windowSize
        self.inverse: bool = inverse

    def apply(self, fpir: FPIR):
        framePointsExt = fpir.getFramePointsWithVirtualEnd(self.windowSize)
        for id, framePoint in enumerate(fpir.framePoints):
            featsToBeMeant = []
            for i in range(id + 1, id + 1 + self.windowSize):
                featsToBeMeant.append(framePointsExt[i].getFlag(self.featFlag))

            meanFeat = self.featOpMean(featsToBeMeant)
            dist = self.featOpDist(framePoint.getFlag(self.featFlag), meanFeat)

            if dist >= self.threshDist:
                framePoint.setFlag(self.dstFlag, not self.inverse)
            else:
                framePoint.setFlag(self.dstFlag, self.inverse)

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
        for framePoint in fpir.getFramePointsWithVirtualEnd():
            for s in range(len(state)):
                if not state[s]: # off -> on
                    if framePoint.getFlag(self.flags[s]):
                        state[s] = True
                        lastBegin[s] = framePoint.index
                else: # on - > off
                    if not framePoint.getFlag(self.flags[s]):
                        state[s] = False
                        intervals.append(Interval(fpir.flagIndexType, self.flags[s], fpir.framePoints[lastBegin[s]].timestamp, framePoint.timestamp, fpir.framePoints[lastBegin[s] : framePoint.index]))
        return intervals

class Interval:
    def __init__(self, flagIndexType: typing.Type[AbstractFlagIndex], mainFlag: AbstractFlagIndex, begin: int, end: int, framePoints: typing.List[FramePoint]):
        self.flagIndexType: typing.Type[AbstractFlagIndex] = flagIndexType
        self.mainFlag: AbstractFlagIndex = mainFlag
        self.framePoints: typing.List[FramePoint] = framePoints
        # begin and end are not promised to align with underlying framePoints
        self.begin: int = begin # timestamp
        self.end: int = end # timestamp
        self.style: str = "Default"
        self.flags: typing.List[typing.Any] = self.flagIndexType.getDefaultFlags()

    def toAss(self, id: int = -1) -> str:
        template = "Dialogue: 0,{},{},{},,0,0,100,,Subtitle_{}_{}"
        sBegin = formatTimestamp(self.begin)
        sEnd = formatTimestamp(self.end)
        return template.format(sBegin, sEnd, self.style, self.mainFlag.name, id)

    def dist(self, other: Interval) -> int:
        l = self
        r = other
        if self.begin > other.begin:
            l = other
            r = self
        return r.begin - l.end

    def intersects(self, other: Interval) -> bool:
        return self.dist(other) < 0

    def touches(self, other: Interval) -> bool:
        return self.dist(other) == 0

class IIR: # Interval Intermediate Representation
    def __init__(self, flagIndexType: typing.Type[AbstractFlagIndex]):
        self.flagIndexType: typing.Type[AbstractFlagIndex] = flagIndexType
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
                if interval.dist(otherInterval) > self.maxGap:
                    break
                if interval.dist(otherInterval) <= 0:
                    otherId += 1
                    continue
                mid = int(interval.end * (1.0 - self.meetPoint) + otherInterval.begin * self.meetPoint)
                interval.end = mid
                otherInterval.begin = mid
                break
        iir.sort()

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
            if abs(dist) <= self.maxGap:
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
            if abs(dist) <= self.maxGap:
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
