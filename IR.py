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
    
    def setDebugFrame(self, debugFrame: cv.Mat):
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

class FPIRPass(abc.ABC):
    @abc.abstractmethod
    def apply(self, fpir: FPIR):
        # returns anything or nothing
        pass

class FPIRPassBuildIntervals(FPIRPass):
    @abc.abstractmethod
    def apply(self, fpir: FPIR) -> typing.List[Interval]:
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
            threshDist: float = 0.5, windowSize: int = 3):
        self.featFlag: AbstractFlagIndex = featFlag
        self.dstFlag: AbstractFlagIndex = dstFlag
        self.featOpMean: typing.Callable[[typing.List[typing.Any]], typing.Any] = featOpMean
        self.featOpDist: typing.Callable[[typing.Any, typing.Any], float] = featOpDist
        self.threshDist: float = threshDist
        self.windowSize: int = windowSize

    def apply(self, fpir: FPIR):
        framePointsExt = fpir.getFramePointsWithVirtualEnd(self.windowSize)
        for id, framePoint in enumerate(fpir.framePoints):
            featsToBeMeant = []
            for i in range(id + 1, id + 1 + self.windowSize):
                featsToBeMeant.append(framePointsExt[i].getFlag(self.featFlag))

            meanFeat = self.featOpMean(featsToBeMeant)
            dist = self.featOpDist(framePoint.getFlag(self.featFlag), meanFeat)

            if dist >= self.threshDist:
                framePoint.setFlag(self.dstFlag, False)
            else:
                framePoint.setFlag(self.dstFlag, True)

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

class FPIRPassBooleanBuildIntervals(FPIRPassBuildIntervals):
    def __init__(self, *flags: AbstractFlagIndex):
        self.flags: typing.Tuple[AbstractFlagIndex, ...] = flags

    def apply(self, fpir: FPIR) -> typing.List[Interval]:
        intervals: typing.List[Interval] = []
        lastBegin: typing.List[int] = [0] * len(self.flags)
        state: typing.List[bool] = [False] * len(self.flags)
        for framePoint in fpir.getFramePointsWithVirtualEnd():
            for i in range(len(state)):
                if not state[i]: # off -> on
                    if framePoint.getFlag(self.flags[i]):
                        state[i] = True
                        lastBegin[i] = framePoint.timestamp
                else: # on - > off
                    if not framePoint.getFlag(self.flags[i]):
                        state[i] = False
                        intervals.append(Interval(lastBegin[i], framePoint.timestamp, self.flags[i]))
        return intervals

class Interval:
    def __init__(self, begin: int, end: int, flag: AbstractFlagIndex):
        # TODO: Ultimately an Interval should be also able to hold a full set of flags
        self.begin: int = begin # timestamp
        self.end: int = end # timestamp
        self.flag: AbstractFlagIndex = flag

    def toAss(self, id: int = -1, track: int = 0) -> str:
        template = "Dialogue: 0,{},{},Default,,0,0,{},,Subtitle_{}_{}"
        sBegin = formatTimestamp(self.begin)
        sEnd = formatTimestamp(self.end)
        marginV: int = 100 + 50 * track
        return template.format(sBegin, sEnd, marginV, self.flag.name, id)

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

    def toAss(self, flag2Track: typing.Dict[AbstractFlagIndex, int] = {}) -> str:
        lines: typing.List[str] = []
        trackCounter: typing.Dict[int, int] = {}
        for _, interval in enumerate(self.intervals):
            track: int = flag2Track.get(interval.flag, 0)
            id = trackCounter.get(interval.flag, 0)
            trackCounter[interval.flag] = id + 1
            lines.append(interval.toAss(id, track) + "\n")
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
            if interval.flag != self.flag:
                continue
            otherId = id + 1
            while otherId < len(iir.intervals):
                otherInterval = iir.intervals[otherId]
                if otherInterval.flag != self.flag:
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
