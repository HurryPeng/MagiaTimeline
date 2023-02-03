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

    def setFlag(self, index: AbstractFlagIndex, val: typing.Any):
        self.flags[index] = val

    def setFlags(self, map: typing.Dict[AbstractFlagIndex, typing.Any]):
        for k, v in map.items():
            self.flags[k] = v

    def setDebugFlag(self, *val: typing.Any):
        self.flags[self.flagIndexType.Debug] = val

    def getDebugFlag(self) -> typing.Any:
        return self.flags[self.flagIndexType.Debug]


    def toString(self) -> str:
        return "frame {} {}".format(self.index, formatTimestamp(self.timestamp))

    def toStringFull(self) -> str:
        return "frame {} {} {}".format(self.index, formatTimestamp(self.timestamp), self.flags)

class FPIR: # Frame Point Intermediate Representation
    def __init__(self, flagIndexType: typing.Type[AbstractFlagIndex]):
        self.flagIndexType: typing.Type[AbstractFlagIndex] = flagIndexType
        self.framePoints: typing.List[FramePoint] = []

    def accept(self, pazz: FPIRPass) -> typing.Any:
        # returns anything
        return pazz.apply(self)

    def genVirtualEnd(self) -> FramePoint:
        index: int = len(self.framePoints)
        timestamp: int = self.framePoints[-1].timestamp
        return FramePoint(self.flagIndexType, index, timestamp)

    def getFramePointsWithVirtualEnd(self) -> typing.List[FramePoint]:
        return self.framePoints + [self.genVirtualEnd()]

class FPIRPass(abc.ABC):
    @abc.abstractmethod
    def apply(self, fpir: FPIR):
        # returns anything
        pass

class FPIRPassRemoveNoise(FPIRPass):
    def __init__(self, flag: AbstractFlagIndex, minPositiveLength: int = 10, minNegativeLength: int = 2):
        self.flag: AbstractFlagIndex = flag
        self.minPositiveLength: int = minPositiveLength # set to 0 to disable removing positive noises
        self.minNegativeLength: int = minNegativeLength # set to 0 to disable removing negative noises

    def apply(self, fpir: FPIR):
        for id, framePoint in enumerate(fpir.framePoints):
            minLength: int = self.minNegativeLength
            if framePoint.flags[self.flag]:
                minLength = self.minPositiveLength
            l = id - minLength
            r = id + minLength
            if l < 0 or r > len(fpir.framePoints) - 1:
                continue
            length = 0
            for i in range(id - 1, l - 1, -1):
                if fpir.framePoints[i].flags[self.flag] != framePoint.flags[self.flag]:
                    break
                length += 1
            for i in range(id + 1, r + 1):
                if fpir.framePoints[i].flags[self.flag] != framePoint.flags[self.flag]:
                    break
                length += 1
            if length < minLength: # flip
                framePoint.flags[self.flag] = not framePoint.flags[self.flag]

class FPIRPassBuildIntervals(FPIRPass):
    def __init__(self, type: AbstractFlagIndex):
        self.type: AbstractFlagIndex = type

    def apply(self, fpir: FPIR) -> typing.List[Interval]:
        intervals: typing.List[Interval] = []
        lastBegin: int = 0
        state: bool = False
        for framePoint in fpir.getFramePointsWithVirtualEnd():
            if not state: # off -> on
                if framePoint.flags[self.type]:
                    state = True
                    lastBegin = framePoint.timestamp
            else: # on - > off
                if not framePoint.flags[self.type]:
                    state = False
                    intervals.append(Interval(lastBegin, framePoint.timestamp, self.type))
        return intervals

class Interval:
    def __init__(self, begin: int, end: int, type: AbstractFlagIndex):
        # TODO: Ultimately an Interval should be also able to hold a full set of flags
        self.begin: int = begin # timestamp
        self.end: int = end # timestamp
        self.type: AbstractFlagIndex = type

    def toAss(self, flag: str = "unknown") -> str:
        template = "Dialogue: 0,{},{},Default,,0,0,0,,SUBTITLE_{}_{}"
        sBegin = formatTimestamp(self.begin)
        sEnd = formatTimestamp(self.end)
        return template.format(sBegin, sEnd, self.type.name, flag)

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
    def __init__(self, fpir: FPIR, flagsToScan: typing.List[AbstractFlagIndex]):
        self.intervals: typing.List[Interval] = []
        for flag in flagsToScan:
            fpirPass = FPIRPassBuildIntervals(flag)
            self.intervals.extend(fpir.accept(fpirPass))
        self.sort()

    def accept(self, pazz: IIRPass):
        # returns anything
        return pazz.apply(self)

    def sort(self):
        self.intervals.sort(key=lambda interval: interval.begin)
        
    def toAss(self) -> str:
        ass = ""
        for id, interval in enumerate(self.intervals):
            ass += interval.toAss(str(id)) + "\n"
        return ass

class IIRPass(abc.ABC):
    @abc.abstractmethod
    def apply(self, iir: IIR) -> typing.Any:
        # returns anything
        pass

class IIRPassFillGap(IIRPass):
    def __init__(self, flag: AbstractFlagIndex, maxGap: int = 300):
        self.flag: AbstractFlagIndex = flag
        self.maxGap: int = maxGap # in millisecs
    
    def apply(self, iir: IIR):
        for id, interval in enumerate(iir.intervals):
            if interval.type != self.flag:
                continue
            otherId = id + 1
            while otherId < len(iir.intervals):
                otherInterval = iir.intervals[otherId]
                if otherInterval.type != self.flag:
                    otherId += 1
                    continue
                if interval.dist(otherInterval) > self.maxGap:
                    break
                if interval.dist(otherInterval) <= 0:
                    otherId += 1
                    continue
                mid = (interval.end + otherInterval.begin) // 2
                interval.end = mid
                otherInterval.begin = mid
                break
        iir.sort()
