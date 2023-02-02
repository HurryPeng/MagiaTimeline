from __future__ import annotations
import abc
import typing
import enum

from Util import *

class SubtitleType(enum.IntEnum):
    DIALOG = 0
    BLACKSCREEN = 1
    WHITESCREEN = 2
    CGSUB = 3

    @staticmethod
    def num() -> int:
        return len(SubtitleType.__members__)

class FramePoint:
    def __init__(self, index: int, timestamp: int, flags: typing.List[bool]):
        self.index: int = index
        self.timestamp: int = timestamp
        if len(flags) != SubtitleType.num():
            raise Exception("len(flags) != SubtitleTypes.num()")
        self.flags: typing.List[bool] = flags

    def toString(self) -> str:
        return "frame {} {}".format(self.index, formatTimestamp(self.timestamp))

    def toStringFull(self) -> str:
        return "frame {} {} {}".format(self.index, formatTimestamp(self.timestamp), self.flags)

class FPIR: # Frame Point Intermediate Representation
    def __init__(self):
        self.framePoints: typing.List[FramePoint] = []

    def accept(self, pazz: FPIRPass) -> typing.Any:
        # returns anything
        return pazz.apply(self)

    def genVirtualEnd(self) -> FramePoint:
        index: int = len(self.framePoints)
        timestamp: int = self.framePoints[-1].timestamp
        flags = [False] * SubtitleType.num()
        return FramePoint(index, timestamp, flags)

    def getFramePointsWithVirtualEnd(self) -> typing.List[FramePoint]:
        return self.framePoints + [self.genVirtualEnd()]

class FPIRPass(abc.ABC):
    @abc.abstractmethod
    def apply(self, fpir: FPIR):
        # returns anything
        pass

class FPIRPassRemoveNoise(FPIRPass):
    def __init__(self, type: SubtitleType, minPositiveLength: int = 10, minNegativeLength: int = 2):
        self.type: SubtitleType = type
        self.minPositiveLength: int = minPositiveLength # set to 0 to disable removing positive noises
        self.minNegativeLength: int = minNegativeLength # set to 0 to disable removing negative noises

    def apply(self, fpir: FPIR):
        for id, framePoint in enumerate(fpir.framePoints):
            minLength: int = self.minNegativeLength
            if framePoint.flags[self.type]:
                minLength = self.minPositiveLength
            l = id - minLength
            r = id + minLength
            if l < 0 or r > len(fpir.framePoints) - 1:
                continue
            length = 0
            for i in range(id - 1, l - 1, -1):
                if fpir.framePoints[i].flags[self.type] != framePoint.flags[self.type]:
                    break
                length += 1
            for i in range(id + 1, r + 1):
                if fpir.framePoints[i].flags[self.type] != framePoint.flags[self.type]:
                    break
                length += 1
            if length < minLength: # flip
                framePoint.flags[self.type] = not framePoint.flags[self.type]

class FPIRPassBuildIntervals(FPIRPass):
    def __init__(self, type: SubtitleType):
        self.type: SubtitleType = type

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
    def __init__(self, begin: int, end: int, type: SubtitleType):
        self.begin: int = begin # timestamp
        self.end: int = end # timestamp
        self.type: SubtitleType = type

    def toAss(self, tag: str = "unknown") -> str:
        template = "Dialogue: 0,{},{},Default,,0,0,0,,SUBTITLE_{}_{}"
        sBegin = formatTimestamp(self.begin)
        sEnd = formatTimestamp(self.end)
        return template.format(sBegin, sEnd, self.type.name, tag)

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
    def __init__(self, fpir: FPIR):
        self.intervals: typing.List[Interval] = []
        for type in SubtitleType:
            fpirPass = FPIRPassBuildIntervals(type)
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
    def __init__(self, type: SubtitleType, maxGap: int = 300):
        self.type: SubtitleType = type
        self.maxGap: int = maxGap # in millisecs
    
    def apply(self, iir: IIR):
        for id, interval in enumerate(iir.intervals):
            if interval.type != self.type:
                continue
            otherId = id + 1
            while otherId < len(iir.intervals):
                otherInterval = iir.intervals[otherId]
                if otherInterval.type != self.type:
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