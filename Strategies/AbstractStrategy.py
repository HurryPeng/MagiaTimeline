from __future__ import annotations
import abc
import typing
import collections

from Rectangle import *
from AbstractFlagIndex import *
from IR import *

class AbstractStrategy(abc.ABC):
    def __init__(self, contentRect: AbstractRectangle) -> None:
        self.contentRect = contentRect

    @classmethod
    @abc.abstractmethod
    def getFlagIndexType(cls) -> typing.Type[AbstractFlagIndex]:
        pass

    def getContentRect(self) -> AbstractRectangle:
        return self.contentRect
    
    def getStyles(self) -> typing.List[str]:
        return []

class AbstractFramewiseStrategy(AbstractStrategy, abc.ABC):
    @abc.abstractmethod
    def getRectangles(self) -> collections.OrderedDict[str, AbstractRectangle]:
        pass

    @abc.abstractmethod
    def getCvPasses(self) -> typing.List[typing.Callable[[cv.Mat, FramePoint], bool]]:
        pass

    @abc.abstractmethod
    def getFpirPasses(self) -> collections.OrderedDict[str, FPIRPass]:
        pass

    @abc.abstractmethod
    def getFpirToIirPasses(self) -> collections.OrderedDict[str, FPIRPassBuildIntervals]:
        pass

    @abc.abstractmethod
    def getIirPasses(self) -> collections.OrderedDict[str, IIRPass]:
        pass

class AbstractSpeculativeStrategy(AbstractStrategy, abc.ABC):

    @classmethod
    @abc.abstractmethod
    def getMainFlagIndex(cls) -> AbstractFlagIndex:
        pass

    @classmethod
    @abc.abstractmethod
    def getFeatureFlagIndex(cls) -> AbstractFlagIndex:
        pass

    @classmethod
    @abc.abstractmethod
    def isEmptyFeature(cls, feature) -> bool:
        pass

    def __init__(self) -> None:
        self.statAnalyzedFrames: int = 0

    @abc.abstractmethod
    def getCvPasses(self) -> typing.List[typing.Callable[[cv.Mat, FramePoint], bool]]:
        pass

    @abc.abstractmethod
    def getSpecIirPasses(self) -> collections.OrderedDict[str, IIRPass]:
        pass

    @abc.abstractmethod
    def decideFeatureMerge(self, oldFeatures: typing.List[typing.Any], newFeatures: typing.List[typing.Any]) -> bool:
        pass

    @abc.abstractmethod
    def aggregateFeatures(self, features: typing.List[typing.Any]) -> typing.Any:
        pass

    def genFramePoint(self, frame: cv.Mat, timestamp: int, timeBase: fractions.Fraction) -> FramePoint:
        self.statAnalyzedFrames += 1
        framePoint = FramePoint(self.getFlagIndexType(), timestamp, timeBase)
        for cvPass in self.getCvPasses():
            cvPass(frame, framePoint)
        framePoint.clearDebugFrame()
        return framePoint
    
    def getStatAnalyzedFrames(self) -> int:
        return self.statAnalyzedFrames

class AbstractExtraJobStrategy(abc.ABC):
    @abc.abstractmethod
    def cutExtraJobFrame(self, frame: cv.Mat) -> cv.Mat:
        pass

    @abc.abstractmethod
    def getExtraJobFrameFlagIndex(self) -> AbstractFlagIndex:
        pass
