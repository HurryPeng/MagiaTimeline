from __future__ import annotations
import abc
import typing
import collections

from Rectangle import *
from AbstractFlagIndex import *
from IR import *

class AbstractStrategy(abc.ABC):
    @classmethod
    @abc.abstractmethod
    def getFlagIndexType(cls) -> typing.Type[AbstractFlagIndex]:
        pass

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

    def getStyles(self) -> typing.List[str]:
        return []
    
    def ocrPass(self, frame: cv.Mat) -> typing.Tuple[cv.Mat, cv.Mat]: # (ocrFrame, debugFrame)
        return frame, None

class SpeculativeStrategy(abc.ABC):
    @abc.abstractmethod
    def genFeature(self, frame: cv.Mat) -> typing.Any:
        pass

    @abc.abstractmethod
    def decideFeatureMerge(self, oldFeatures: typing.List[typing.Any], newFeature: typing.Any) -> bool:
        pass

    @abc.abstractmethod
    def getIirPasses(self) -> collections.OrderedDict[str, IIRPass]:
        pass

class OcrStrategy(abc.ABC):
    @abc.abstractmethod
    def cutOcrFrame(self, frame: cv.Mat) -> cv.Mat:
        pass

    @abc.abstractmethod
    def cutCleanOcrFrame(self, frame: cv.Mat) -> cv.Mat:
        pass