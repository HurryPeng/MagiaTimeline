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

    def getFlag2Track(self) -> typing.Dict[AbstractFlagIndex, int]:
        return {}
