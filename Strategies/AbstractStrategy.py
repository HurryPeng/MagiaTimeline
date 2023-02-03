from __future__ import annotations
import abc
import typing

from Rectangle import *
from AbstractFlagIndex import *
from IR import *

class AbstractStrategy(abc.ABC):
    @classmethod
    @abc.abstractmethod
    def getFlagIndexType(cls) -> typing.Type[AbstractFlagIndex]:
        pass

    @abc.abstractmethod
    def getRectangles(self) -> typing.List[AbstractRectangle]:
        pass

    @abc.abstractmethod
    def getCvPasses(self) -> typing.List[typing.Callable[[cv.Mat, FramePoint], bool]]:
        pass
