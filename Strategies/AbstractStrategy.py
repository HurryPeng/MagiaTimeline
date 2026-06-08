from __future__ import annotations
import abc
import typing
import collections

from Rectangle import *
from AbstractFlagIndex import *
from IR import *


class ExtraJobFrameKey(AttachmentKey):
    """Attachment key for the image frame cut by the engine for extra job passes."""
    pass

class AbstractStrategy(abc.ABC):
    # Common required attributes; all strategies must set in __init__
    rectangles: typing.Optional[collections.OrderedDict[str, AbstractRectangle]] = None
    cvPasses: typing.Optional[typing.List[typing.Callable[[cv.Mat, FramePoint], bool]]] = None

    def __init__(self, contentRect: AbstractRectangle) -> None:
        self.contentRect = contentRect

    def _ensureNonNull(self, attr: str):
        """Validate that a required attribute has been set (not None)."""
        val = getattr(self, attr)
        if val is None:
            raise AttributeError(
                f"{self.__class__.__name__} has not set self.{attr}. "
                f"Assign it in __init__."
            )
        return val

    @classmethod
    @abc.abstractmethod
    def getFlagIndexType(cls) -> typing.Type[AbstractFlagIndex]:
        pass

    def getContentRect(self) -> AbstractRectangle:
        return self.contentRect

    def getDebugString(self) -> str:
        return ""

    def getRectangles(self) -> collections.OrderedDict[str, AbstractRectangle]:
        return self._ensureNonNull('rectangles')

    def getCvPasses(self) -> typing.List[typing.Callable[[cv.Mat, FramePoint], bool]]:
        return self._ensureNonNull('cvPasses')

class AbstractFramewiseStrategy(AbstractStrategy, abc.ABC):
    # Framewise-specific required attributes
    fpirPasses: typing.Optional[collections.OrderedDict[str, FPIRPass]] = None
    fpirToIirPasses: typing.Optional[collections.OrderedDict[str, FPIRPassBuildIntervals]] = None
    iirPasses: typing.Optional[collections.OrderedDict[str, IIRPass]] = None

    def getFpirPasses(self) -> collections.OrderedDict[str, FPIRPass]:
        return self._ensureNonNull('fpirPasses')

    def getFpirToIirPasses(self) -> collections.OrderedDict[str, FPIRPassBuildIntervals]:
        return self._ensureNonNull('fpirToIirPasses')

    def getIirPasses(self) -> collections.OrderedDict[str, IIRPass]:
        return self._ensureNonNull('iirPasses')

class AbstractSpeculativeStrategy(AbstractStrategy, abc.ABC):

    class AggregatedFeatureKey(AttachmentKey):
        """Attachment key for the aggregated feature stored on an Interval by the speculative engine."""
        pass

    # Speculative-specific required attributes
    specIirPasses: typing.Optional[collections.OrderedDict[str, IIRPass]] = None

    def __init__(self) -> None:
        # Note: concrete subclasses call AbstractStrategy.__init__(contentRect) directly.
        # This class does not forward contentRect because speculative strategies
        # may initialize it differently from framewise strategies.
        self.statAnalyzedFrames: int = 0

    def getSpecIirPasses(self) -> collections.OrderedDict[str, IIRPass]:
        return self._ensureNonNull('specIirPasses')

    @abc.abstractmethod
    def decideFeatureMerge(self, oldFeatures: typing.List[typing.Any], newFeatures: typing.List[typing.Any]) -> bool:
        pass

    @abc.abstractmethod
    def aggregateFeatures(self, features: typing.List[typing.Any]) -> typing.Any:
        pass

    @abc.abstractmethod
    def isFpNonEmpty(self, fp: FramePoint) -> bool:
        pass

    @abc.abstractmethod
    def getFpFeature(self, fp: FramePoint) -> typing.Any:
        pass

    @abc.abstractmethod
    def freeFpFeature(self, fp: FramePoint) -> None:
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

class AbstractExtraJobStrategy(AbstractStrategy, abc.ABC):
    @abc.abstractmethod
    def cutExtraJobFrame(self, frame: cv.Mat) -> cv.Mat:
        pass

    @classmethod
    def getExtraJobFrameKey(cls) -> type:
        return ExtraJobFrameKey
