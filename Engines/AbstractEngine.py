from __future__ import annotations
import abc
import av.container
import av.container.input
import av.video
import av.video.stream
import av.frame

from Strategies.AbstractStrategy import *
from IR import *

class AbstractEngine(abc.ABC):
    @abc.abstractmethod
    def getRequiredAbstractStrategyType(self) -> typing.Type[AbstractStrategy]:
        pass

    @abc.abstractmethod
    def runImpl(self, strategy: AbstractStrategy, container: "av.container.InputContainer", stream: "av.video.stream.VideoStream") -> IIR:
        pass

    def run(self, strategy: AbstractStrategy, container: "av.container.InputContainer", stream: "av.video.stream.VideoStream") -> IIR:
        if not isinstance(strategy, self.getRequiredAbstractStrategyType()):
            raise Exception(self.__class__.__name__ + " requires a " + self.getRequiredAbstractStrategyType().__name__ + " strategy!")
        return self.runImpl(strategy, container, stream)

