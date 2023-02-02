import abc
import typing
import cv2 as cv

class AbstractRect(abc.ABC):
    @abc.abstractmethod
    def getSizeFloat(self) -> typing.Tuple[float, float]:
        # returns: width, height
        pass

    @abc.abstractmethod
    def getOffsetsFloat(self) -> typing.Tuple[float, float]:
        # returns: leftOffset, topOffset
        pass

    def getSizeInt(self) -> typing.Tuple[int, int]:
        # returns: width, height
        return tuple([int(x) for x in self.getSizeFloat()])
    
    def getOffsetsInt(self) -> typing.Tuple[int, int]:
        # returns: leftOffset, topOffset
        return tuple([int(x) for x in self.getOffsetsFloat()])

    def getBottomRightOffsetsFloat(self) -> typing.Tuple[float, float]:
        # returns: rightOffset, bottomOffset
        return tuple([i + j for i, j in zip(self.getOffsetsFloat(), self.getSizeFloat())])

    def getBottomRightOffsetsInt(self) -> typing.Tuple[int, int]:
        # returns: rightOffset, bottomOffset
        return tuple([int(x) for x in self.getBottomRightOffsetsFloat()])

    def getWidthInt(self) -> int:
        return self.getSizeInt()[0]

    def getHeightInt(self) -> int:
        return self.getSizeInt()[1]

    def cutRoi(self, frame: cv.Mat) -> cv.Mat:
        leftOffset, topOffset = self.getOffsetsInt()
        rightOffset, bottomOffset = self.getBottomRightOffsetsInt()
        return frame[topOffset:bottomOffset, leftOffset:rightOffset]

    def draw(self, frame: cv.Mat) -> cv.Mat:
        return cv.rectangle(frame, self.getOffsetsInt(), self.getBottomRightOffsetsInt(), (0, 0, 255), 1)

class RatioRect(AbstractRect):
    def __init__(self, parent: AbstractRect, leftRatio: float, rightRatio: float, topRatio: float, bottomRatio: float) -> None:
        self.parent: AbstractRect = parent
        self.leftRatio: float = leftRatio
        self.rightRatio: float = rightRatio
        self.topRatio: float = topRatio
        self.bottomRatio: float = bottomRatio

        parentLeftOffset, parentTopOffset = self.parent.getOffsetsFloat()
        parentWidth, parentHeight = self.parent.getSizeFloat()

        self.leftOffset: float = parentLeftOffset + parentWidth * self.leftRatio
        self.rightOffset: float = parentLeftOffset + parentWidth * self.rightRatio
        self.topOffset: float = parentTopOffset + parentHeight * self.topRatio
        self.bottomOffset: float = parentTopOffset + parentHeight * self.bottomRatio
        self.width: float = self.rightOffset - self.leftOffset
        self.height: float = self.bottomOffset - self.topOffset

    def getSizeFloat(self) -> typing.Tuple[float, float]:
        return self.width, self.height

    def getOffsetsFloat(self) -> typing.Tuple[float, float]:
        return self.leftOffset, self.topOffset

class SrcRect(AbstractRect):
    def __init__(self, src: cv.VideoCapture):
        self.width: float = float(src.get(cv.CAP_PROP_FRAME_WIDTH))
        self.height: float = float(src.get(cv.CAP_PROP_FRAME_HEIGHT))

    def getSizeFloat(self) -> typing.Tuple[float, float]:
        return self.width, self.height

    def getOffsetsFloat(self) -> typing.Tuple[float, float]:
        return 0.0, 0.0