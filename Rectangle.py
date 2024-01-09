from __future__ import annotations
import abc
import typing
import cv2 as cv

class AbstractRectangle(abc.ABC):
    @abc.abstractmethod
    def getParent(self) -> typing.Optional[AbstractRectangle]:
        pass

    @abc.abstractmethod
    def getCornersFloat(self, canvasRect: typing.Optional[AbstractRectangle] = None) -> typing.Tuple[float, float, float, float]:
        # returns: left, right, top, bottom
        pass

    def getCornersInt(self, canvasRect: typing.Optional[AbstractRectangle] = None) -> typing.Tuple[int, int, int, int]:
        # returns: left, right, top, bottom
        return tuple([int(x) for x in self.getCornersFloat(canvasRect)])

    def getSizeFloat(self, canvasRect: typing.Optional[AbstractRectangle] = None) -> typing.Tuple[float, float]:
        # returns: width, height
        l, r, t, b = self.getCornersFloat(canvasRect)
        return r - l, b - t

    def getSizeInt(self, canvasRect: typing.Optional[AbstractRectangle] = None) -> typing.Tuple[int, int]:
        # returns: width, height
        return tuple([int(x) for x in self.getSizeFloat(canvasRect)])
    
    def getArea(self, canvasRect: typing.Optional[AbstractRectangle] = None) -> int:
        w, h = self.getSizeInt(canvasRect)
        return w * h

    def cutRoi(self, frame: cv.UMat, canvasRect: typing.Optional[AbstractRectangle] = None) -> cv.UMat:
        l, r, t, b = self.getCornersInt(canvasRect)
        return cv.UMat(frame, (t, b), (l, r))
    
    def cutRoiAndGetShape(self, frame: cv.UMat, canvasRect: typing.Optional[AbstractRectangle] = None) -> typing.Tuple[cv.UMat, typing.Tuple[int, int]]:
        l, r, t, b = self.getCornersInt(canvasRect)
        return cv.UMat(frame, (t, b), (l, r)), (r - l, b - t) # width, height
        
    def draw(self, frame: cv.UMat, canvasRect: typing.Optional[AbstractRectangle] = None) -> cv.UMat:
        l, r, t, b = self.getCornersInt(canvasRect)
        return cv.rectangle(frame, (l, t), (r, b), (0, 0, 255), 1)

class RatioRectangle(AbstractRectangle):
    def __init__(self, parent: AbstractRectangle, leftRatio: float, rightRatio: float, topRatio: float, bottomRatio: float) -> None:
        self.parent: AbstractRectangle = parent
        self.updateRatios(leftRatio, rightRatio, topRatio, bottomRatio)

    def updateRatios(self, leftRatio: float, rightRatio: float, topRatio: float, bottomRatio: float):
        if leftRatio > rightRatio or topRatio > bottomRatio:
            raise Exception("Invalid ratio rectangle configuration "
            + str([leftRatio, rightRatio, topRatio, bottomRatio])
            + ". Left/top ratio cannot exceed right/bottom ratio"
        )
        self.leftRatio: float = leftRatio
        self.rightRatio: float = rightRatio
        self.topRatio: float = topRatio
        self.bottomRatio: float = bottomRatio
    
    def getParent(self) -> typing.Optional[AbstractRectangle]:
        return self.parent

    def getCornersFloat(self, canvasRect: typing.Optional[AbstractRectangle] = None) -> typing.Tuple[float, float, float, float]:
        parentLeft, parentRight, parentTop, parentBottom = self.parent.getCornersFloat(canvasRect)
        parentWidth = parentRight - parentLeft
        parentHeight = parentBottom - parentTop
        width = parentWidth * (self.rightRatio - self.leftRatio)
        height = parentHeight * (self.bottomRatio - self.topRatio)
        
        if canvasRect is self:
            return 0.0, width, 0.0, height
        
        left = parentLeft + parentWidth * self.leftRatio
        right = left + width
        top = parentTop + parentHeight * self.topRatio
        bottom = top + height
        
        return left, right, top, bottom

class SrcRectangle(AbstractRectangle):
    def __init__(self, src: cv.VideoCapture):
        self.width: float = float(src.get(cv.CAP_PROP_FRAME_WIDTH))
        self.height: float = float(src.get(cv.CAP_PROP_FRAME_HEIGHT))

    def getParent(self) -> typing.Optional[AbstractRectangle]:
        return None

    def getCornersFloat(self, canvasRect: typing.Optional[AbstractRectangle] = None) -> typing.Tuple[float, float, float, float]:
        return 0.0, self.width, 0.0, self.height
