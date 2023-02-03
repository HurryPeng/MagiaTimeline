import typing
import enum
import collections

from Util import *
from Strategies.AbstractStrategy import *
from AbstractFlagIndex import *
from Rectangle import *
from IR import *

class MagirecoStrategy(AbstractStrategy):
    class FlagIndex(AbstractFlagIndex):
        Dialog = enum.auto()
        DialogBg = enum.auto()
        DialogText = enum.auto()
        DialogOutline = enum.auto()

        Blackscreen = enum.auto()
        BlackscreenBg = enum.auto()
        BlackscreenText = enum.auto()

        Whitescreen = enum.auto()
        WhitescreenBg = enum.auto()
        WhitescreenText = enum.auto()

        CgSub = enum.auto()
        CgSubContrast = enum.auto()
        CgSubBorder = enum.auto()
        CgSubText = enum.auto()

        @classmethod
        def getDefaultFlagsImpl(cls) -> typing.List[typing.Any]:
            return [False] * cls.getNum()

    def __init__(self, config, contentRect: AbstractRectangle) -> None:
        self.dialogOutlineRect = RatioRectangle(contentRect, 0.25, 0.75, 0.60, 0.95)
        self.dialogBgRect = RatioRectangle(contentRect, 0.3125, 0.6797, 0.7264, 0.8784)
        self.blackscreenRect = RatioRectangle(contentRect, 0.15, 0.85, 0.00, 1.00)
        self.whitescreenRect = RatioRectangle(contentRect, 0.15, 0.65, 0.00, 1.00)
        self.cgSubAboveRect = RatioRectangle(contentRect, 0.0, 1.0, 0.60, 0.65)
        self.cgSubBorderRect = RatioRectangle(contentRect, 0.0, 1.0, 0.65, 0.70)
        self.cgSubBelowRect = RatioRectangle(contentRect, 0.0, 1.0, 0.70, 0.75)
        self.cgSubTextRect = RatioRectangle(contentRect, 0.3, 0.7, 0.70, 1.00)

        self.rectangles = collections.OrderedDict()
        self.rectangles["dialogOutlineRect"] = self.dialogOutlineRect
        self.rectangles["dialogBgRect"] = self.dialogBgRect
        self.rectangles["blackscreenRect"] = self.blackscreenRect
        self.rectangles["whitescreenRect"] = self.whitescreenRect
        self.rectangles["cgSubAboveRect"] = self.cgSubAboveRect
        self.rectangles["cgSubBorderRect"] = self.cgSubBorderRect
        self.rectangles["cgSubBelowRect"] = self.cgSubBelowRect
        self.rectangles["cgSubTextRect"] = self.cgSubTextRect

    @classmethod
    def getFlagIndexType(cls) -> typing.Type[AbstractFlagIndex]:
        return cls.FlagIndex

    def getRectangles(self) -> collections.OrderedDict[str, AbstractRectangle]:
        return self.rectangles

    def cvPassDialog(self, frame: cv.Mat, framePoint: FramePoint) -> bool:
        roiDialogBg = self.dialogBgRect.cutRoi(frame)
        roiDialogBgGray = cv.cvtColor(roiDialogBg, cv.COLOR_BGR2GRAY)
        roiDialogBgHSV = cv.cvtColor(roiDialogBg, cv.COLOR_BGR2HSV)
        roiDialogBgBin = inRange(roiDialogBgHSV, [0, 0, 160], [255, 32, 255])
        _, roiDialogBgTextBin = cv.threshold(roiDialogBgGray, 192, 255, cv.THRESH_BINARY)
        meanDialogTextBin: float = cv.mean(roiDialogBgTextBin)[0]
        meanDialogBgBin: float = cv.mean(roiDialogBgBin)[0]
        hasDialogBg: bool = meanDialogBgBin > 160
        hasDialogText: bool = meanDialogTextBin < 254 and meanDialogTextBin > 192

        roiDialogOutline = self.dialogOutlineRect.cutRoi(frame)
        roiDialogOutlineHSV = cv.cvtColor(roiDialogOutline, cv.COLOR_BGR2HSV)
        roiDialogOutlineBin = inRange(roiDialogOutlineHSV, [10, 40, 90], [30, 130, 190])
        meanDialogOutlineBin: float = cv.mean(roiDialogOutlineBin)[0]
        hasDialogOutline: bool = meanDialogOutlineBin > 3

        isValidDialog = hasDialogBg and hasDialogText and hasDialogOutline

        framePoint.setFlag(MagirecoStrategy.FlagIndex.Dialog, isValidDialog)
        framePoint.setFlag(MagirecoStrategy.FlagIndex.DialogBg, hasDialogBg)
        framePoint.setFlag(MagirecoStrategy.FlagIndex.DialogText, hasDialogText)
        framePoint.setFlag(MagirecoStrategy.FlagIndex.DialogOutline, hasDialogOutline)
        return isValidDialog

    def cvPassBlackscreen(self, frame: cv.Mat, framePoint: FramePoint) -> bool:
        roiBlackscreen = self.blackscreenRect.cutRoi(frame)
        roiBlackscreenGray = cv.cvtColor(roiBlackscreen, cv.COLOR_BGR2GRAY)
        _, roiBlackscreenBgBin = cv.threshold(roiBlackscreenGray, 80, 255, cv.THRESH_BINARY)
        _, roiBlackscreenTextBin = cv.threshold(roiBlackscreenGray, 160, 255, cv.THRESH_BINARY)
        meanBlackscreenBgBin: float = cv.mean(roiBlackscreenBgBin)[0]
        meanBlackscreenTextBin: float = cv.mean(roiBlackscreenTextBin)[0]
        hasBlackscreenBg: bool = meanBlackscreenBgBin < 20
        hasBlackscreenText: bool = meanBlackscreenTextBin > 0.1 and meanBlackscreenTextBin < 16

        isValidBlackscreen = hasBlackscreenBg and hasBlackscreenText

        framePoint.setFlag(MagirecoStrategy.FlagIndex.Blackscreen, isValidBlackscreen)
        framePoint.setFlag(MagirecoStrategy.FlagIndex.BlackscreenBg, hasBlackscreenBg)
        framePoint.setFlag(MagirecoStrategy.FlagIndex.BlackscreenText, hasBlackscreenText)
        return isValidBlackscreen

    def cvPassWhitescreen(self, frame: cv.Mat, framePoint: FramePoint) -> bool:
        roiWhitescreen = self.whitescreenRect.cutRoi(frame)
        roiWhitescreenGray = cv.cvtColor(roiWhitescreen, cv.COLOR_BGR2GRAY)
        _, roiWhitescreenBgBin = cv.threshold(roiWhitescreenGray, 160, 255, cv.THRESH_BINARY)
        _, roiWhitescreenTextBin = cv.threshold(roiWhitescreenGray, 160, 255, cv.THRESH_BINARY_INV)
        meanWhitescreenBgBin: float = cv.mean(roiWhitescreenBgBin)[0]
        meanWhitescreenTextBin: float = cv.mean(roiWhitescreenTextBin)[0]
        hasWhitescreenBg: bool = meanWhitescreenBgBin > 230
        hasWhitescreenText: bool = meanWhitescreenTextBin > 0.8 and meanWhitescreenTextBin < 16

        isValidWhitescreen = hasWhitescreenBg and hasWhitescreenText

        framePoint.setFlag(MagirecoStrategy.FlagIndex.Whitescreen, isValidWhitescreen)
        framePoint.setFlag(MagirecoStrategy.FlagIndex.WhitescreenBg, hasWhitescreenBg)
        framePoint.setFlag(MagirecoStrategy.FlagIndex.WhitescreenText, hasWhitescreenText)
        return isValidWhitescreen

    def cvPassCgSub(self, frame: cv.Mat, framePoint: FramePoint) -> bool:
        roiCgSubAbove = self.cgSubAboveRect.cutRoi(frame)
        roiCgSubAboveGray = cv.cvtColor(roiCgSubAbove, cv.COLOR_BGR2GRAY)
        meanCgSubAboveGray = cv.mean(roiCgSubAboveGray)[0]
        roiCgSubBelow = self.cgSubBelowRect.cutRoi(frame)
        roiCgSubBelowGray = cv.cvtColor(roiCgSubBelow, cv.COLOR_BGR2GRAY)
        _, roiCgSubBelowGrayNoText = cv.threshold(roiCgSubBelowGray, 160, 255, cv.THRESH_TOZERO_INV)
        meanCgSubBelowGrayNoText: float = cv.mean(roiCgSubBelowGrayNoText)[0]
        cgSubBrightnessDecrVal: float = meanCgSubAboveGray - meanCgSubBelowGrayNoText
        cgSubBrightnessDecrRate: float = 1 - meanCgSubBelowGrayNoText / max(meanCgSubAboveGray, 1.0)
        hasCgSubContrast: bool = cgSubBrightnessDecrVal > 15.0 and cgSubBrightnessDecrRate > 0.30

        roiCgSubBorder = self.cgSubBorderRect.cutRoi(frame)
        roiCgSubBorderGray = cv.cvtColor(roiCgSubBorder, cv.COLOR_BGR2GRAY)
        roiCgSubBorderEdge = cv.convertScaleAbs(cv.Sobel(roiCgSubBorderGray, cv.CV_16S, 0, 1, ksize=3))
        roiCgSubBorderErode = cv.morphologyEx(roiCgSubBorderEdge, cv.MORPH_ERODE, kernel=cv.getStructuringElement(cv.MORPH_RECT, (51, 1)))
        roiCgSubBorderRowReduce = cv.reduce(roiCgSubBorderErode, 1, cv.REDUCE_AVG, dtype=cv.CV_32F)
        maxCgSubBorderRowReduce: float = cv.minMaxLoc(roiCgSubBorderRowReduce)[1]
        hasCgSubBorder: bool = maxCgSubBorderRowReduce > 25.0

        roiCgSubText = self.cgSubTextRect.cutRoi(frame)
        roiCgSubTextGray = cv.cvtColor(roiCgSubText, cv.COLOR_BGR2GRAY)
        _, roiCgSubTextBin = cv.threshold(roiCgSubTextGray, 160, 255, cv.THRESH_BINARY)
        meanCgSubTextBin: float = cv.mean(roiCgSubTextBin)[0]
        hasCgSubText: bool = meanCgSubTextBin > 0.5 and meanCgSubTextBin < 30

        isValidCgSub = hasCgSubContrast and hasCgSubBorder and hasCgSubText

        framePoint.setFlag(MagirecoStrategy.FlagIndex.CgSub, isValidCgSub)
        framePoint.setFlag(MagirecoStrategy.FlagIndex.CgSubContrast, hasCgSubContrast)
        framePoint.setFlag(MagirecoStrategy.FlagIndex.CgSubBorder, hasCgSubBorder)
        framePoint.setFlag(MagirecoStrategy.FlagIndex.CgSubText, hasCgSubText)
        return isValidCgSub

    def getCvPasses(self) -> typing.List[typing.Callable[[cv.Mat, FramePoint], bool]]:
        return [self.cvPassDialog, self.cvPassBlackscreen, self.cvPassWhitescreen, self.cvPassCgSub]
