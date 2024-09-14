import typing
import enum
import collections

from Util import *
from Strategies.AbstractStrategy import *
from AbstractFlagIndex import *
from Rectangle import *
from IR import *

class MagirecoStrategy(AbstractFramewiseStrategy):
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

    def __init__(self, config: dict, contentRect: AbstractRectangle) -> None:
        AbstractStrategy.__init__(self, contentRect)
        self.rectangles: collections.OrderedDict[str, AbstractRectangle] = collections.OrderedDict()
        for k, v in config.items():
            self.rectangles[k] = RatioRectangle(contentRect, *v)

        self.dialogOutlineRect = self.rectangles["dialogOutlineRect"]
        self.dialogBgRect = self.rectangles["dialogBgRect"]
        self.blackscreenRect = self.rectangles["blackscreenRect"]
        self.whitescreenRect = self.rectangles["whitescreenRect"]
        self.cgSubAboveRect = self.rectangles["cgSubAboveRect"]
        self.cgSubBorderRect = self.rectangles["cgSubBorderRect"]
        self.cgSubBelowRect = self.rectangles["cgSubBelowRect"]
        self.cgSubTextRect = self.rectangles["cgSubTextRect"]

        self.cvPasses = [self.cvPassDialog, self.cvPassBlackscreen, self.cvPassWhitescreen, self.cvPassCgSub]

        self.fpirPasses = collections.OrderedDict()
        self.fpirPasses["fpirPassRemoveNoiseDialogFalse"] = FPIRPassBooleanRemoveNoise(MagirecoStrategy.FlagIndex.Dialog, False, 2)
        self.fpirPasses["fpirPassRemoveNoiseDialogTrue"] = FPIRPassBooleanRemoveNoise(MagirecoStrategy.FlagIndex.Dialog, True, 10)
        self.fpirPasses["fpirPassRemoveNoiseBlackscreenFalse"] = FPIRPassBooleanRemoveNoise(MagirecoStrategy.FlagIndex.Blackscreen, False, 3)
        self.fpirPasses["fpirPassRemoveNoiseBlackscreenTrue"] = FPIRPassBooleanRemoveNoise(MagirecoStrategy.FlagIndex.Blackscreen, True, 10)
        self.fpirPasses["fpirPassRemoveNoiseWhitescreenFalse"] = FPIRPassBooleanRemoveNoise(MagirecoStrategy.FlagIndex.Whitescreen, False, 3)
        self.fpirPasses["fpirPassRemoveNoiseWhitescreenTrue"] = FPIRPassBooleanRemoveNoise(MagirecoStrategy.FlagIndex.Whitescreen, True, 10)
        self.fpirPasses["fpirPassRemoveNoiseCgSubFalse"] = FPIRPassBooleanRemoveNoise(MagirecoStrategy.FlagIndex.CgSub, False, 2)
        self.fpirPasses["fpirPassRemoveNoiseCgSubTrue"] = FPIRPassBooleanRemoveNoise(MagirecoStrategy.FlagIndex.CgSub, True, 10)

        self.fpirToIirPasses = collections.OrderedDict()
        self.fpirToIirPasses["fpirPassBuildIntervals"] = FPIRPassBooleanBuildIntervals(
            MagirecoStrategy.FlagIndex.Dialog, 
            MagirecoStrategy.FlagIndex.Blackscreen, 
            MagirecoStrategy.FlagIndex.Whitescreen, 
            MagirecoStrategy.FlagIndex.CgSub
        )

        self.iirPasses = collections.OrderedDict()
        self.iirPasses["iirPassFillGapDialog"] = IIRPassFillGap(MagirecoStrategy.FlagIndex.Dialog, 300)
        self.iirPasses["iirPassFillGapBlackscreen"] = IIRPassFillGap(MagirecoStrategy.FlagIndex.Blackscreen, 1200)
        self.iirPasses["iirPassFillGapWhitescreen"] = IIRPassFillGap(MagirecoStrategy.FlagIndex.Whitescreen, 1200)
        self.iirPasses["iirPassFillGapCgSub"] = IIRPassFillGap(MagirecoStrategy.FlagIndex.CgSub, 1200)

    @classmethod
    def getFlagIndexType(cls) -> typing.Type[AbstractFlagIndex]:
        return cls.FlagIndex

    def getRectangles(self) -> collections.OrderedDict[str, AbstractRectangle]:
        return self.rectangles

    def getCvPasses(self) -> typing.List[typing.Callable[[cv.Mat, FramePoint], bool]]:
        return self.cvPasses

    def getFpirPasses(self) -> collections.OrderedDict[str, FPIRPass]:
        return self.fpirPasses

    def getFpirToIirPasses(self) -> collections.OrderedDict[str, FPIRPassBuildIntervals]:
        return self.fpirToIirPasses

    def getIirPasses(self) -> collections.OrderedDict[str, IIRPass]:
        return self.iirPasses

    def cvPassDialog(self, frame: cv.Mat, framePoint: FramePoint) -> bool:
        roiDialogBg = self.dialogBgRect.cutRoiToUmat(frame)
        roiDialogBgGray = cv.cvtColor(roiDialogBg, cv.COLOR_BGR2GRAY)
        roiDialogBgHSV = cv.cvtColor(roiDialogBg, cv.COLOR_BGR2HSV)
        roiDialogBgBin = inRange(roiDialogBgHSV, [0, 0, 160], [255, 32, 255])
        _, roiDialogBgTextBin = cv.threshold(roiDialogBgGray, 192, 255, cv.THRESH_BINARY)
        meanDialogTextBin: float = cv.mean(roiDialogBgTextBin)[0]
        meanDialogBgBin: float = cv.mean(roiDialogBgBin)[0]
        hasDialogBg: bool = meanDialogBgBin > 160
        hasDialogText: bool = meanDialogTextBin < 254 and meanDialogTextBin > 192

        roiDialogOutline = self.dialogOutlineRect.cutRoiToUmat(frame)
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
        roiBlackscreen = self.blackscreenRect.cutRoiToUmat(frame)
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
        roiWhitescreen = self.whitescreenRect.cutRoiToUmat(frame)
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
        roiCgSubAbove = self.cgSubAboveRect.cutRoiToUmat(frame)
        roiCgSubAboveGray = cv.cvtColor(roiCgSubAbove, cv.COLOR_BGR2GRAY)
        meanCgSubAboveGray = cv.mean(roiCgSubAboveGray)[0]
        roiCgSubBelow = self.cgSubBelowRect.cutRoiToUmat(frame)
        roiCgSubBelowGray = cv.cvtColor(roiCgSubBelow, cv.COLOR_BGR2GRAY)
        _, roiCgSubBelowGrayNoText = cv.threshold(roiCgSubBelowGray, 160, 255, cv.THRESH_TOZERO_INV)
        meanCgSubBelowGrayNoText: float = cv.mean(roiCgSubBelowGrayNoText)[0]
        cgSubBrightnessDecrVal: float = meanCgSubAboveGray - meanCgSubBelowGrayNoText
        cgSubBrightnessDecrRate: float = 1 - meanCgSubBelowGrayNoText / max(meanCgSubAboveGray, 1.0)
        hasCgSubContrast: bool = cgSubBrightnessDecrVal > 15.0 and cgSubBrightnessDecrRate > 0.30

        roiCgSubBorder = self.cgSubBorderRect.cutRoiToUmat(frame)
        roiCgSubBorderGray = cv.cvtColor(roiCgSubBorder, cv.COLOR_BGR2GRAY)
        roiCgSubBorderEdge = cv.convertScaleAbs(cv.Sobel(roiCgSubBorderGray, cv.CV_16S, 0, 1, ksize=3))
        _, roiCgSubBorderEdgeBin = cv.threshold(roiCgSubBorderEdge, 5, 255, cv.THRESH_BINARY)
        roiCgSubBorderBinErode = cv.morphologyEx(roiCgSubBorderEdgeBin, cv.MORPH_ERODE, kernel=cv.getStructuringElement(cv.MORPH_RECT, (20, 1)))
        roiCgSubBorderRowReduce = cv.reduce(roiCgSubBorderBinErode, 1, cv.REDUCE_AVG, dtype=cv.CV_32F)
        maxCgSubBorderRowReduce: float = cv.minMaxLoc(roiCgSubBorderRowReduce)[1]
        hasCgSubBorder: bool = maxCgSubBorderRowReduce > 200.0

        roiCgSubText = self.cgSubTextRect.cutRoiToUmat(frame)
        roiCgSubTextGray = cv.cvtColor(roiCgSubText, cv.COLOR_BGR2GRAY)
        _, roiCgSubTextBin = cv.threshold(roiCgSubTextGray, 160, 255, cv.THRESH_BINARY)
        meanCgSubTextBin: float = cv.mean(roiCgSubTextBin)[0]
        hasCgSubText: bool = meanCgSubTextBin > 0.5 and meanCgSubTextBin < 50

        isValidCgSub = hasCgSubContrast and hasCgSubBorder and hasCgSubText

        framePoint.setFlag(MagirecoStrategy.FlagIndex.CgSub, isValidCgSub)
        framePoint.setFlag(MagirecoStrategy.FlagIndex.CgSubContrast, hasCgSubContrast)
        framePoint.setFlag(MagirecoStrategy.FlagIndex.CgSubBorder, hasCgSubBorder)
        framePoint.setFlag(MagirecoStrategy.FlagIndex.CgSubText, hasCgSubText)
        return isValidCgSub
