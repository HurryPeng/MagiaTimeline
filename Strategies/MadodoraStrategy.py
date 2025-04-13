import typing
import enum
import collections

from Util import *
from Strategies.AbstractStrategy import *
from AbstractFlagIndex import *
from Rectangle import *
from IR import *

class MadodoraStrategy(AbstractFramewiseStrategy):
    class FlagIndex(AbstractFlagIndex):
        HomeDialog = enum.auto()
        HomeDialogBg = enum.auto()
        HomeDialogText = enum.auto()
        HomeDialogUnder = enum.auto()

        Dialog = enum.auto()
        DialogBg = enum.auto()
        DialogText = enum.auto()

        Whitescreen = enum.auto()
        WhitescreenBg = enum.auto()
        WhitescreenText = enum.auto()

        Blackscreen = enum.auto()
        BlackscreenBg = enum.auto()
        BlackscreenText = enum.auto()

        @classmethod
        def getDefaultFlagsImpl(cls) -> typing.List[typing.Any]:
            return [False] * cls.getNum()

    def __init__(self, config: dict, contentRect: AbstractRectangle) -> None:
        AbstractStrategy.__init__(self, contentRect)
        self.rectangles: collections.OrderedDict[str, AbstractRectangle] = collections.OrderedDict()
        for k, v in config.items():
            self.rectangles[k] = RatioRectangle(contentRect, *v)

        self.homeDialogRect = self.rectangles["homeDialogRect"]
        self.underHomeDialogRect = self.rectangles["underHomeDialogRect"]
        self.dialogRect = self.rectangles["dialogRect"]
        self.whitescreenRect = self.rectangles["whitescreenRect"]
        self.blackscreenRect = self.rectangles["blackscreenRect"]

        self.cvPasses = [self.cvPassHomeDialog, self.cvPassDialog, self.cvPassWhitescreen, self.cvPassBlackscreen]

        self.fpirPasses = collections.OrderedDict()

        self.fpirPasses["fpirPassRemoveNoiseHomeDialogTrue"] = FPIRPassBooleanRemoveNoise(MadodoraStrategy.FlagIndex.HomeDialog, True, 10)

        self.fpirPasses["fpirPassRemoveNoiseDialogTrue"] = FPIRPassBooleanRemoveNoise(MadodoraStrategy.FlagIndex.Dialog, True, 10)

        self.fpirPasses["fpirPassPrioritizeHomeDialog"] = FPIRPassFramewiseFunctional(
            lambda framePoint: framePoint.setFlag(
                MadodoraStrategy.FlagIndex.Dialog,
                framePoint.getFlag(MadodoraStrategy.FlagIndex.Dialog)
                    and not framePoint.getFlag(MadodoraStrategy.FlagIndex.HomeDialog)
            )
        )

        self.fpirToIirPasses = collections.OrderedDict()
        self.fpirToIirPasses["fpirPassBuildIntervals"] = FPIRPassBooleanBuildIntervals(
            MadodoraStrategy.FlagIndex.HomeDialog,
            MadodoraStrategy.FlagIndex.Dialog,
            MadodoraStrategy.FlagIndex.Whitescreen,
            MadodoraStrategy.FlagIndex.Blackscreen,
        )

        self.iirPasses = collections.OrderedDict()
        self.iirPasses["iirPassFillGapHomeDialog"] = IIRPassFillGap(MadodoraStrategy.FlagIndex.HomeDialog, 300, 0.0)
        self.iirPasses["iirPassExtendHomeDialog"] = IIRPassExtend(MadodoraStrategy.FlagIndex.HomeDialog, 300, 0)
        self.iirPasses["iirPassFillGapDialog"] = IIRPassFillGap(MadodoraStrategy.FlagIndex.Dialog, 300, 0.0)
        self.iirPasses["iirPassExtendDialog"] = IIRPassExtend(MadodoraStrategy.FlagIndex.Dialog, 300, 0)
        self.iirPasses["iirPassFillGapWhitescreen"] = IIRPassFillGap(MadodoraStrategy.FlagIndex.Whitescreen, 2000, 0.5)
        self.iirPasses["iirPassExtendWhitescreen"] = IIRPassExtend(MadodoraStrategy.FlagIndex.Whitescreen, 500, 500)
        self.iirPasses["iirPassFillGapBlackscreen"] = IIRPassFillGap(MadodoraStrategy.FlagIndex.Blackscreen, 2000, 0.5)
        self.iirPasses["iirPassExtendBlackscreen"] = IIRPassExtend(MadodoraStrategy.FlagIndex.Blackscreen, 500, 500)

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
    
    def cvPassHomeDialog(self, frame: cv.Mat, framePoint: FramePoint) -> bool:
        roiDialog = self.homeDialogRect.cutRoiToUmat(frame)
        roiDialogGray = cv.cvtColor(roiDialog, cv.COLOR_BGR2GRAY)
        _, roiDialogTextBin = cv.threshold(roiDialogGray, 192, 255, cv.THRESH_BINARY)
        _, roiDialogBgBin = cv.threshold(roiDialogGray, 100, 255, cv.THRESH_BINARY)
        meanDialogTextBin: float = cv.mean(roiDialogTextBin)[0]
        meanDialogBgBin: float = cv.mean(roiDialogBgBin)[0]
        hasDialogBg: bool = meanDialogBgBin < 20 and meanDialogBgBin > 0.1
        hasDialogText: bool = meanDialogTextBin < 10 and meanDialogTextBin > 0.1

        roiUnderDialog = self.underHomeDialogRect.cutRoiToUmat(frame)
        roiUnderDialogGray = cv.cvtColor(roiUnderDialog, cv.COLOR_BGR2GRAY)
        _, roiUnderDialogBin = cv.threshold(roiUnderDialogGray, 20, 255, cv.THRESH_BINARY)
        meanUnderDialogBin: float = cv.mean(roiUnderDialogBin)[0]
        varUnderDialogGray: float = cv.mean(cv.meanStdDev(roiUnderDialogGray)[1])[0]
        hasUnderBlackBar: bool = meanUnderDialogBin < 1 and varUnderDialogGray < 0.01

        isValidDialog = hasDialogBg and hasDialogText and hasUnderBlackBar

        framePoint.setFlag(MadodoraStrategy.FlagIndex.HomeDialog, isValidDialog)
        framePoint.setFlag(MadodoraStrategy.FlagIndex.HomeDialogBg, hasDialogBg)
        framePoint.setFlag(MadodoraStrategy.FlagIndex.HomeDialogText, hasDialogText)
        framePoint.setFlag(MadodoraStrategy.FlagIndex.HomeDialogUnder, hasUnderBlackBar)

        return isValidDialog

    def cvPassDialog(self, frame: cv.Mat, framePoint: FramePoint) -> bool:
        roiDialog = self.dialogRect.cutRoiToUmat(frame)
        roiDialogGray = cv.cvtColor(roiDialog, cv.COLOR_BGR2GRAY)
        _, roiDialogTextBin = cv.threshold(roiDialogGray, 192, 255, cv.THRESH_BINARY)
        _, roiDialogBgBin = cv.threshold(roiDialogGray, 100, 255, cv.THRESH_BINARY)
        meanDialogTextBin: float = cv.mean(roiDialogTextBin)[0]
        meanDialogBgBin: float = cv.mean(roiDialogBgBin)[0]
        hasDialogBg: bool = meanDialogBgBin < 20 and meanDialogBgBin > 0.1
        hasDialogText: bool = meanDialogTextBin < 10 and meanDialogTextBin > 0.1

        isValidDialog = hasDialogBg and hasDialogText

        framePoint.setFlag(MadodoraStrategy.FlagIndex.Dialog, isValidDialog)
        framePoint.setFlag(MadodoraStrategy.FlagIndex.DialogBg, hasDialogBg)
        framePoint.setFlag(MadodoraStrategy.FlagIndex.DialogText, hasDialogText)

        return isValidDialog
    
    def cvPassWhitescreen(self, frame: cv.Mat, framePoint: FramePoint) -> bool:
        roiWhitescreen = self.whitescreenRect.cutRoiToUmat(frame)
        roiWhitescreenGray = cv.cvtColor(roiWhitescreen, cv.COLOR_BGR2GRAY)
        _, roiWhitescreenBgBin = cv.threshold(roiWhitescreenGray, 220, 255, cv.THRESH_BINARY_INV)
        _, roiWhitescreenTextBin = cv.threshold(roiWhitescreenGray, 20, 255, cv.THRESH_BINARY_INV)
        meanWhitescreenBgBin: float = cv.mean(roiWhitescreenBgBin)[0]
        meanWhitescreenTextBin: float = cv.mean(roiWhitescreenTextBin)[0]
        hasWhitescreenBg: bool = meanWhitescreenBgBin < 20
        hasWhitescreenText: bool = meanWhitescreenTextBin < 8 and meanWhitescreenTextBin > 0.1
        hasWhitescreen: bool = hasWhitescreenBg and hasWhitescreenText

        framePoint.setFlag(MadodoraStrategy.FlagIndex.WhitescreenBg, hasWhitescreenBg)
        framePoint.setFlag(MadodoraStrategy.FlagIndex.WhitescreenText, hasWhitescreenText)
        framePoint.setFlag(MadodoraStrategy.FlagIndex.Whitescreen, hasWhitescreen)

        return hasWhitescreenBg
    
    def cvPassBlackscreen(self, frame: cv.Mat, framePoint: FramePoint) -> bool:
        roiBlackscreen = self.blackscreenRect.cutRoiToUmat(frame)
        roiBlackscreenGray = cv.cvtColor(roiBlackscreen, cv.COLOR_BGR2GRAY)
        _, roiBlackscreenBgBin = cv.threshold(roiBlackscreenGray, 10, 255, cv.THRESH_BINARY)
        _, roiBlackscreenTextBin = cv.threshold(roiBlackscreenGray, 240, 255, cv.THRESH_BINARY)
        meanBlackscreenBgBin: float = cv.mean(roiBlackscreenBgBin)[0]
        meanBlackscreenTextBin: float = cv.mean(roiBlackscreenTextBin)[0]

        hasBlackscreenBg: bool = meanBlackscreenBgBin < 15
        hasBlackscreenText: bool = meanBlackscreenTextBin < 5 and meanBlackscreenTextBin > 0.1
        hasBlackscreen: bool = hasBlackscreenBg and hasBlackscreenText

        framePoint.setFlag(MadodoraStrategy.FlagIndex.BlackscreenBg, hasBlackscreenBg)
        framePoint.setFlag(MadodoraStrategy.FlagIndex.BlackscreenText, hasBlackscreenText)
        framePoint.setFlag(MadodoraStrategy.FlagIndex.Blackscreen, hasBlackscreen)
