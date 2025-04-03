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
        HomeDialogFuture = enum.auto()
        HomeDialogBg = enum.auto()
        HomeDialogText = enum.auto()
        HomeDialogUnder = enum.auto()

        Dialog = enum.auto()
        DialogFuture = enum.auto()
        DialogBg = enum.auto()
        DialogText = enum.auto()


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

        self.cvPasses = [self.cvPassHomeDialog, self.cvPassDialog]

        self.fpirPasses = collections.OrderedDict()

        self.fpirPasses["fpirPassHomeDialogFuture"] = FPIRPassShift(MadodoraStrategy.FlagIndex.HomeDialogFuture, MadodoraStrategy.FlagIndex.HomeDialog, -30, False)
        self.fpirPasses["fpirPassHomeDialogRefine"] = FPIRPassFramewiseFunctional(
            lambda framePoint: framePoint.setFlag(
                MadodoraStrategy.FlagIndex.HomeDialog,
                framePoint.getFlag(MadodoraStrategy.FlagIndex.HomeDialog)
                    or framePoint.getFlag(MadodoraStrategy.FlagIndex.HomeDialogFuture) and framePoint.getFlag(MadodoraStrategy.FlagIndex.HomeDialogBg)
            )
        )
        self.fpirPasses["fpirPassRemoveNoiseHomeDialogTrue"] = FPIRPassBooleanRemoveNoise(MadodoraStrategy.FlagIndex.HomeDialog, True, 10)

        self.fpirPasses["fpirPassDialogFuture"] = FPIRPassShift(MadodoraStrategy.FlagIndex.DialogFuture, MadodoraStrategy.FlagIndex.Dialog, -30, False)
        self.fpirPasses["fpirPassDialogRefine"] = FPIRPassFramewiseFunctional(
            lambda framePoint: framePoint.setFlag(
                MadodoraStrategy.FlagIndex.Dialog,
                framePoint.getFlag(MadodoraStrategy.FlagIndex.Dialog)
                    or framePoint.getFlag(MadodoraStrategy.FlagIndex.DialogFuture) and framePoint.getFlag(MadodoraStrategy.FlagIndex.DialogBg)
            )
        )
        self.fpirPasses["fpirPassRemoveNoiseDialogTrue"] = FPIRPassBooleanRemoveNoise(MadodoraStrategy.FlagIndex.Dialog, True, 10)

        self.fpirPasses["fpirPassPrioritizeHomeDialog"] = FPIRPassFramewiseFunctional(
            lambda framePoint: framePoint.setFlag(
                MadodoraStrategy.FlagIndex.Dialog,
                framePoint.getFlag(MadodoraStrategy.FlagIndex.Dialog)
                    and not framePoint.getFlag(MadodoraStrategy.FlagIndex.HomeDialog)
                    and not framePoint.getFlag(MadodoraStrategy.FlagIndex.HomeDialogFuture)
            )
        )

        self.fpirToIirPasses = collections.OrderedDict()
        self.fpirToIirPasses["fpirPassBuildIntervals"] = FPIRPassBooleanBuildIntervals(
            MadodoraStrategy.FlagIndex.HomeDialog,
            MadodoraStrategy.FlagIndex.Dialog,
        )

        self.iirPasses = collections.OrderedDict()
        self.iirPasses["iirPassFillGapHomeDialog"] = IIRPassFillGap(MadodoraStrategy.FlagIndex.HomeDialog, 300, 0.0)
        self.iirPasses["iirPassFillGapDialog"] = IIRPassFillGap(MadodoraStrategy.FlagIndex.Dialog, 300, 0.0)

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
        meanUnderDialogGray: float = cv.mean(roiUnderDialogBin)[0]
        hasUnderBlackBar: bool = meanUnderDialogGray < 1

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
