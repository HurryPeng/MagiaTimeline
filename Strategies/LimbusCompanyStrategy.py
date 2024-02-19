import typing
import enum
import collections

from Util import *
from Strategies.AbstractStrategy import *
from AbstractFlagIndex import *
from Rectangle import *
from IR import *

class LimbusCompanyStrategy(AbstractStrategy):
    class FlagIndex(AbstractFlagIndex):
        Dialog = enum.auto()
        DialogText = enum.auto()
        DialogTextVal = enum.auto()
        DialogBg = enum.auto()
        DialogBgVal = enum.auto()

        @classmethod
        def getDefaultFlagsImpl(cls) -> typing.List[typing.Any]:
            return [False, False, 0.0, False, 0.0]

    def __init__(self, config: dict, contentRect: AbstractRectangle) -> None:
        self.rectangles = collections.OrderedDict()
        for k, v in config.items():
            self.rectangles[k] = RatioRectangle(contentRect, *v)

        self.dialogRect = self.rectangles["dialogRect"]

        self.cvPasses = [self.cvPassDialog]

        self.fpirPasses = collections.OrderedDict()
        self.fpirPasses["fpirPassRemoveNoiseDialog"] = FPIRPassBooleanRemoveNoise(LimbusCompanyStrategy.FlagIndex.Dialog, True, 10)

        self.fpirToIirPasses = collections.OrderedDict()
        self.fpirToIirPasses["fpirPassBuildIntervals"] = FPIRPassBooleanBuildIntervals(
            LimbusCompanyStrategy.FlagIndex.Dialog
        )

        self.iirPasses = collections.OrderedDict()
        self.iirPasses["iirPassFillGapDialog"] = IIRPassFillGap(LimbusCompanyStrategy.FlagIndex.Dialog, 1000, 0.0)

    @classmethod
    def getFlagIndexType(cls) -> typing.Type[AbstractFlagIndex]:
        return cls.FlagIndex

    def getRectangles(self) -> collections.OrderedDict[str, AbstractRectangle]:
        return self.rectangles

    def getCvPasses(self) -> typing.List[typing.Callable[[cv.UMat, FramePoint], bool]]:
        return self.cvPasses

    def getFpirPasses(self) -> collections.OrderedDict[str, FPIRPass]:
        return self.fpirPasses

    def getFpirToIirPasses(self) -> collections.OrderedDict[str, FPIRPassBuildIntervals]:
        return self.fpirToIirPasses

    def getIirPasses(self) -> collections.OrderedDict[str, IIRPass]:
        return self.iirPasses

    def cvPassDialog(self, frame: cv.UMat, framePoint: FramePoint) -> bool:
        roiDialog = self.dialogRect.cutRoi(frame) # Converted to cv.Mat for temporal convenience
        roiDialogGray = cv.cvtColor(roiDialog, cv.COLOR_BGR2GRAY)

        _, roiDialogTextBin = cv.threshold(roiDialogGray, 128, 255, cv.THRESH_BINARY)
        roiDialogTextBinTophat = cv.morphologyEx(roiDialogTextBin, cv.MORPH_TOPHAT, kernel=cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5)))

        meanDialogTextBinTophat: float = cv.mean(roiDialogTextBinTophat)[0]
        dialogTextVal: float = meanDialogTextBinTophat / self.dialogRect.getArea()
        hasDialogText: bool = dialogTextVal > 1e-6 and dialogTextVal < 100e-6

        roiDialogBgBin = cv.adaptiveThreshold(roiDialogGray, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 7, 0)
        roiDialogBgBinNoText = cv.bitwise_and(roiDialogBgBin, cv.bitwise_not(roiDialogTextBinTophat))
        roiDialogBgBinNoTextDenoise = cv.morphologyEx(roiDialogBgBinNoText, cv.MORPH_CLOSE, kernel=cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5)))
        roiDialogBgBinNoTextOpen = cv.morphologyEx(roiDialogBgBinNoText, cv.MORPH_OPEN, kernel=cv.getStructuringElement(cv.MORPH_RECT, (25, 1)))

        meanDialogBgBinNoTextOpen: float = cv.mean(roiDialogBgBinNoTextOpen)[0]
        dialogBgVal: float = meanDialogBgBinNoTextOpen / self.dialogRect.getArea()
        hasDialogBg: bool = dialogBgVal > 80e-6

        isValidDialog = hasDialogText and hasDialogBg
        # framePoint.setDebugFrame(roiDialogBgBinNoTextOpen)

        framePoint.setFlag(LimbusCompanyStrategy.FlagIndex.Dialog, isValidDialog)
        framePoint.setFlag(LimbusCompanyStrategy.FlagIndex.DialogText, hasDialogText)
        framePoint.setFlag(LimbusCompanyStrategy.FlagIndex.DialogTextVal, dialogTextVal)
        framePoint.setFlag(LimbusCompanyStrategy.FlagIndex.DialogBg, hasDialogBg)
        framePoint.setFlag(LimbusCompanyStrategy.FlagIndex.DialogBgVal, dialogBgVal)
        return False
