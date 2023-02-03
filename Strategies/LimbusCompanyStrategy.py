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
        DialogBgColour = enum.auto()
        DialogText = enum.auto()

        @classmethod
        def getDefaultFlagsImpl(cls) -> typing.List[typing.Any]:
            return [False] * cls.getNum()

    def __init__(self, config, contentRect: AbstractRectangle) -> None:
        self.dialogRect = RatioRectangle(contentRect, 0.18, 0.82, 0.75, 0.95)

        self.rectangles = collections.OrderedDict()
        self.rectangles["dialogRect"] = self.dialogRect

        self.cvPasses = [self.cvPassDialog]

        self.fpirPasses = collections.OrderedDict()
        self.fpirPasses["fpirPassRemoveNoiseDialog"] = FPIRPassBooleanRemoveNoise(LimbusCompanyStrategy.FlagIndex.Dialog, minNegativeLength=0)

        self.fpirToIirPasses = collections.OrderedDict()
        self.fpirToIirPasses["fpirPassBuildIntervals"] = FPIRPassBooleanBuildIntervals(
            LimbusCompanyStrategy.FlagIndex.Dialog
        )

        self.iirPasses = collections.OrderedDict()
        self.iirPasses["iirPassFillGapDialog"] = IIRPassFillGap(LimbusCompanyStrategy.FlagIndex.Dialog, 300)

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
        roiDialog = self.dialogRect.cutRoi(frame)
        roiDialogGray = cv.cvtColor(roiDialog, cv.COLOR_BGR2GRAY)
        _, roiDialogTextBin = cv.threshold(roiDialogGray, 172, 255, cv.THRESH_BINARY)
        roiDialogTextBinDialate = cv.morphologyEx(roiDialogTextBin, cv.MORPH_DILATE, kernel=cv.getStructuringElement(cv.MORPH_RECT, (3, 3)))
        roiDialogNoText = cv.bitwise_and(roiDialog, roiDialog, mask=255-roiDialogTextBinDialate)
        roiDialogNoTextHSV = cv.cvtColor(roiDialogNoText, cv.COLOR_BGR2HSV)
        meanDialogNoTextHSV = cv.mean(roiDialogNoTextHSV)
        meanDialogTextBin: float = cv.mean(roiDialogTextBin)[0]
        hasDialogBgColour: bool = meanDialogNoTextHSV[0] > 10 and meanDialogNoTextHSV[0] < 45 \
            and meanDialogNoTextHSV[1] > 20 and meanDialogNoTextHSV[1] < 100 \
            and meanDialogNoTextHSV[2] > 10 and meanDialogNoTextHSV[2] < 45
        hasDialogText: bool = meanDialogTextBin > 0.5 and meanDialogTextBin < 30.0

        isValidDialog = hasDialogBgColour and hasDialogText

        framePoint.setFlag(LimbusCompanyStrategy.FlagIndex.Dialog, isValidDialog)
        framePoint.setFlag(LimbusCompanyStrategy.FlagIndex.DialogBgColour, hasDialogBgColour)
        framePoint.setFlag(LimbusCompanyStrategy.FlagIndex.DialogText, hasDialogText)
        return isValidDialog
