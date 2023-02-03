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

class LimbusCompanyMechanicsStrategy(AbstractStrategy):
    class FlagIndex(AbstractFlagIndex):
        Dialog = enum.auto()
        DialogBgColour = enum.auto()
        DialogText = enum.auto()
        DialogTextMin = enum.auto()
        DialogTextFloat = enum.auto()

        @classmethod
        def getDefaultFlagsImpl(cls) -> typing.List[typing.Any]:
            return [False] * cls.getNum()

    def __init__(self, config, contentRect: AbstractRectangle) -> None:
        self.dialogRect = RatioRectangle(contentRect, 0.18, 0.82, 0.88, 0.93)

        self.rectangles = collections.OrderedDict()
        self.rectangles["dialogRect"] = self.dialogRect

        self.cvPasses = [self.cvPassDialog]

        self.fpirPasses = collections.OrderedDict()
        self.fpirPasses["fpirPassDetectFloatJump"] = FPIRPassDetectFloatJump(
            srcFlag=LimbusCompanyMechanicsStrategy.FlagIndex.DialogTextFloat,
            dstFlag=LimbusCompanyMechanicsStrategy.FlagIndex.DialogText
        )
        self.fpirPasses["fpirPassBooleanAnd1"] = FPIRPassBooleanAnd(
            dstFlag=LimbusCompanyMechanicsStrategy.FlagIndex.DialogText,
            op1Flag=LimbusCompanyMechanicsStrategy.FlagIndex.DialogText,
            op2Flag=LimbusCompanyMechanicsStrategy.FlagIndex.DialogTextMin
        )
        self.fpirPasses["fpirPassBooleanAnd2"] = FPIRPassBooleanAnd(
            dstFlag=LimbusCompanyMechanicsStrategy.FlagIndex.Dialog,
            op1Flag=LimbusCompanyMechanicsStrategy.FlagIndex.DialogText,
            op2Flag=LimbusCompanyMechanicsStrategy.FlagIndex.DialogBgColour
        )

        self.fpirToIirPasses = collections.OrderedDict()
        self.fpirToIirPasses["fpirPassBuildIntervals"] = FPIRPassBooleanBuildIntervals(
            LimbusCompanyMechanicsStrategy.FlagIndex.Dialog
        )

        self.iirPasses = collections.OrderedDict()
        self.iirPasses["iirPassFillGapDialog"] = IIRPassFillGap(LimbusCompanyMechanicsStrategy.FlagIndex.Dialog, 300, 1.0)

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
        _, roiDialogTextBin = cv.threshold(roiDialogGray, 128, 255, cv.THRESH_BINARY)
        roiDialogTextBinDialate = cv.morphologyEx(roiDialogTextBin, cv.MORPH_DILATE, kernel=cv.getStructuringElement(cv.MORPH_RECT, (3, 3)))
        roiDialogGrayNoText = cv.bitwise_and(roiDialogGray, roiDialogGray, mask=255-roiDialogTextBinDialate)

        meanDialogGrayNoText = cv.mean(roiDialogGrayNoText)[0]
        meanDialogTextBin: float = cv.mean(roiDialogTextBin)[0]
        hasDialogBgColour: bool = meanDialogGrayNoText < 10

        framePoint.setFlag(LimbusCompanyMechanicsStrategy.FlagIndex.DialogBgColour, hasDialogBgColour)
        framePoint.setFlag(LimbusCompanyMechanicsStrategy.FlagIndex.DialogTextMin, meanDialogTextBin > 10)
        framePoint.setFlag(LimbusCompanyMechanicsStrategy.FlagIndex.DialogTextFloat, meanDialogTextBin)
        # framePoint.setDebugFrame(roiDialogGrayNoText)
        framePoint.setDebugFlag(meanDialogGrayNoText, meanDialogTextBin)
        return False
