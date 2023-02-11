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
        self.iirPasses["iirPassFillGapDialog"] = IIRPassFillGap(LimbusCompanyStrategy.FlagIndex.Dialog, 300, 0.0)

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
        roiDialogAllBrown = roiDialog.copy()
        # roiDialogAllBrownHSV[:] = (27, 75, 25)
        roiDialogAllBrown[:] = (18, 25, 26) # (27, 75, 25) in HSV
        roiDialogNoText = cv.bitwise_and(roiDialog, roiDialog, mask=255-roiDialogTextBinDialate)
        # roiDialogBrownTextHSV = cv.bitwise_and(roiDialogAllBrownHSV, roiDialogAllBrownHSV, mask=roiDialogTextBinDialate)
        roiDialogBrownText = cv.bitwise_and(roiDialogAllBrown, roiDialogAllBrown, mask=roiDialogTextBinDialate)
        # roiDialogNoTextHSV = cv.cvtColor(roiDialogNoText, cv.COLOR_BGR2HSV)
        # roiDialogWithBrownTextHSV = roiDialogNoTextHSV + roiDialogBrownTextHSV
        roiDialogWithBrownText = roiDialogNoText + roiDialogBrownText
        roiDialogWithBrownTextBlur = cv.blur(roiDialogWithBrownText, (10, 10))
        roiDialogWithBrownTextBlurHSV = cv.cvtColor(roiDialogWithBrownTextBlur, cv.COLOR_BGR2HSV)
        # roiDialogWithBrownTextBlurHSV = cv.blur(roiDialogWithBrownTextHSV, (5, 5))
        roiDialogWithBrownTextBlurHSVBin = inRange(roiDialogWithBrownTextBlurHSV, [10, 20, 0], [35, 255, 128])
        meanDialogWithBrownTextBlurHSVBin = cv.mean(roiDialogWithBrownTextBlurHSVBin)[0]

        # roiDialogNoTextBlur = cv.blur(roiDialogNoText, (5, 5))
        # roiDialogNoTextBlurHSV = cv.cvtColor(roiDialogNoTextBlur, cv.COLOR_BGR2HSV)
        # meanDialogNoTextBlurHSV = cv.mean(roiDialogNoTextBlurHSV)
        # roiDialogNoTextBlurHSVBin = inRange(roiDialogNoTextBlurHSV, [10, 20, 0], [45, 255, 128])
        meanDialogTextBin: float = cv.mean(roiDialogTextBin)[0]
        hasDialogBgColour: bool = meanDialogWithBrownTextBlurHSVBin > 200
        hasDialogText: bool = meanDialogTextBin > 0.8 and meanDialogTextBin < 30.0

        isValidDialog = hasDialogBgColour and hasDialogText

        framePoint.setFlag(LimbusCompanyStrategy.FlagIndex.Dialog, isValidDialog)
        framePoint.setFlag(LimbusCompanyStrategy.FlagIndex.DialogBgColour, hasDialogBgColour)
        framePoint.setFlag(LimbusCompanyStrategy.FlagIndex.DialogText, hasDialogText)
        framePoint.setDebugFrame(cv.cvtColor(roiDialogWithBrownTextBlurHSV, cv.COLOR_HSV2BGR))
        # framePoint.setDebugFrame(roiDialogWithBrownTextHSVBin)
        framePoint.setDebugFlag(meanDialogWithBrownTextBlurHSVBin)
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
        self.dialogRect = RatioRectangle(contentRect, 0.18, 0.82, 0.875, 0.915)

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
        return False
