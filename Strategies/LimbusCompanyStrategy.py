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

    def __init__(self, config: dict, contentRect: AbstractRectangle) -> None:
        self.rectangles = collections.OrderedDict()
        for k, v in config.items():
            self.rectangles[k] = RatioRectangle(contentRect, *v)
        self.dialogRect = RatioRectangle(contentRect, 0.18, 0.82, 0.75, 0.95)
        self.dialogAboveRect = RatioRectangle(contentRect, 0.18, 0.82, 0.65, 0.75)

        self.dialogRect = self.rectangles["dialogRect"]
        self.dialogAboveRect = self.rectangles["dialogAboveRect"]

        self.cvPasses = [self.cvPassDialog]

        self.fpirPasses = collections.OrderedDict()
        self.fpirPasses["fpirPassRemoveNoiseDialog"] = FPIRPassBooleanRemoveNoise(LimbusCompanyStrategy.FlagIndex.Dialog, True, 10)

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
        alpha = 0.85
        # stdDialogBGR = np.array((15.11, 23.24, 25.42))
        # stdDialogHSV = np.array((27.0, 102.0, 25.5))

        roiDialog = self.dialogRect.cutRoi(frame)
        roiDialogAbove = self.dialogAboveRect.cutRoi(frame)

        roiDialogGray = cv.cvtColor(roiDialog, cv.COLOR_BGR2GRAY)
        _, roiDialogTextBin = cv.threshold(roiDialogGray, 128, 255, cv.THRESH_BINARY)
        meanDialogTextBin: float = cv.mean(roiDialogTextBin)[0]
        roiDialogTextBinDialate = cv.morphologyEx(roiDialogTextBin, cv.MORPH_DILATE, kernel=cv.getStructuringElement(cv.MORPH_RECT, (3, 3)))
        meanDialog = cv.mean(roiDialog)
        roiDialogMealAll = roiDialog.copy()
        roiDialogMealAll[:] = meanDialog[0:3]
        roiDialogNoText = cv.bitwise_and(roiDialog, roiDialog, mask=255-roiDialogTextBinDialate)
        roiDialogMeanText = cv.bitwise_and(roiDialogMealAll, roiDialogMealAll, mask=roiDialogTextBinDialate)
        roiDialogWithMeanText = roiDialogNoText + roiDialogMeanText
        meanDialogAbove = np.array(cv.mean(roiDialogAbove)[0:3])
        roiDialogWithMeanTextCorrected: cv.Mat = np.uint8((1 / alpha) * roiDialogWithMeanText - (1 / alpha - 1) * meanDialogAbove) # type: ignore
        roiDialogWithMeanTextCorrectedBlur = cv.blur(roiDialogWithMeanTextCorrected, (10, 10))
        roiDialogWithMeanTextCorrectedBlurHSV = cv.cvtColor(roiDialogWithMeanTextCorrectedBlur, cv.COLOR_BGR2HSV)
        roiDialogWithMeanTextCorrectedBlurHSVBin = inRange(roiDialogWithMeanTextCorrectedBlurHSV, [10, 20, 10], [45, 255, 50])
        meanDialogWithMeanTextCorrectedBlurHSVBin = cv.mean(roiDialogWithMeanTextCorrectedBlurHSVBin)[0]

        hasDialogBgColour: bool = meanDialogWithMeanTextCorrectedBlurHSVBin > 180.0
        hasDialogText: bool = meanDialogTextBin > 0.8 and meanDialogTextBin < 30.0

        isValidDialog = hasDialogBgColour and hasDialogText

        framePoint.setFlag(LimbusCompanyStrategy.FlagIndex.Dialog, isValidDialog)
        framePoint.setFlag(LimbusCompanyStrategy.FlagIndex.DialogBgColour, hasDialogBgColour)
        framePoint.setFlag(LimbusCompanyStrategy.FlagIndex.DialogText, hasDialogText)
        return isValidDialog

class LimbusCompanyMechanicsStrategy(AbstractStrategy):
    class FlagIndex(AbstractFlagIndex):
        Dialog = enum.auto()
        DialogBgColour = enum.auto()
        DialogTextCont = enum.auto()
        DialogTextMin = enum.auto()
        DialogTextFeat = enum.auto()

        DialogTextFrame = enum.auto()

        @classmethod
        def getDefaultFlagsImpl(cls) -> typing.List[typing.Any]:
            return [False, False, False, False, np.array([1.0] * 4), None]

    def __init__(self, config, contentRect: AbstractRectangle) -> None:
        self.rectangles = collections.OrderedDict()
        for k, v in config.items():
            self.rectangles[k] = RatioRectangle(contentRect, *v)

        self.dialogRect = self.rectangles["dialogRect"]

        self.cvPasses = [self.cvPassDialog]

        self.fpirPasses = collections.OrderedDict()
        self.fpirPasses["fpirPassDetectFeatureJump"] = FPIRPassDetectFeatureJump(
            featFlag=LimbusCompanyMechanicsStrategy.FlagIndex.DialogTextFeat,
            dstFlag=LimbusCompanyMechanicsStrategy.FlagIndex.DialogTextCont, 
            featOpMean=lambda feats : np.mean(feats, 0),
            featOpDist=lambda lhs, rhs : 0.5 - cosineSimilarity(lhs, rhs) / 2,
            threshDist=0.01
        )
        def reduceToDialogText(framePoint: FramePoint):
            framePoint.setFlag(LimbusCompanyMechanicsStrategy.FlagIndex.Dialog,
                framePoint.getFlag(LimbusCompanyMechanicsStrategy.FlagIndex.DialogTextCont)
                and framePoint.getFlag(LimbusCompanyMechanicsStrategy.FlagIndex.DialogTextMin)
                and framePoint.getFlag(LimbusCompanyMechanicsStrategy.FlagIndex.DialogBgColour)
            )
        self.fpirPasses["fpirPassFramewiseFunctional"] = FPIRPassFramewiseFunctional(
            func=reduceToDialogText
        )
        # self.fpirPasses["fpirPassTrainPCA"] = LimbusCompanyMechanicsStrategy.fpirPassTrainPCA()

        self.fpirToIirPasses = collections.OrderedDict()
        self.fpirToIirPasses["fpirPassBuildIntervals"] = FPIRPassBooleanBuildIntervals(
            LimbusCompanyMechanicsStrategy.FlagIndex.Dialog
        )

        self.iirPasses = collections.OrderedDict()
        self.iirPasses["iirPassFillGapDialog"] = IIRPassFillGap(LimbusCompanyMechanicsStrategy.FlagIndex.Dialog, 500, 1.0)

        self.pcaParams: np.ndarray = np.load("./Strategies/Models/lcb-mech-dialog-pca.npz")

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

        roiDialogGrayResize = cv.resize(roiDialogGray, (100, 20))
        roiDialogGrayResizeFlatten = roiDialogGrayResize.flatten()

        roiDialogGrayResizeCompressed = cv.PCAProject(roiDialogGrayResizeFlatten, mean=self.pcaParams["mean"].T, eigenvectors=self.pcaParams["eigenvectors"]).T[0]

        framePoint.setFlag(LimbusCompanyMechanicsStrategy.FlagIndex.DialogBgColour, hasDialogBgColour)
        framePoint.setFlag(LimbusCompanyMechanicsStrategy.FlagIndex.DialogTextMin, meanDialogTextBin > 10)
        framePoint.setFlag(LimbusCompanyMechanicsStrategy.FlagIndex.DialogTextFeat, roiDialogGrayResizeCompressed)
        return False

    class FPIRPassTrainPCA(FPIRPass):
        def apply(self, fpir: FPIR):
            feats: np.ndarray = np.array([])
            for i, framePoint in enumerate(fpir.framePoints):
                feat: np.ndarray = framePoint.getFlag(LimbusCompanyMechanicsStrategy.FlagIndex.DialogTextFrame)
                if i % 50 != 0: # sample only a few frames because one single subtitle lasts for seconds
                    continue
                if i == 0:
                    feats = feat
                else:
                    feats = np.vstack((feats, feat))

            mean, eigenvectors = cv.PCACompute(feats, mean=None, maxComponents=4)
            np.savez("./Strategies/Models/lcb-mech-dialog-pca.npz", mean=mean, eigenvectors=eigenvectors)
