import typing
import enum
import collections

from Util import *
from Strategies.AbstractStrategy import *
from AbstractFlagIndex import *
from Rectangle import *
from IR import *

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
        self.pcaParams: np.ndarray = np.load("./Strategies/Models/lcb-mech-dialog-pca.npz")
        
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
            threshDist=0.01,
            inverse=True
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
        roiDialog = self.dialogRect.cutRoi(frame)
        roiDialogGray = cv.cvtColor(roiDialog, cv.COLOR_BGR2GRAY)
        _, roiDialogTextBin = cv.threshold(roiDialogGray, 128, 255, cv.THRESH_BINARY)
        roiDialogTextBinDialate = cv.morphologyEx(roiDialogTextBin, cv.MORPH_DILATE, kernel=cv.getStructuringElement(cv.MORPH_RECT, (3, 3)))
        roiDialogGrayNoText = cv.bitwise_and(roiDialogGray, roiDialogGray, mask=255-roiDialogTextBinDialate)

        meanDialogGrayNoText = cv.mean(roiDialogGrayNoText)[0]
        meanDialogTextBin: float = cv.mean(roiDialogTextBin)[0]
        hasDialogBgColour: bool = meanDialogGrayNoText < 10

        roiDialogGrayResize = cv.resize(roiDialogGray, (100, 20))
        roiDialogGrayFlatten = roiDialogGrayResize.flatten()

        roiDialogGrayFeat = cv.PCAProject(roiDialogGrayFlatten, mean=self.pcaParams["mean"].T, eigenvectors=self.pcaParams["eigenvectors"]).T[0]

        framePoint.setFlag(LimbusCompanyMechanicsStrategy.FlagIndex.DialogBgColour, hasDialogBgColour)
        framePoint.setFlag(LimbusCompanyMechanicsStrategy.FlagIndex.DialogTextMin, meanDialogTextBin > 10)
        framePoint.setFlag(LimbusCompanyMechanicsStrategy.FlagIndex.DialogTextFeat, roiDialogGrayFeat)
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
