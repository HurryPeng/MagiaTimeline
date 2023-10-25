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
        
        Speaker = enum.auto()
        SpeakerText = enum.auto()
        SpeakerFeat = enum.auto()
        SpeakerCont = enum.auto()

        SpeakerTextFrame = enum.auto()

        @classmethod
        def getDefaultFlagsImpl(cls) -> typing.List[typing.Any]:
            return [False, False, False, False, False, np.array([1.0] * 4), False, None]

    def __init__(self, config: dict, contentRect: AbstractRectangle) -> None:
        self.pcaParams: np.ndarray = np.load("./Strategies/Models/lcb-speaker-pca.npz")
        
        self.rectangles = collections.OrderedDict()
        for k, v in config.items():
            self.rectangles[k] = RatioRectangle(contentRect, *v)

        self.dialogRect = self.rectangles["dialogRect"]
        self.dialogAboveRect = self.rectangles["dialogAboveRect"]
        self.speakerRect = self.rectangles["speakerRect"]

        self.cvPasses = [self.cvPassDialog, self.cvPassSpeaker]

        self.fpirPasses = collections.OrderedDict()
        # self.fpirPasses["fpirPassTrainPCA"] = LimbusCompanyStrategy.FPIRPassTrainPCA()
        self.fpirPasses["fpirPassRemoveNoiseDialog"] = FPIRPassBooleanRemoveNoise(LimbusCompanyStrategy.FlagIndex.Dialog, True, 10)
        self.fpirPasses["fpirPassDetectFeatureJumpSpeaker"] = FPIRPassDetectFeatureJump(
            featFlag=LimbusCompanyStrategy.FlagIndex.SpeakerFeat,
            dstFlag=LimbusCompanyStrategy.FlagIndex.SpeakerCont,
            featOpMean=lambda feats : np.mean(feats, 0),
            featOpDist=lambda lhs, rhs : 0.5 - cosineSimilarity(lhs, rhs) / 2,
            threshDist=0.0001,
            inverse=True
        )
        def reduceSpeaker(framePoint: FramePoint):
            framePoint.setFlag(LimbusCompanyStrategy.FlagIndex.Speaker,
                framePoint.getFlag(LimbusCompanyStrategy.FlagIndex.SpeakerText)
                and framePoint.getFlag(LimbusCompanyStrategy.FlagIndex.SpeakerCont)
            )
        self.fpirPasses["fpirPassReduceSpeaker"] = FPIRPassFramewiseFunctional(
            func=reduceSpeaker
        )
        self.fpirPasses["fpirPassRemoveNoiseSpeaker"] = FPIRPassBooleanRemoveNoise(LimbusCompanyStrategy.FlagIndex.Speaker, True, 10)

        self.fpirToIirPasses = collections.OrderedDict()
        self.fpirToIirPasses["fpirPassBuildIntervals"] = FPIRPassBooleanBuildIntervals(
            LimbusCompanyStrategy.FlagIndex.Dialog, 
            LimbusCompanyStrategy.FlagIndex.Speaker
        )

        self.iirPasses = collections.OrderedDict()
        self.iirPasses["iirPassFillGapDialog"] = IIRPassFillGap(LimbusCompanyStrategy.FlagIndex.Dialog, 500, 0.0)
        self.iirPasses["iirPassFillGapSpeaker"] = IIRPassFillGap(LimbusCompanyStrategy.FlagIndex.Speaker, 500, 1.0)
        self.iirPasses["iirPassFillAlignDialogToSpeaker"] = IIRPassAlign(
            tgtFlag=LimbusCompanyStrategy.FlagIndex.Dialog,
            refFlag=LimbusCompanyStrategy.FlagIndex.Speaker,
            maxGap=600
        )

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
    
    def getFlag2Track(self) -> typing.Dict[AbstractFlagIndex, int]:
        return {LimbusCompanyStrategy.FlagIndex.Speaker: 1}

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
        roiDialogWithMeanTextCorrectedBlurHSVBin = inRange(roiDialogWithMeanTextCorrectedBlurHSV, [0, 20, 10], [180, 255, 50])
        meanDialogWithMeanTextCorrectedBlurHSVBin = cv.mean(roiDialogWithMeanTextCorrectedBlurHSVBin)[0]

        hasDialogBgColour: bool = meanDialogWithMeanTextCorrectedBlurHSVBin > 180.0
        hasDialogText: bool = meanDialogTextBin > 0.3 and meanDialogTextBin < 30.0

        isValidDialog = hasDialogBgColour and hasDialogText

        framePoint.setFlag(LimbusCompanyStrategy.FlagIndex.Dialog, isValidDialog)
        framePoint.setFlag(LimbusCompanyStrategy.FlagIndex.DialogBgColour, hasDialogBgColour)
        framePoint.setFlag(LimbusCompanyStrategy.FlagIndex.DialogText, hasDialogText)
        return False
    
    def cvPassSpeaker(self, frame: cv.Mat, framePoint: FramePoint) -> bool:
        roiSpeaker = self.speakerRect.cutRoi(frame)

        roiSpeakerGray = cv.cvtColor(roiSpeaker, cv.COLOR_BGR2GRAY)
        _, roiSpeakerTextBin = cv.threshold(roiSpeakerGray, 192, 255, cv.THRESH_TOZERO)
        roiSpeakerTextResize = cv.resize(roiSpeakerGray, (75, 30))
        roiSpeakerTextFlatten = roiSpeakerTextResize.flatten()
        roiSpeakerTextFeat = cv.PCAProject(roiSpeakerTextFlatten, mean=self.pcaParams["mean"].T, eigenvectors=self.pcaParams["eigenvectors"]).T[0]
        meanSpeakerTextBin: float = cv.mean(roiSpeakerTextBin)[0]

        hasSpeakerText: bool = meanSpeakerTextBin > 2.5
        
        framePoint.setFlag(LimbusCompanyStrategy.FlagIndex.SpeakerText, hasSpeakerText)
        framePoint.setFlag(LimbusCompanyStrategy.FlagIndex.SpeakerFeat, roiSpeakerTextFeat)
        return False
    
    class FPIRPassTrainPCA(FPIRPass):
        def apply(self, fpir: FPIR):
            feats: np.ndarray = np.array([])
            for i, framePoint in enumerate(fpir.framePoints):
                feat: np.ndarray = framePoint.getFlag(LimbusCompanyStrategy.FlagIndex.SpeakerTextFrame)
                if i % 15 != 0: # sample only a few frames because one single subtitle lasts for seconds
                    continue
                if i == 0:
                    feats = feat
                else:
                    feats = np.vstack((feats, feat))

            mean, eigenvectors = cv.PCACompute(feats, mean=None, maxComponents=4)
            np.savez("./Strategies/Models/lcb-speaker-pca.npz", mean=mean, eigenvectors=eigenvectors)

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
