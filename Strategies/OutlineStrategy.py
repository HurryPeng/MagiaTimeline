import typing
import enum
import collections

from IR import IIRPass
from Util import *
from Strategies.AbstractStrategy import *
from AbstractFlagIndex import *
from Rectangle import *
from IR import *

class OutlineStrategy(AbstractFramewiseStrategy, AbstractSpeculativeStrategy, AbstractOcrStrategy):
    class FlagIndex(AbstractFlagIndex):
        Dialog = enum.auto()
        DialogFeat = enum.auto()
        DialogFeatJump = enum.auto()
        OcrFrame = enum.auto()

        @classmethod
        def getDefaultFlagsImpl(cls) -> typing.List[typing.Any]:
            return [False, np.zeros(64), False, None]

    def __init__(self, config: dict, contentRect: AbstractRectangle) -> None:
        AbstractStrategy.__init__(self, contentRect)
        AbstractSpeculativeStrategy.__init__(self)
        self.rectangles: collections.OrderedDict[str, AbstractRectangle] = collections.OrderedDict()
        self.rectangles["dialogRect"] = RatioRectangle(contentRect, *config["dialogRect"])

        self.dialogRect = self.rectangles["dialogRect"]
        self.fastMode: bool = config["fastMode"]
        self.textWeightMin: int = config["textWeightMin"]
        self.textWeightMax: int = config["textWeightMax"]
        self.textHSVRanges: typing.List[typing.List[int]] = config["textHSVRanges"]
        self.outlineWeightMax: int = config["outlineWeightMax"]
        self.outlineHSVRanges: typing.List[typing.List[int]] = config["outlineHSVRanges"]
        self.boundCompensation: int = config["boundCompensation"]
        self.sobelThreshold: int = config["sobelThreshold"]
        self.nestingSuppression: int = config["nestingSuppression"]
        self.featureThreshold: float = config["featureThreshold"]
        self.featureJumpThreshold: float = config["featureJumpThreshold"]
        self.featureJumpStddevThreshold: float = config["featureJumpStddevThreshold"]
        self.debugLevel: int = config["debugLevel"]

        self.cvPasses = [self.cvPassDialog]

        self.fpirPasses = collections.OrderedDict()

        self.fpirPasses["fpirPassDetectDialogJump"] = FPIRPassDetectFeatureJump(
            featFlag=OutlineStrategy.FlagIndex.DialogFeat,
            dstFlag=OutlineStrategy.FlagIndex.DialogFeatJump, 
            featOpMean=lambda feats : np.mean(feats, axis=0),
            featOpDist=lambda lhs, rhs : np.linalg.norm(lhs - rhs),
            threshDist=self.featureJumpThreshold,
            windowSize=5,
            featOpStd=lambda feats: np.mean(np.std(feats, axis=0)),
            threshStd=self.featureJumpStddevThreshold
        )


        def breakDialogJump(framePoint: FramePoint):
            framePoint.setFlag(OutlineStrategy.FlagIndex.Dialog,
                framePoint.getFlag(OutlineStrategy.FlagIndex.Dialog)
                and not framePoint.getFlag(OutlineStrategy.FlagIndex.DialogFeatJump)
            )
        self.fpirPasses["fpirPassBreakDialogJump"] = FPIRPassFramewiseFunctional(
            func=breakDialogJump
        )
        
        self.fpirToIirPasses = collections.OrderedDict()
        self.fpirToIirPasses["fpirPassBuildIntervals"] = FPIRPassBooleanBuildIntervals(
            OutlineStrategy.FlagIndex.Dialog
        )

        self.iirPasses = collections.OrderedDict()
        self.iirPasses["iirPassFillGapDialog"] = IIRPassFillGap(OutlineStrategy.FlagIndex.Dialog, 300, meetPoint=1.0)
        self.iirPasses["iirPassDenoise"] = IIRPassDenoise(OutlineStrategy.FlagIndex.Dialog, 100)

        self.specIirPasses = collections.OrderedDict()
        self.specIirPasses["iirPassMerge"] = IIRPassMerge(
            lambda iir, interval0, interval1:
                self.decideFeatureMerge(
                    [interval0.getFlag(self.getFeatureFlagIndex())],
                    [interval1.getFlag(self.getFeatureFlagIndex())]
                )
        )
        self.specIirPasses["iirPassDenoise"] = IIRPassDenoise(OutlineStrategy.FlagIndex.Dialog, 300)
        self.specIirPasses["iirPassMerge2"] = self.specIirPasses["iirPassMerge"]

    @classmethod
    def getFlagIndexType(cls) -> typing.Type[AbstractFlagIndex]:
        return cls.FlagIndex
    
    @classmethod
    def getMainFlagIndex(cls) -> AbstractFlagIndex:
        return cls.FlagIndex.Dialog

    @classmethod
    def getFeatureFlagIndex(cls) -> AbstractFlagIndex:
        return cls.FlagIndex.DialogFeat

    @classmethod
    def isEmptyFeature(cls, feature: np.ndarray) -> bool:
        return np.all(feature == 0)
    
    @classmethod
    def getOcrFrameFlagIndex(cls) -> AbstractFlagIndex:
        return cls.FlagIndex.OcrFrame

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
    
    def getSpecIirPasses(self) -> collections.OrderedDict[str, IIRPass]:
        return self.specIirPasses
    
    def decideFeatureMerge(self, oldFeatures: typing.List[np.ndarray], newFeatures: typing.List[np.ndarray]) -> bool:
        return np.linalg.norm(np.mean(oldFeatures, axis=0) - np.mean(newFeatures, axis=0)) < self.featureJumpThreshold

    def aggregateFeatures(self, features: typing.List[np.ndarray]) -> np.ndarray:
        return np.mean(features, axis=0)

    def releaseFeatureOnHook(self) -> bool:
        return False
    
    def cutOcrFrame(self, frame: cv.Mat) -> cv.Mat:
        return self.dialogRect.cutRoi(frame)
    
    def cutCleanOcrFrame(self, frame: cv.Mat) -> cv.Mat:
        return self.ocrPass(frame, fastMode=False)[0]

    def cvPassDialog(self, frame: cv.Mat, framePoint: FramePoint) -> bool:
        roiDialogText, debugFrame = self.ocrPass(frame, fastMode=False)

        framePoint.setDebugFrame(debugFrame)

        meanDialogText = cv.mean(roiDialogText)[0]

        hasDialog = meanDialogText > self.featureThreshold
        dctFeat = np.zeros(64)
        if hasDialog:
            roiDialogTextResized = cv.resize(roiDialogText, (150, 50)).get()
            dctFeat = dctDescriptor(roiDialogTextResized, 8, 8)

        framePoint.setFlag(OutlineStrategy.FlagIndex.Dialog, hasDialog)
        framePoint.setFlag(OutlineStrategy.FlagIndex.DialogFeat, dctFeat)

        return False

    def ocrPass(self, frame: cv.Mat, fastMode: bool = False) -> typing.Tuple[cv.Mat, cv.Mat]:
        debugFrame = None

        textWeightMin = self.textWeightMin
        textWeightMax = self.textWeightMax
        textHSVRanges = [np.array([range[:3], range[3:]]) for range in self.textHSVRanges]
        outlineWeightMax = self.outlineWeightMax
        outlineHSVRanges = [np.array([range[:3], range[3:]]) for range in self.outlineHSVRanges]
        boundCompensation = self.boundCompensation
        sobelThreshold = self.sobelThreshold
        nestingSuppression = self.nestingSuppression

        roiDialog = self.dialogRect.cutRoiToUmat(frame)
        roiDialogHSV = cv.cvtColor(roiDialog, cv.COLOR_BGR2HSV)

        roiDialogText = None
        for textHSVRange in textHSVRanges:
            temp = cv.inRange(roiDialogHSV, textHSVRange[0], textHSVRange[1])
            if roiDialogText is None:
                roiDialogText = temp
            else:
                roiDialogText = cv.bitwise_or(roiDialogText, temp)

        if self.debugLevel == 1:
            debugFrame = roiDialogText

        roiDialogOutline = None
        for outlineHSVRange in outlineHSVRanges:
            temp = cv.inRange(roiDialogHSV, outlineHSVRange[0], outlineHSVRange[1])
            if roiDialogOutline is None:
                roiDialogOutline = temp
            else:
                roiDialogOutline = cv.bitwise_or(roiDialogOutline, temp)

        if self.debugLevel == 2:
            debugFrame = roiDialogOutline

        if not fastMode:
            roiDialogSobel = rgbSobel(roiDialog, ksize=3)
            roiDialogSobelBin = cv.threshold(roiDialogSobel, sobelThreshold, 255, cv.THRESH_BINARY)[1]
            roiDialogSobelBinClose = cv.morphologyEx(roiDialogSobelBin, cv.MORPH_CLOSE, kernel=cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3)))

        # roiDialogOutlineLB = morphologyWeightLowerBound(roiDialogOutline, erodeWeight=outlineWeightMin, dilateWeight=outlineWeightMin + boundCompensation)
        # roiDialogOutlineUB = morphologyWeightUpperBound(roiDialogOutlineLB, erodeWeight=outlineWeightMax, dilateWeight=outlineWeightMax + boundCompensation)
        roiDialogOutlineUB = roiDialogOutline

        if textWeightMin > 0:
            roiDialogTextLB = morphologyWeightLowerBound(roiDialogText, erodeWeight=textWeightMin, dilateWeight=textWeightMin + boundCompensation)
        else:
            roiDialogTextLB = roiDialog
        # roiDialogTextUB = morphologyWeightUpperBound(roiDialogTextLB, erodeWeight=textWeightMax, dilateWeight=textWeightMax + boundCompensation)
        roiDialogTextUB = roiDialogTextLB

        if not fastMode and nestingSuppression > 0:
            roiDialogOutlineOrText = cv.bitwise_or(roiDialogOutlineUB, roiDialogTextUB)
            roiDialogOutlineOrTextDialate = cv.dilate(roiDialogOutlineOrText, kernel=cv.getStructuringElement(cv.MORPH_ELLIPSE, (boundCompensation + 1, boundCompensation + 1)))
            roiDialogOutlineOrTextErode = cv.erode(roiDialogOutlineOrTextDialate, kernel=cv.getStructuringElement(cv.MORPH_ELLIPSE, (nestingSuppression, nestingSuppression)))
            roiDialogTextUB = cv.bitwise_and(roiDialogTextUB, roiDialogOutlineOrTextErode)

        if not fastMode:
            roiDialogOutlineNearSobel = morphologyNear(roiDialogOutlineUB, roiDialogSobelBinClose, outlineWeightMax)
            roiDialogTextNearSobel = morphologyNear(roiDialogTextUB, roiDialogSobelBinClose, textWeightMax)

            roiDialogOutlineNearSobelClose = cv.morphologyEx(roiDialogOutlineNearSobel, cv.MORPH_CLOSE, kernel=cv.getStructuringElement(cv.MORPH_ELLIPSE, (textWeightMax + boundCompensation, textWeightMax + boundCompensation)))
            roiDialogTextClosedByOutline = cv.bitwise_and(roiDialogOutlineNearSobelClose, roiDialogTextNearSobel)
        else:
            roiDialogTextNearOutline = morphologyNear(roiDialogTextUB, roiDialogOutlineUB, textWeightMax + boundCompensation)
            roiDialogTextClosedByOutline = roiDialogTextNearOutline

        if self.debugLevel == 3:
            debugFrame = roiDialogTextClosedByOutline

        return roiDialogTextClosedByOutline, debugFrame
