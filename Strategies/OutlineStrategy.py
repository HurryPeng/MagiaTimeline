import typing
import enum
import collections

from Util import *
from Strategies.AbstractStrategy import *
from AbstractFlagIndex import *
from Rectangle import *
from IR import *

class OutlineStrategy(AbstractStrategy, SpeculativeStrategy, OcrStrategy):
    class FlagIndex(AbstractFlagIndex):
        Dialog = enum.auto()
        DialogFeat = enum.auto()
        DialogFeatJump = enum.auto()
        OcrFrame = enum.auto()

        @classmethod
        def getDefaultFlagsImpl(cls) -> typing.List[typing.Any]:
            return [False, np.zeros(64), False, None]

    def __init__(self, config: dict, contentRect: AbstractRectangle) -> None:
        SpeculativeStrategy.__init__(self)
        self.rectangles: collections.OrderedDict[str, AbstractRectangle] = collections.OrderedDict()
        for k, v in config.items():
            self.rectangles[k] = RatioRectangle(contentRect, *v)

        self.rectangles["dialogRect"] = self.rectangles["dialogRect"]

        self.dialogRect = self.rectangles["dialogRect"]

        self.cvPasses = [self.cvPassDialog]

        self.fpirPasses = collections.OrderedDict()

        self.fpirPasses["fpirPassDetectDialogJump"] = FPIRPassDetectFeatureJump(
            featFlag=OutlineStrategy.FlagIndex.DialogFeat,
            dstFlag=OutlineStrategy.FlagIndex.DialogFeatJump, 
            featOpMean=lambda feats : np.mean(feats, axis=0),
            featOpDist=lambda lhs, rhs : np.linalg.norm(lhs - rhs),
            threshDist=0.1,
            windowSize=5,
            featOpStd=lambda feats: np.mean(np.std(feats, axis=0)),
            threshStd=0.005
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
    
    def decideFeatureMerge(self, oldFeatures: typing.List[typing.Any], newFeature: typing.Any) -> bool:
        return np.linalg.norm(np.mean(oldFeatures, axis=0) - newFeature) < 0.1
    
    def cutOcrFrame(self, frame: cv.Mat) -> cv.Mat:
        return self.dialogRect.cutRoi(frame)
    
    def cutCleanOcrFrame(self, frame: cv.Mat) -> cv.Mat:
        return self.ocrPass(frame, fastMode=False)[0]

    def cvPassDialog(self, frame: cv.Mat, framePoint: FramePoint) -> bool:
        roiDialogText, debugFrame = self.ocrPass(frame, fastMode=False)

        framePoint.setDebugFrame(debugFrame)

        meanDialogText = cv.mean(roiDialogText)[0]

        hasDialog = meanDialogText > 1.0
        dctFeat = np.zeros(64)
        if hasDialog:
            roiDialogTextResized = cv.resize(roiDialogText, (150, 50)).get()
            dctFeat = dctDescriptor(roiDialogTextResized, 8, 8)

        framePoint.setFlag(OutlineStrategy.FlagIndex.Dialog, meanDialogText > 1.0)
        framePoint.setFlag(OutlineStrategy.FlagIndex.DialogFeat, dctFeat)

        return False

    def ocrPass(self, frame: cv.Mat, fastMode: bool = False) -> typing.Tuple[cv.Mat, cv.Mat]:
        debugFrame = None

        nestingSuppression = 0

        # # Yukkuri Museum
        # # dialogRect: [0.00, 1.00, 0.75, 1.00]
        # textWeightMin = 3
        # textWeightMax = 25
        # textHSVRanges = [((0, 0, 240), (180, 16, 255))]
        # outlineWeightMin = 1
        # outlineWeightMax = 15
        # outlineHSVRanges = [((0, 0, 0), (180, 255, 16))]
        # boundCompensation = 4
        # sobelThreshold = 250
        # nestingSuppression = 0

        # # Yukkuri Kakueki
        # # dialogRect: [0.00, 1.00, 0.75, 1.00]
        # textWeightMin = 5
        # textWeightMax = 25
        # textHSVRanges = [
        #     ((0, 200, 128), (30, 255, 255)),
        #     ((170, 200, 128), (180, 255, 255)),
        #     ((105, 100, 128), (135, 255, 255))
        # ]
        # outlineWeightMin = 1
        # outlineWeightMax = 5
        # outlineHSVRanges = [((0, 0, 180), (180, 64, 255))]
        # boundCompensation = 4
        # sobelThreshold = 192
        # nestingSuppression = 9

        # # Hotel Zundamon
        # # dialogRect: [0.00, 1.00, 0.75, 1.00]
        # textWeightMin = 5
        # textWeightMax = 19
        # textHSVRanges = [
        #     ((0, 200, 100), (10, 255, 180)),
        #     ((170, 200, 100), (180, 255, 180)),
        #     ((25, 100, 100), (55, 200, 200))
        # ]
        # outlineWeightMin = 1
        # outlineWeightMax = 5
        # outlineHSVRanges = [((0, 0, 200), (180, 32, 255))]
        # boundCompensation = 2
        # sobelThreshold = 200

        # # Zunda House
        # # dialogRect: [0.00, 1.00, 0.75, 1.00]
        # textWeightMin = 1
        # textWeightMax = 25
        # textHSVRanges = [
        #     ((155, 100, 200), (180, 200, 255)),
        #     ((55, 150, 150), (75, 220, 220))
        # ]
        # outlineWeightMin = 1
        # outlineWeightMax = 15
        # outlineHSVRanges = [((0, 0, 240), (180, 64, 255))]
        # boundCompensation = 4
        # sobelThreshold = 230
        # nestingSuppression = 23

        # # Shioneru
        # # dialogRect: [0.00, 1.00, 0.75, 1.00]
        # textWeightMin = 1
        # textWeightMax = 9
        # textHSVRanges = [((0, 0, 0), (180, 255, 16))]
        # outlineWeightMin = 1
        # outlineWeightMax = 15
        # outlineHSVRanges = [((0, 0, 200), (180, 64, 255))]
        # boundCompensation = 4
        # sobelThreshold = 200

        # # JapanTrafficLab
        # # dialogRect: [0.00, 1.00, 0.75, 1.00]
        # textWeightMin = 1
        # textWeightMax = 19
        # textHSVRanges = [((70, 180, 180), (100, 255, 255))]
        # outlineWeightMin = 1
        # outlineWeightMax = 15
        # outlineHSVRanges = [((95, 180, 180), (140, 255, 255))]
        # boundCompensation = 4
        # sobelThreshold = 100
        # nestingSuppression = 0

        # # Uemon
        # # dialogRect: [0.00, 1.00, 0.75, 1.00]
        # textWeightMin = 1
        # textWeightMax = 25
        # textHSVRanges = [((0, 0, 230), (180, 64, 255))]
        # # outlineWeightMin = 1
        # outlineWeightMax = 15
        # outlineHSVRanges = [((0, 0, 0), (180, 255, 128))]
        # boundCompensation = 4
        # sobelThreshold = 200
        # nestingSuppression = 0

        # # Haruki
        # # dialogRect: [0.00, 1.00, 0.75, 1.00]
        # textWeightMin = 1
        # textWeightMax = 25
        # textHSVRanges = [
        #     ((75, 100, 210), (95, 230, 255)),
        #     ((140, 100, 210), (180, 230, 255)),
        #     ((50, 100, 210), (70, 150, 255))
        # ]
        # # outlineWeightMin = 1
        # outlineWeightMax = 15
        # # outlineHSVRanges = [((0, 0, 0), (180, 255, 32))]
        # outlineHSVRanges = [((0, 0, 0), (180, 255, 32))]
        # boundCompensation = 4
        # sobelThreshold = 240
        # nestingSuppression = 0

        # Fushigi
        # dialogRect: [0.00, 1.00, 0.75, 1.00]
        textWeightMin = 3
        textWeightMax = 29
        textHSVRanges = [
            ((0, 200, 180), (10, 255, 255)),
            ((160, 200, 180), (180, 255, 255)),
            ((15, 150, 200), (45, 255, 255)),
        ]
        outlineWeightMin = 1
        outlineWeightMax = 11
        outlineHSVRanges = [
            ((0, 0, 180), (180, 64, 255)),
            ((0, 0, 0), (180, 255, 16))
        ]
        boundCompensation = 4
        sobelThreshold = 150
        nestingSuppression = 0

        roiDialog = self.dialogRect.cutRoiToUmat(frame)
        roiDialogHSV = cv.cvtColor(roiDialog, cv.COLOR_BGR2HSV)

        roiDialogText = None
        for textHSVRange in textHSVRanges:
            temp = cv.inRange(roiDialogHSV, textHSVRange[0], textHSVRange[1])
            if roiDialogText is None:
                roiDialogText = temp
            else:
                roiDialogText = cv.bitwise_or(roiDialogText, temp)

        roiDialogOutline = None
        for outlineHSVRange in outlineHSVRanges:
            temp = cv.inRange(roiDialogHSV, outlineHSVRange[0], outlineHSVRange[1])
            if roiDialogOutline is None:
                roiDialogOutline = temp
            else:
                roiDialogOutline = cv.bitwise_or(roiDialogOutline, temp)

        if not fastMode:
            roiDialogSobel = rgbSobel(roiDialog, ksize=3)
            roiDialogSobelBin = cv.threshold(roiDialogSobel, sobelThreshold, 255, cv.THRESH_BINARY)[1]
            roiDialogSobelBinClose = cv.morphologyEx(roiDialogSobelBin, cv.MORPH_CLOSE, kernel=cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3)))

        # roiDialogOutlineLB = morphologyWeightLowerBound(roiDialogOutline, erodeWeight=outlineWeightMin, dilateWeight=outlineWeightMin + boundCompensation)
        # roiDialogOutlineUB = morphologyWeightUpperBound(roiDialogOutlineLB, erodeWeight=outlineWeightMax, dilateWeight=outlineWeightMax + boundCompensation)
        roiDialogOutlineUB = roiDialogOutline

        roiDialogTextLB = morphologyWeightLowerBound(roiDialogText, erodeWeight=textWeightMin, dilateWeight=textWeightMin + boundCompensation)
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

        debugFrame = roiDialogTextClosedByOutline

        return roiDialogTextClosedByOutline, debugFrame
