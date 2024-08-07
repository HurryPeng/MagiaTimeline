import typing
import enum
import collections

from Util import *
from Strategies.AbstractStrategy import *
from AbstractFlagIndex import *
from Rectangle import *
from IR import *

class OutlineStrategy(AbstractStrategy):
    class FlagIndex(AbstractFlagIndex):
        Dialog = enum.auto()
        DialogFeat = enum.auto()
        DialogFeatJump = enum.auto()

        @classmethod
        def getDefaultFlagsImpl(cls) -> typing.List[typing.Any]:
            return [False, np.zeros(64), False]

    def __init__(self, config: dict, contentRect: AbstractRectangle) -> None:
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
        roiDialogText, debugFrame = self.ocrPass(frame)

        framePoint.setDebugFrame(debugFrame)

        meanDialogText = cv.mean(roiDialogText)[0]

        roiDialogTextResized = cv.resize(roiDialogText, (150, 50)).get()
        dctFeat = dctDescriptor(roiDialogTextResized, 8, 8)

        framePoint.setFlag(OutlineStrategy.FlagIndex.Dialog, meanDialogText > 1.0)
        framePoint.setFlag(OutlineStrategy.FlagIndex.DialogFeat, dctFeat)

        return False

    def ocrPass(self, frame: cv.Mat) -> typing.Tuple[cv.Mat, cv.Mat]:
        debugFrame = None

        # # Yukkuri Museum
        # textWeightMin = 3
        # textWeightMax = 25
        # textHSVRanges = [((0, 0, 250), (180, 16, 255))]
        # outlineWeightMin = 1
        # outlineWeightMax = 15
        # outlineHSVRanges = [((0, 0, 0), (180, 255, 16))]
        # boundCompensation = 4
        # sobelThreshold = 250

        # # Yukkuri Kakueki
        # textWeightMin = 5
        # textWeightMax = 25
        # textHSVRanges = [
        #     ((0, 200, 128), (10, 255, 255)),
        #     ((170, 200, 128), (180, 255, 255)),
        #     ((115, 200, 128), (135, 255, 255))
        # ]
        # outlineWeightMin = 1
        # outlineWeightMax = 5
        # outlineHSVRanges = [((0, 0, 180), (180, 64, 255))]
        # boundCompensation = 4
        # sobelThreshold = 192

        # # Hotel Zundamon
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

        # Zunda House
        textWeightMin = 3
        textWeightMax = 17
        textHSVRanges = [
            ((155, 100, 200), (180, 200, 255)),
            ((55, 150, 150), (75, 220, 220))
        ]
        outlineWeightMin = 1
        outlineWeightMax = 13
        outlineHSVRanges = [((0, 0, 200), (180, 64, 255))]
        boundCompensation = 4
        sobelThreshold = 200

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

        roiDialogSobel = rgbSobel(roiDialog, ksize=3)
        roiDialogSobelBin = cv.threshold(roiDialogSobel, sobelThreshold, 255, cv.THRESH_BINARY)[1]

        roiDialogSobelBinClose = cv.morphologyEx(roiDialogSobelBin, cv.MORPH_CLOSE, kernel=cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3)))

        # roiDialogOutlineLB = morphologyWeightLowerBound(roiDialogOutline, erodeWeight=outlineWeightMin, dilateWeight=outlineWeightMin + boundCompensation)
        # roiDialogOutlineUB = morphologyWeightUpperBound(roiDialogOutlineLB, erodeWeight=outlineWeightMax, dilateWeight=outlineWeightMax + boundCompensation)
        roiDialogOutlineUB = roiDialogOutline
        
        # debugFrame = roiDialogOutlineUB

        roiDialogTextLB = morphologyWeightLowerBound(roiDialogText, erodeWeight=textWeightMin, dilateWeight=textWeightMin + boundCompensation)
        # roiDialogTextUB = morphologyWeightUpperBound(roiDialogTextLB, erodeWeight=textWeightMax, dilateWeight=textWeightMax + boundCompensation)
        roiDialogTextUB = roiDialogTextLB

        roiDialogOutlineNearSobel = morphologyNear(roiDialogOutlineUB, roiDialogSobelBinClose, outlineWeightMax)
        roiDialogTextNearSobel = morphologyNear(roiDialogTextUB, roiDialogSobelBinClose, textWeightMax)

        roiDialogOutlineNearSobelClose = cv.morphologyEx(roiDialogOutlineNearSobel, cv.MORPH_CLOSE, kernel=cv.getStructuringElement(cv.MORPH_ELLIPSE, (textWeightMax + boundCompensation, textWeightMax + boundCompensation)))
        roiDialogTextClosedByOutline = cv.bitwise_and(roiDialogOutlineNearSobelClose, roiDialogTextNearSobel)

        # roiDialogOutlineNearText = morphologyNear(roiDialogOutlineNearSobel, roiDialogTextClosedByOutline, outlineWeightMax)
        # roiDialogTextNearOutline = morphologyNear(roiDialogTextClosedByOutline, roiDialogOutlineNearText, textWeightMax)

        # roiDialogTextNearOutline = cv.bitwise_and(roiDialogTextNearOutline, roiDialogOutlineNearSobelClose)

        debugFrame = roiDialogTextClosedByOutline

        return roiDialogTextClosedByOutline, debugFrame
