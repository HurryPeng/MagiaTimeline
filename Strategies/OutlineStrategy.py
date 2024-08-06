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
        roiDialogText = self.ocrPass(frame)

        meanDialogText = cv.mean(roiDialogText)[0]

        roiDialogTextResized = cv.resize(roiDialogText, (150, 50)).get()
        dctFeat = dctDescriptor(roiDialogTextResized, 8, 8)

        framePoint.setFlag(OutlineStrategy.FlagIndex.Dialog, meanDialogText > 1.0)
        framePoint.setFlag(OutlineStrategy.FlagIndex.DialogFeat, dctFeat)

        return False

    def ocrPass(self, frame: cv.Mat) -> cv.Mat:
        roiDialog = self.dialogRect.cutRoiToUmat(frame)
        roiDialogHSV = cv.cvtColor(roiDialog, cv.COLOR_BGR2HSV)

        roiDialogText = cv.inRange(roiDialogHSV, (0, 0, 250), (180, 16, 255))
        roiDialogOutline = cv.inRange(roiDialogHSV, (0, 0, 0), (180, 255, 16))

        roiDialogSobel = rgbSobel(roiDialog, ksize=3)
        roiDialogSobelBin = cv.threshold(roiDialogSobel, 250, 255, cv.THRESH_BINARY)[1]

        roiDialogSobelBinClose = cv.morphologyEx(roiDialogSobelBin, cv.MORPH_CLOSE, kernel=cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3)))

        # roiDialogOutlineLB = morphologyWidthLowerBound(roiDialogOutline, erodeWidth=1, dilateWidth=5)
        # roiDialogOutlineUB = morphologyWidthUpperBound(roiDialogOutlineLB, erodeWidth=15, dilateWidth=19)
        roiDialogOutlineUB = roiDialogOutline

        roiDialogTextLB = morphologyWidthLowerBound(roiDialogText, erodeWidth=3, dilateWidth=7)
        # roiDialogTextUB = morphologyWidthUpperBound(roiDialogTextLB, erodeWidth=25, dilateWidth=29)
        roiDialogTextUB = roiDialogTextLB

        roiDialogOutlineNearSobel = morphologyNear(roiDialogOutlineUB, roiDialogSobelBinClose, 15)
        roiDialogTextNearSobel = morphologyNear(roiDialogTextUB, roiDialogSobelBinClose, 25)

        roiDialogOutlineNearSobelClose = cv.morphologyEx(roiDialogOutlineNearSobel, cv.MORPH_CLOSE, kernel=cv.getStructuringElement(cv.MORPH_ELLIPSE, (29, 29)))
        roiDialogTextClosedByOutline = cv.bitwise_and(roiDialogOutlineNearSobelClose, roiDialogTextNearSobel)

        # roiDialogOutlineNearText = morphologyNear(roiDialogOutlineNearSobel, roiDialogTextClosedByOutline, 15)
        # roiDialogTextNearOutline = morphologyNear(roiDialogTextClosedByOutline, roiDialogOutlineNearText, 25)

        # roiDialogTextNearOutline = cv.bitwise_and(roiDialogTextNearOutline, roiDialogOutlineNearSobelClose)

        return roiDialogTextClosedByOutline
