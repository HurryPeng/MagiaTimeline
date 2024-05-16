import typing
import enum
import collections

from Util import *
from Strategies.AbstractStrategy import *
from AbstractFlagIndex import *
from Rectangle import *
from IR import *

class ParakoStrategy(AbstractStrategy):
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
            featFlag=ParakoStrategy.FlagIndex.DialogFeat,
            dstFlag=ParakoStrategy.FlagIndex.DialogFeatJump, 
            featOpMean=lambda feats : np.mean(feats, 0),
            featOpDist=lambda lhs, rhs : np.linalg.norm(lhs - rhs),
            threshDist=0.1,
            windowSize=3
        )

        def breakDialogJump(framePoint: FramePoint):
            framePoint.setFlag(ParakoStrategy.FlagIndex.Dialog,
                framePoint.getFlag(ParakoStrategy.FlagIndex.Dialog)
                and not framePoint.getFlag(ParakoStrategy.FlagIndex.DialogFeatJump)
            )
        self.fpirPasses["fpirPassBreakDialogJump"] = FPIRPassFramewiseFunctional(
            func=breakDialogJump
        )
        
        self.fpirToIirPasses = collections.OrderedDict()
        self.fpirToIirPasses["fpirPassBuildIntervals"] = FPIRPassBooleanBuildIntervals(
            ParakoStrategy.FlagIndex.Dialog
        )

        self.iirPasses = collections.OrderedDict()
        self.iirPasses["iirPassFillGapDialog"] = IIRPassFillGap(ParakoStrategy.FlagIndex.Dialog, 300, meetPoint=1.3)

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
        roiDialogHSV = cv.cvtColor(roiDialog, cv.COLOR_BGR2HSV)

        # Select out the whitest part of the dialog
        roiDialogShade = cv.inRange(roiDialogHSV, (0, 0, 240), (180, 16, 255))

        meanStdDevDialogShade = cv.meanStdDev(roiDialogShade)
        meanDialogShade = meanStdDevDialogShade[0][0][0]
        # devDialogShade = meanStdDevDialogShade[1][0][0]

        roiDialogShadeResized = cv.resize(roiDialogShade, (150, 50))
        dctFeat = dctDescriptor(roiDialogShadeResized, 8, 8)

        framePoint.setFlag(ParakoStrategy.FlagIndex.Dialog, meanDialogShade > 1.0)
        framePoint.setFlag(ParakoStrategy.FlagIndex.DialogFeat, dctFeat)

        # framePoint.setDebugFrame(roiDialogShadeResized)

        return False
