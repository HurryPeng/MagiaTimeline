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
        DialogVal = enum.auto()
        DialogValJump = enum.auto()

        @classmethod
        def getDefaultFlagsImpl(cls) -> typing.List[typing.Any]:
            return [False, (0.0, 0.0), False]

    def __init__(self, config: dict, contentRect: AbstractRectangle) -> None:
        self.rectangles: collections.OrderedDict[str, AbstractRectangle] = collections.OrderedDict()
        for k, v in config.items():
            self.rectangles[k] = RatioRectangle(contentRect, *v)

        self.rectangles["dialogRect"] = self.rectangles["dialogRect"]

        self.dialogRect = self.rectangles["dialogRect"]

        self.cvPasses = [self.cvPassDialog]

        self.fpirPasses = collections.OrderedDict()

        self.fpirPasses["fpirPassDetectDialogJump"] = FPIRPassDetectFeatureJump(
            featFlag=ParakoStrategy.FlagIndex.DialogVal,
            dstFlag=ParakoStrategy.FlagIndex.DialogValJump, 
            featOpMean=lambda feats : np.mean(feats, 0),
            featOpDist=lambda lhs, rhs : np.linalg.norm(lhs - rhs),
            threshDist=1,
            windowSize=2
        )

        def breakDialogJump(framePoint: FramePoint):
            framePoint.setFlag(ParakoStrategy.FlagIndex.Dialog,
                framePoint.getFlag(ParakoStrategy.FlagIndex.Dialog)
                and not framePoint.getFlag(ParakoStrategy.FlagIndex.DialogValJump)
            )
        self.fpirPasses["fpirPassBreakDialogJump"] = FPIRPassFramewiseFunctional(
            func=breakDialogJump
        )
        
        self.fpirToIirPasses = collections.OrderedDict()
        self.fpirToIirPasses["fpirPassBuildIntervals"] = FPIRPassBooleanBuildIntervals(
            ParakoStrategy.FlagIndex.Dialog
        )

        self.iirPasses = collections.OrderedDict()
        self.iirPasses["iirPassFillGapDialog"] = IIRPassFillGap(ParakoStrategy.FlagIndex.Dialog, 300, meetPoint=1.5)

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

        _, roiDialogShade = cv.threshold(roiDialogGray, 230, 255, cv.THRESH_BINARY)

        meanStdDevDialogShade = cv.meanStdDev(roiDialogShade)
        meanDialogShade = meanStdDevDialogShade[0][0][0]
        devDialogShade = meanStdDevDialogShade[1][0][0]

        framePoint.setFlag(ParakoStrategy.FlagIndex.Dialog, meanDialogShade > 2.0)
        framePoint.setFlag(ParakoStrategy.FlagIndex.DialogVal, (meanDialogShade, devDialogShade))
        
        # framePoint.setDebugFrame(roiDialogShade)

        return False
