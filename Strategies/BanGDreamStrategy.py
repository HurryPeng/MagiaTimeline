import typing
import enum
import collections

from Util import *
from Strategies.AbstractStrategy import *
from AbstractFlagIndex import *
from Rectangle import *
from IR import *

class BanGDreamStrategy(AbstractFramewiseStrategy):
    class FlagIndex(AbstractFlagIndex):
        Dialog = enum.auto()
        DialogBgVal = enum.auto()
        DialogTextVal = enum.auto()
        DialogTextJump = enum.auto()

        @classmethod
        def getDefaultFlagsImpl(cls) -> typing.List[typing.Any]:
            return [False, 0.0, 0.0, False]

    def __init__(self, config: dict, contentRect: AbstractRectangle) -> None:
        AbstractStrategy.__init__(self, contentRect)
        self.rectangles: collections.OrderedDict[str, AbstractRectangle] = collections.OrderedDict()
        self.rectangles["dialogRect"] = RatioRectangle(contentRect, *config["dialogRect"])

        self.dialogRect = self.rectangles["dialogRect"]
        self.jumpThreshDist: float = config["jumpThreshDist"]
        self.fillGapMaxGap: int = config["fillGapMaxGap"]
        self.fillGapMeetPoint: float = config["fillGapMeetPoint"]
        self.bgMinBrightness: int = config["bgMinBrightness"]
        self.bgMaxBrightness: int = config["bgMaxBrightness"]
        self.bgMinMeanVal: float = config["bgMinMeanVal"]
        self.bgMaxMeanVal: float = config["bgMaxMeanVal"]
        self.textMinBrightness: int = config["textMinBrightness"]
        self.textMaxBrightness: int = config["textMaxBrightness"]
        self.textMinMeanVal: float = config["textMinMeanVal"]
        self.textMaxMeanVal: float = config["textMaxMeanVal"]

        self.cvPasses = [self.cvPassDialog]

        self.fpirPasses = collections.OrderedDict()

        self.fpirPasses["fpirPassDetectDialogJump"] = FPIRPassDetectFeatureJump(
            featFlag=BanGDreamStrategy.FlagIndex.DialogTextVal,
            dstFlag=BanGDreamStrategy.FlagIndex.DialogTextJump, 
            featOpMean=lambda feats : np.mean(feats, 0),
            featOpDist=lambda lhs, rhs : abs(lhs-rhs),
            threshDist=self.jumpThreshDist
        )

        def breakDialogJump(framePoint: FramePoint):
            framePoint.setFlag(BanGDreamStrategy.FlagIndex.Dialog,
                framePoint.getFlag(BanGDreamStrategy.FlagIndex.Dialog)
                and not framePoint.getFlag(BanGDreamStrategy.FlagIndex.DialogTextJump)
            )
        self.fpirPasses["fpirPassBreakDialogJump"] = FPIRPassFramewiseFunctional(
            func=breakDialogJump
        )
        
        self.fpirToIirPasses = collections.OrderedDict()
        self.fpirToIirPasses["fpirPassBuildIntervals"] = FPIRPassBooleanBuildIntervals(
            BanGDreamStrategy.FlagIndex.Dialog
        )

        self.iirPasses = collections.OrderedDict()
        self.iirPasses["iirPassFillGapDialog"] = IIRPassFillGap(BanGDreamStrategy.FlagIndex.Dialog, self.fillGapMaxGap, meetPoint=self.fillGapMeetPoint)

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
        roiDialog = self.dialogRect.cutRoiToUmat(frame)
        roiDialogGray = cv.cvtColor(roiDialog, cv.COLOR_BGR2GRAY)

        roiDialogBinBg = cv.inRange(roiDialogGray, self.bgMinBrightness, self.bgMaxBrightness)
        meanDialogBinBg = cv.mean(roiDialogBinBg)[0]

        roiDialogBinText = cv.inRange(roiDialogGray, self.textMinBrightness, self.textMaxBrightness)
        meanDialogBinText = cv.mean(roiDialogBinText)[0]

        hasDialog = meanDialogBinBg >= self.bgMinMeanVal and meanDialogBinBg <= self.bgMaxMeanVal \
                    and meanDialogBinText >= self.textMinMeanVal and meanDialogBinText <= self.textMaxMeanVal

        framePoint.setFlag(BanGDreamStrategy.FlagIndex.Dialog, hasDialog)
        framePoint.setFlag(BanGDreamStrategy.FlagIndex.DialogBgVal, meanDialogBinBg)
        framePoint.setFlag(BanGDreamStrategy.FlagIndex.DialogTextVal, meanDialogBinText)

        return False
