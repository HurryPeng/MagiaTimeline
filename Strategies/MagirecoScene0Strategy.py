import typing
import enum
import collections

from Util import *
from Strategies.AbstractStrategy import *
from AbstractFlagIndex import *
from Rectangle import *
from IR import *

class MagirecoScene0Strategy(AbstractStrategy):
    class FlagIndex(AbstractFlagIndex):
        Dialog = enum.auto()
        DialogVal = enum.auto()
        DialogValJump = enum.auto()

        @classmethod
        def getDefaultFlagsImpl(cls) -> typing.List[typing.Any]:
            return [False] * cls.getNum()

    def __init__(self, config: dict, contentRect: AbstractRectangle) -> None:
        self.rectangles: collections.OrderedDict[str, AbstractRectangle] = collections.OrderedDict()
        for k, v in config.items():
            self.rectangles[k] = RatioRectangle(contentRect, *v)

        self.dialogRect = self.rectangles["dialogRect"]

        self.cvPasses = [self.cvPassDialog]

        self.fpirPasses = collections.OrderedDict()
        self.fpirPasses["fpirPassDetectFeatureJump"] = FPIRPassDetectFeatureJump(
            featFlag=MagirecoScene0Strategy.FlagIndex.DialogVal,
            dstFlag=MagirecoScene0Strategy.FlagIndex.DialogValJump, 
            featOpMean=lambda feats : np.mean(feats, 0),
            featOpDist=lambda lhs, rhs : abs(lhs-rhs),
            threshDist=10
        )
        def breakDialogJump(framePoint: FramePoint):
            framePoint.setFlag(MagirecoScene0Strategy.FlagIndex.Dialog,
                framePoint.getFlag(MagirecoScene0Strategy.FlagIndex.Dialog)
                and not framePoint.getFlag(MagirecoScene0Strategy.FlagIndex.DialogValJump)
            )
        self.fpirPasses["fpirPassBreakDialogJump"] = FPIRPassFramewiseFunctional(
            func=breakDialogJump
        )
        self.fpirPasses["fpirPassRemoveNoiseDialogFalse"] = FPIRPassBooleanRemoveNoise(MagirecoScene0Strategy.FlagIndex.Dialog, False, 2)
        self.fpirPasses["fpirPassRemoveNoiseDialogTrue"] = FPIRPassBooleanRemoveNoise(MagirecoScene0Strategy.FlagIndex.Dialog, True, 10)

        self.fpirToIirPasses = collections.OrderedDict()
        self.fpirToIirPasses["fpirPassBuildIntervals"] = FPIRPassBooleanBuildIntervals(
            MagirecoScene0Strategy.FlagIndex.Dialog, 
        )

        self.iirPasses = collections.OrderedDict()
        self.iirPasses["iirPassFillGapDialog"] = IIRPassFillGap(MagirecoScene0Strategy.FlagIndex.Dialog, 500)

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
        # roiDialogHSV = cv.cvtColor(roiDialog, cv.COLOR_BGR2HSV)

        _, roiDialogShade0BinFix = cv.threshold(roiDialogGray, 50, 255, cv.THRESH_BINARY_INV)
        roiDialogShade0BinAdap = cv.adaptiveThreshold(roiDialogGray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 7, 3)
        roiDialogShade0Bin = cv.bitwise_or(roiDialogShade0BinFix, roiDialogShade0BinAdap)
        ccnum, cclabels, ccstats, cccentroids = cv.connectedComponentsWithStats(cv.bitwise_not(roiDialogShade0Bin), connectivity=4)

        roiDialogCCFilterBin = roiDialogShade0Bin
        for n in range(ccnum):
            stat = ccstats[n]
            if not(stat[4] > 20 and stat[4] < 300 and stat[2] < 25 and stat[3] < 25):
                roiDialogCCFilterBin[cclabels == n] = 255

        _, roiDialogShade1BinFix = cv.threshold(roiDialogGray, 40, 255, cv.THRESH_BINARY_INV)
        roiDialogShade1BinAdap = cv.adaptiveThreshold(roiDialogGray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 7, 6)
        roiDialogShade1Bin = cv.bitwise_and(roiDialogShade1BinFix, roiDialogShade1BinAdap)

        roiDialogShade1BinClose = cv.morphologyEx(roiDialogShade1Bin, cv.MORPH_CLOSE, kernel=cv.getStructuringElement(cv.MORPH_ELLIPSE, (11, 11)))
        
        roiDialogText1Bin = cv.bitwise_and(roiDialogShade1BinClose, cv.bitwise_not(roiDialogShade1Bin))

        # roiDialogText1BinDialate = cv.morphologyEx(roiDialogText1Bin, cv.MORPH_DILATE, kernel=cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3)))
        roiDialogText1BinOpen = cv.morphologyEx(roiDialogText1Bin, cv.MORPH_OPEN, kernel=cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3)))
        
        roiDialogText2Bin = cv.bitwise_and(cv.bitwise_not(roiDialogCCFilterBin), roiDialogText1BinOpen)

        framePoint.setDebugFrame(roiDialogText2Bin)

        # roiDialogShade2Bin = cv.bitwise_and(roiDialogShade1Bin, roiDialogText1BinDialate)
        # roiDialogShade2BinDialate = cv.morphologyEx(roiDialogShade1Bin, cv.MORPH_CLOSE, kernel=cv.getStructuringElement(cv.MORPH_ELLIPSE, (11, 11)))

        # roiDialogText2Bin = cv.bitwise_and(roiDialogShade2BinDialate, roiDialogText1Bin)

        # roiDialogCompoundBin = cv.bitwise_xor(roiDialogText1BinOpen, roiDialogShade2Bin)

        meanDialogText2Bin: float = cv.mean(roiDialogText2Bin)[0]
        hasDialog: bool = meanDialogText2Bin > 3

        framePoint.setDebugFlag(meanDialogText2Bin, ccnum)
        # framePoint.setDebugFlag(meanDialogText1BinOpen, num, stats)

        framePoint.setFlag(MagirecoScene0Strategy.FlagIndex.Dialog, hasDialog)
        framePoint.setFlag(MagirecoScene0Strategy.FlagIndex.DialogVal, meanDialogText2Bin)
        return hasDialog
