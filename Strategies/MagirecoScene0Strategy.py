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

        _, roiDialogShade1BinFix = cv.threshold(roiDialogGray, 50, 255, cv.THRESH_BINARY)
        roiDialogShade1BinAdap = cv.adaptiveThreshold(roiDialogGray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 7, 3)
        roiDialogShade1Bin = cv.bitwise_and(roiDialogShade1BinFix, roiDialogShade1BinAdap)
        
        cc1Num, cc1Labels, cc1Stats, cc1Centroids = cv.connectedComponentsWithStats(roiDialogShade1Bin, connectivity=4)
        roiDialogText1Bin = roiDialogShade1Bin
        for n in range(cc1Num):
            stat = cc1Stats[n]
            if not(stat[4] > 20 and stat[4] < 300 and stat[2] < 25 and stat[3] < 25 and (stat[2] > 3 and stat[4] > 3)):
                roiDialogText1Bin[cc1Labels == n] = 0
        roiDialogText1BinDialate = cv.morphologyEx(roiDialogText1Bin, cv.MORPH_DILATE, kernel=cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5)))

        _, roiDialogShade2BinFix = cv.threshold(roiDialogGray, 40, 255, cv.THRESH_BINARY_INV)
        roiDialogShade2BinAdap = cv.adaptiveThreshold(roiDialogGray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 7, 6)
        roiDialogShade2Bin = cv.bitwise_and(roiDialogShade2BinFix, roiDialogShade2BinAdap)
        roiDialogShade2BinFiltered = cv.bitwise_and(roiDialogShade2Bin, roiDialogText1BinDialate)

        roiDialogShade2BinFilteredClose = cv.morphologyEx(roiDialogShade2BinFiltered, cv.MORPH_CLOSE, kernel=cv.getStructuringElement(cv.MORPH_ELLIPSE, (11, 11)))
        
        roiDialogText2Bin = cv.bitwise_and(roiDialogShade2BinFilteredClose, roiDialogText1Bin)

        cc2Num, cc2Labels, cc2Stats, cc2Centroids = cv.connectedComponentsWithStats(roiDialogText2Bin, connectivity=4)
        roiDialogText3Bin = roiDialogText2Bin
        cc2LeagalAreaSum: int = 0
        for n in range(cc2Num):
            stat = cc2Stats[n]
            if (stat[4] > 20 and stat[4] < 300 and stat[2] < 25 and stat[3] < 25 and (stat[2] > 3 and stat[4] > 3)):
                cc2LeagalAreaSum += stat[4]
        cc2LeagalAreaRatio: float = cc2LeagalAreaSum / self.dialogRect.getArea() * 255

        # roiDialogText1BinDialate = cv.morphologyEx(roiDialogText1Bin, cv.MORPH_DILATE, kernel=cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3)))
        # roiDialogText2BinOpen = cv.morphologyEx(roiDialogText2Bin, cv.MORPH_OPEN, kernel=cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3)))
        
        # roiDialogText2Bin = cv.bitwise_and(roiDialogText1Bin, roiDialogText2BinOpen)

        framePoint.setDebugFrame(roiDialogText3Bin)

        # roiDialogShade2Bin = cv.bitwise_and(roiDialogShade1Bin, roiDialogText1BinDialate)
        # roiDialogShade2BinDialate = cv.morphologyEx(roiDialogShade1Bin, cv.MORPH_CLOSE, kernel=cv.getStructuringElement(cv.MORPH_ELLIPSE, (11, 11)))

        # roiDialogText2Bin = cv.bitwise_and(roiDialogShade2BinDialate, roiDialogText1Bin)

        # roiDialogCompoundBin = cv.bitwise_xor(roiDialogText1BinOpen, roiDialogShade2Bin)

        hasDialog: bool = cc2LeagalAreaRatio > 3

        framePoint.setDebugFlag(cc2LeagalAreaRatio, cc1Num)
        # framePoint.setDebugFlag(meanDialogText1BinOpen, num, stats)

        framePoint.setFlag(MagirecoScene0Strategy.FlagIndex.Dialog, hasDialog)
        framePoint.setFlag(MagirecoScene0Strategy.FlagIndex.DialogVal, cc2LeagalAreaRatio)
        return hasDialog
