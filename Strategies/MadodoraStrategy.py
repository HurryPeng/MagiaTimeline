import typing
import enum
import collections

from Util import *
from Strategies.AbstractStrategy import *
from AbstractFlagIndex import *
from Rectangle import *
from IR import *

class MadodoraStrategy(AbstractFramewiseStrategy):
    class FlagIndex(AbstractFlagIndex):
        HomeDialog = enum.auto()
        HomeDialogBg = enum.auto()
        HomeDialogText = enum.auto()
        HomeDialogUnder = enum.auto()

        Dialog = enum.auto()
        DialogBg = enum.auto()
        DialogText = enum.auto()

        Whitescreen = enum.auto()
        WhitescreenBg = enum.auto()
        WhitescreenText = enum.auto()

        Blackscreen = enum.auto()
        BlackscreenBg = enum.auto()
        BlackscreenText = enum.auto()

        LeftBubble = enum.auto()
        LeftBubbleBg = enum.auto()
        LeftBubbleText = enum.auto()

        RightBubble = enum.auto()
        RightBubbleBg = enum.auto()
        RightBubbleText = enum.auto()

        @classmethod
        def getDefaultFlagsImpl(cls) -> typing.List[typing.Any]:
            return [False] * cls.getNum()

    def __init__(self, config: dict, contentRect: AbstractRectangle) -> None:
        AbstractStrategy.__init__(self, contentRect)
        self.rectangles: collections.OrderedDict[str, AbstractRectangle] = collections.OrderedDict()
        for k, v in config.items():
            self.rectangles[k] = RatioRectangle(contentRect, *v)

        self.homeDialogRect = self.rectangles["homeDialogRect"]
        self.underHomeDialogRect = self.rectangles["underHomeDialogRect"]
        self.dialogRect = self.rectangles["dialogRect"]
        self.whitescreenRect = self.rectangles["whitescreenRect"]
        self.blackscreenRect = self.rectangles["blackscreenRect"]
        self.leftBubbleRect = self.rectangles["leftBubbleRect"]
        self.rightBubbleRect = self.rectangles["rightBubbleRect"]

        self.cvPasses = [
            self.cvPassHomeDialog,
            self.cvPassDialog,
            self.cvPassWhitescreen,
            self.cvPassBlackscreen,
            self.cvPassLeftBubble,
            self.cvPassRightBubble,
        ]

        self.fpirPasses = collections.OrderedDict()

        self.fpirPasses["fpirPassRemoveNoiseHomeDialogTrue"] = FPIRPassBooleanRemoveNoise(MadodoraStrategy.FlagIndex.HomeDialog, True, 10)

        self.fpirPasses["fpirPassRemoveNoiseDialogTrue"] = FPIRPassBooleanRemoveNoise(MadodoraStrategy.FlagIndex.Dialog, True, 10)

        self.fpirPasses["fpirPassPrioritizeHomeDialog"] = FPIRPassFramewiseFunctional(
            lambda framePoint: framePoint.setFlag(
                MadodoraStrategy.FlagIndex.Dialog,
                framePoint.getFlag(MadodoraStrategy.FlagIndex.Dialog)
                    and not framePoint.getFlag(MadodoraStrategy.FlagIndex.HomeDialog)
            )
        )

        self.fpirToIirPasses = collections.OrderedDict()
        self.fpirToIirPasses["fpirPassBuildIntervals"] = FPIRPassBooleanBuildIntervals(
            MadodoraStrategy.FlagIndex.HomeDialog,
            MadodoraStrategy.FlagIndex.Dialog,
            MadodoraStrategy.FlagIndex.Whitescreen,
            MadodoraStrategy.FlagIndex.Blackscreen,
            MadodoraStrategy.FlagIndex.LeftBubble,
            MadodoraStrategy.FlagIndex.RightBubble,
        )

        self.iirPasses = collections.OrderedDict()
        self.iirPasses["iirPassFillGapHomeDialog"] = IIRPassFillGap(MadodoraStrategy.FlagIndex.HomeDialog, 300, 0.0)
        self.iirPasses["iirPassExtendHomeDialog"] = IIRPassExtend(MadodoraStrategy.FlagIndex.HomeDialog, 300, 0)
        self.iirPasses["iirPassFillGapDialog"] = IIRPassFillGap(MadodoraStrategy.FlagIndex.Dialog, 300, 0.0)
        self.iirPasses["iirPassExtendDialog"] = IIRPassExtend(MadodoraStrategy.FlagIndex.Dialog, 300, 0)
        self.iirPasses["iirPassFillGapWhitescreen"] = IIRPassFillGap(MadodoraStrategy.FlagIndex.Whitescreen, 2000, 0.5)
        self.iirPasses["iirPassExtendWhitescreen"] = IIRPassExtend(MadodoraStrategy.FlagIndex.Whitescreen, 400, 400)
        self.iirPasses["iirPassFillGapBlackscreen"] = IIRPassFillGap(MadodoraStrategy.FlagIndex.Blackscreen, 2000, 0.5)
        self.iirPasses["iirPassExtendBlackscreen"] = IIRPassExtend(MadodoraStrategy.FlagIndex.Blackscreen, 400, 400)
        self.iirPasses["iirPassFillGapLeftBubble"] = IIRPassFillGap(MadodoraStrategy.FlagIndex.LeftBubble, 300, 0.0)
        self.iirPasses["iirPassExtendLeftBubble"] = IIRPassExtend(MadodoraStrategy.FlagIndex.LeftBubble, 200, 0)
        self.iirPasses["iirPassFillGapRightBubble"] = IIRPassFillGap(MadodoraStrategy.FlagIndex.RightBubble, 300, 0.0)
        self.iirPasses["iirPassExtendRightBubble"] = IIRPassExtend(MadodoraStrategy.FlagIndex.RightBubble, 200, 0)

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
    
    def cvPassHomeDialog(self, frame: cv.Mat, framePoint: FramePoint) -> bool:
        roiDialog = self.homeDialogRect.cutRoiToUmat(frame)
        roiDialogGray = cv.cvtColor(roiDialog, cv.COLOR_BGR2GRAY)
        _, roiDialogTextBin = cv.threshold(roiDialogGray, 192, 255, cv.THRESH_BINARY)
        _, roiDialogBgBin = cv.threshold(roiDialogGray, 100, 255, cv.THRESH_BINARY)
        meanDialogTextBin: float = cv.mean(roiDialogTextBin)[0]
        meanDialogBgBin: float = cv.mean(roiDialogBgBin)[0]
        hasDialogBg: bool = meanDialogBgBin < 25 and meanDialogBgBin > 0.1
        hasDialogText: bool = meanDialogTextBin < 20 and meanDialogTextBin > 0.1

        roiUnderDialog = self.underHomeDialogRect.cutRoiToUmat(frame)
        roiUnderDialogGray = cv.cvtColor(roiUnderDialog, cv.COLOR_BGR2GRAY)
        _, roiUnderDialogBin = cv.threshold(roiUnderDialogGray, 20, 255, cv.THRESH_BINARY)
        meanUnderDialogBin: float = cv.mean(roiUnderDialogBin)[0]
        varUnderDialogGray: float = cv.mean(cv.meanStdDev(roiUnderDialogGray)[1])[0]
        hasUnderBlackBar: bool = meanUnderDialogBin < 1 and varUnderDialogGray < 0.01

        isValidDialog = hasDialogBg and hasDialogText and hasUnderBlackBar

        framePoint.setFlag(MadodoraStrategy.FlagIndex.HomeDialog, isValidDialog)
        framePoint.setFlag(MadodoraStrategy.FlagIndex.HomeDialogBg, hasDialogBg)
        framePoint.setFlag(MadodoraStrategy.FlagIndex.HomeDialogText, hasDialogText)
        framePoint.setFlag(MadodoraStrategy.FlagIndex.HomeDialogUnder, hasUnderBlackBar)

        return isValidDialog

    def cvPassDialog(self, frame: cv.Mat, framePoint: FramePoint) -> bool:
        roiDialog = self.dialogRect.cutRoiToUmat(frame)
        roiDialogGray = cv.cvtColor(roiDialog, cv.COLOR_BGR2GRAY)
        _, roiDialogTextBin = cv.threshold(roiDialogGray, 192, 255, cv.THRESH_BINARY)
        _, roiDialogBgBin = cv.threshold(roiDialogGray, 100, 255, cv.THRESH_BINARY)
        meanDialogTextBin: float = cv.mean(roiDialogTextBin)[0]
        meanDialogBgBin: float = cv.mean(roiDialogBgBin)[0]
        hasDialogBg: bool = meanDialogBgBin < 25 and meanDialogBgBin > 0.1
        hasDialogText: bool = meanDialogTextBin < 20 and meanDialogTextBin > 0.1

        isValidDialog = hasDialogBg and hasDialogText

        framePoint.setFlag(MadodoraStrategy.FlagIndex.Dialog, isValidDialog)
        framePoint.setFlag(MadodoraStrategy.FlagIndex.DialogBg, hasDialogBg)
        framePoint.setFlag(MadodoraStrategy.FlagIndex.DialogText, hasDialogText)

        return isValidDialog
    
    def cvPassWhitescreen(self, frame: cv.Mat, framePoint: FramePoint) -> bool:
        roiWhitescreen = self.whitescreenRect.cutRoiToUmat(frame)
        roiWhitescreenGray = cv.cvtColor(roiWhitescreen, cv.COLOR_BGR2GRAY)
        _, roiWhitescreenBgBin = cv.threshold(roiWhitescreenGray, 170, 255, cv.THRESH_BINARY_INV)
        _, roiWhitescreenTextBin = cv.threshold(roiWhitescreenGray, 20, 255, cv.THRESH_BINARY_INV)
        meanWhitescreenBgBin: float = cv.mean(roiWhitescreenBgBin)[0]
        meanWhitescreenTextBin: float = cv.mean(roiWhitescreenTextBin)[0]
        hasWhitescreenBg: bool = meanWhitescreenBgBin < 20
        hasWhitescreenText: bool = meanWhitescreenTextBin < 8 and meanWhitescreenTextBin > 0.1
        hasWhitescreen: bool = hasWhitescreenBg and hasWhitescreenText

        framePoint.setFlag(MadodoraStrategy.FlagIndex.WhitescreenBg, hasWhitescreenBg)
        framePoint.setFlag(MadodoraStrategy.FlagIndex.WhitescreenText, hasWhitescreenText)
        framePoint.setFlag(MadodoraStrategy.FlagIndex.Whitescreen, hasWhitescreen)

        return hasWhitescreenBg
    
    def cvPassBlackscreen(self, frame: cv.Mat, framePoint: FramePoint) -> bool:
        roiBlackscreen = self.blackscreenRect.cutRoiToUmat(frame)
        roiBlackscreenGray = cv.cvtColor(roiBlackscreen, cv.COLOR_BGR2GRAY)
        _, roiBlackscreenBlackBgBin = cv.threshold(roiBlackscreenGray, 10, 255, cv.THRESH_BINARY)
        _, roiBlackscreenDimBgBin = cv.threshold(roiBlackscreenGray, 100, 255, cv.THRESH_BINARY)
        _, roiBlackscreenTextBin = cv.threshold(roiBlackscreenGray, 240, 255, cv.THRESH_BINARY)
        roiBlackscreenTextBinDialate = cv.morphologyEx(roiBlackscreenTextBin, cv.MORPH_DILATE, kernel=cv.getStructuringElement(cv.MORPH_RECT, (9, 9)))
        roiBlackscreenDimBgBinExcludeText = cv.bitwise_and(roiBlackscreenDimBgBin, cv.bitwise_not(roiBlackscreenTextBinDialate))

        meanBlackscreenBlackBgBin: float = cv.mean(roiBlackscreenBlackBgBin)[0]
        meanBlackscreenDimBgBin: float = cv.mean(roiBlackscreenDimBgBinExcludeText)[0]
        meanBlackscreenTextBin: float = cv.mean(roiBlackscreenTextBin)[0]

        hasBlackscreenBg: bool = meanBlackscreenBlackBgBin < 15 or meanBlackscreenDimBgBin < 0.1
        hasBlackscreenText: bool = meanBlackscreenTextBin < 5 and meanBlackscreenTextBin > 0.1
        hasBlackscreen: bool = hasBlackscreenBg and hasBlackscreenText

        framePoint.setFlag(MadodoraStrategy.FlagIndex.BlackscreenBg, hasBlackscreenBg)
        framePoint.setFlag(MadodoraStrategy.FlagIndex.BlackscreenText, hasBlackscreenText)
        framePoint.setFlag(MadodoraStrategy.FlagIndex.Blackscreen, hasBlackscreen)

    def cvPassLeftBubble(self, frame: cv.Mat, framePoint: FramePoint) -> bool:
        roiLeftBubble: cv.UMat = self.leftBubbleRect.cutRoiToUmat(frame)
        # Around (213, 223, 229) +- 10
        roiLeftBubbleBgBin: cv.UMat = cv.inRange(roiLeftBubble, (203, 213, 219), (223, 233, 239))
        # Find from right to left the index of the first coloumn where over 90% of the pixels are white
        leftBubbleWidth, leftBubbleHeight = self.leftBubbleRect.getSizeInt()
        roiLeftBubbleBgBinColSum: cv.Mat = cv.reduce(roiLeftBubbleBgBin, 0, cv.REDUCE_SUM, dtype=cv.CV_32S).get()
        colBouldIdx = 0
        for i in range(leftBubbleWidth - 1, int(leftBubbleWidth * 0.3), -1): # Less than 30% of the width is cosidered nonexistent
            if roiLeftBubbleBgBinColSum[0][i] >= 255 * leftBubbleHeight * 0.9:
                colBouldIdx = i
                break

        if colBouldIdx == 0 or colBouldIdx >= leftBubbleWidth * 0.9:
            return False

        refinedRect = RatioRectangle(self.leftBubbleRect, 0.0, float(colBouldIdx) / leftBubbleWidth, 0.0, 1.0)
        roiRefined = refinedRect.cutRoiToUmat(frame)
        roiRefinedGray = cv.cvtColor(roiRefined, cv.COLOR_BGR2GRAY)
        roiRefinedBgBin = cv.inRange(roiRefined, (203, 213, 219), (223, 233, 239))
        _, roiRefinedTextBin = cv.threshold(roiRefinedGray, 150, 255, cv.THRESH_BINARY_INV)
        meanRefinedBgBin: float = 255 - cv.mean(roiRefinedBgBin)[0]
        meanRefinedTextBin: float = cv.mean(roiRefinedTextBin)[0]

        hasLeftBubbleBg: bool = meanRefinedBgBin < 70 and meanRefinedBgBin > 1
        hasLeftBubbleText: bool = meanRefinedTextBin < 30 and meanRefinedTextBin > 1
        hasLeftBubble: bool = hasLeftBubbleBg and hasLeftBubbleText

        framePoint.setFlag(MadodoraStrategy.FlagIndex.LeftBubble, hasLeftBubble)
        framePoint.setFlag(MadodoraStrategy.FlagIndex.LeftBubbleBg, hasLeftBubbleBg)
        framePoint.setFlag(MadodoraStrategy.FlagIndex.LeftBubbleText, hasLeftBubbleText)

        return hasLeftBubble

    def cvPassRightBubble(self, frame: cv.Mat, framePoint: FramePoint) -> bool:
        roiRightBubble: cv.UMat = self.rightBubbleRect.cutRoiToUmat(frame)
        # Around (213, 223, 229) +- 10
        roiRightBubbleBgBin: cv.UMat = cv.inRange(roiRightBubble, (203, 213, 219), (223, 233, 239))
        # Find from left to right the index of the first coloumn where over 90% of the pixels are white
        rightBubbleWidth, rightBubbleHeight = self.rightBubbleRect.getSizeInt()
        roiRightBubbleBgBinColSum: cv.Mat = cv.reduce(roiRightBubbleBgBin, 0, cv.REDUCE_SUM, dtype=cv.CV_32S).get()
        colBouldIdx = rightBubbleWidth - 1
        for i in range(0, int(rightBubbleWidth * 0.7)):
            if roiRightBubbleBgBinColSum[0][i] >= 255 * rightBubbleHeight * 0.9:
                colBouldIdx = i
                break
        
        if colBouldIdx == rightBubbleWidth - 1 or colBouldIdx <= rightBubbleWidth * 0.1:
            return False
        
        refinedRect = RatioRectangle(self.rightBubbleRect, float(colBouldIdx) / rightBubbleWidth, 1.0, 0.0, 1.0)
        roiRefined = refinedRect.cutRoiToUmat(frame)
        roiRefinedGray = cv.cvtColor(roiRefined, cv.COLOR_BGR2GRAY)
        roiRefinedBgBin = cv.inRange(roiRefined, (203, 213, 219), (223, 233, 239))
        _, roiRefinedTextBin = cv.threshold(roiRefinedGray, 150, 255, cv.THRESH_BINARY_INV)
        meanRefinedBgBin: float = 255 - cv.mean(roiRefinedBgBin)[0]
        meanRefinedTextBin: float = cv.mean(roiRefinedTextBin)[0]

        hasRightBubbleBg: bool = meanRefinedBgBin < 70 and meanRefinedBgBin > 1
        hasRightBubbleText: bool = meanRefinedTextBin < 30 and meanRefinedTextBin > 1
        hasRightBubble: bool = hasRightBubbleBg and hasRightBubbleText

        framePoint.setFlag(MadodoraStrategy.FlagIndex.RightBubble, hasRightBubble)
        framePoint.setFlag(MadodoraStrategy.FlagIndex.RightBubbleBg, hasRightBubbleBg)
        framePoint.setFlag(MadodoraStrategy.FlagIndex.RightBubbleText, hasRightBubbleText)

        return hasRightBubble
