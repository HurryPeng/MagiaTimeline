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
        Blackscreen = enum.auto()
        BlackscreenBg = enum.auto()
        BlackscreenText = enum.auto()

        Dialog = enum.auto()
        DialogNameVal = enum.auto()
        DialogNameValJump = enum.auto()
        DialogContentVal = enum.auto()
        DialogContentValJumpDown = enum.auto()
        DialogContentValJumpUp = enum.auto()
        DialogContentValJumpUpJumpUp = enum.auto()

        Balloon = enum.auto()
        BalloonVal = enum.auto()
        BalloonValJump = enum.auto()

        MeanTextColour = enum.auto()

        @classmethod
        def getDefaultFlagsImpl(cls) -> typing.List[typing.Any]:
            return [
                    False, False, False,
                    False, 0.0, False, 0.0, False, False, False,
                    False, 0.0, False, 
                    np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
                ]

    def __init__(self, config: dict, contentRect: AbstractRectangle) -> None:
        self.rectangles: collections.OrderedDict[str, AbstractRectangle] = collections.OrderedDict()
        for k, v in config.items():
            self.rectangles[k] = RatioRectangle(contentRect, *v)

        self.rectangles["dialogNameRect"] = RatioRectangle(self.rectangles["dialogRect"], 0, 0.5, 0, 0.33)
        self.rectangles["dialogContentRect"] = RatioRectangle(self.rectangles["dialogRect"], 0, 1, 0.34, 1)
        self.rectangles["balloonRect"] = RatioRectangle(contentRect, 0.15, 0.85, 0.1, 0.75)
        self.rectangles["floatingBalloonRect"] = RatioRectangle(contentRect, 0, 1, 0, 1)

        self.dialogRect = self.rectangles["dialogRect"]
        self.blackscreenRect = self.rectangles["blackscreenRect"]
        self.dialogNameRect = self.rectangles["dialogNameRect"]
        self.dialogContentRect = self.rectangles["dialogContentRect"]
        self.balloonRect = self.rectangles["balloonRect"]
        self.floatingBalloonRect = self.rectangles["floatingBalloonRect"]

        self.cvPasses = [self.cvPassDialog, self.cvPassBalloon, self.cvPassBlackscreen]

        self.fpirPasses = collections.OrderedDict()

        self.fpirPasses["fpirPassRemoveNoiseBlackscreenFalse"] = FPIRPassBooleanRemoveNoise(MagirecoScene0Strategy.FlagIndex.Blackscreen, False, 3)
        self.fpirPasses["fpirPassRemoveNoiseBlackscreenTrue"] = FPIRPassBooleanRemoveNoise(MagirecoScene0Strategy.FlagIndex.Blackscreen, True, 10)
        
        self.fpirPasses["fpirPassDetectDialogContentJumpDown"] = FPIRPassDetectFeatureJump(
            featFlag=MagirecoScene0Strategy.FlagIndex.DialogContentVal,
            dstFlag=MagirecoScene0Strategy.FlagIndex.DialogContentValJumpDown, 
            featOpMean=lambda feats : np.mean(feats, 0),
            featOpDist=lambda lhs, rhs : lhs-rhs, # No abs(), only detects falling edge
            threshDist=2.0
        )
        self.fpirPasses["fpirPassDetectDialogContentJumpUp"] = FPIRPassDetectFeatureJump(
            featFlag=MagirecoScene0Strategy.FlagIndex.DialogContentVal,
            dstFlag=MagirecoScene0Strategy.FlagIndex.DialogContentValJumpUp, 
            featOpMean=lambda feats : np.mean(feats, 0),
            featOpDist=lambda lhs, rhs : -(lhs-rhs), # No abs(), only detects rising edge
            threshDist=2.0,
            windowSize=5
        )
        self.fpirPasses["fpirPassDetectDialogContentJumpUpJumpUp"] = FPIRPassDetectFeatureJump(
            featFlag=MagirecoScene0Strategy.FlagIndex.DialogContentValJumpUp,
            dstFlag=MagirecoScene0Strategy.FlagIndex.DialogContentValJumpUpJumpUp, 
            featOpMean=lambda feats : np.mean([float(x) for x in feats], 0),
            featOpDist=lambda lhs, rhs : -(lhs-rhs), # No abs(), only detects rising edge
            threshDist=0.5
        )
        self.fpirPasses["fpirPassDetectDialogNameJump"] = FPIRPassDetectFeatureJump(
            featFlag=MagirecoScene0Strategy.FlagIndex.DialogNameVal,
            dstFlag=MagirecoScene0Strategy.FlagIndex.DialogNameValJump, 
            featOpMean=lambda feats : np.mean(feats, 0),
            featOpDist=lambda lhs, rhs : abs(lhs-rhs),
            threshDist=1.0
        )
        def breakDialogJump(framePoint: FramePoint):
            framePoint.setFlag(MagirecoScene0Strategy.FlagIndex.Dialog,
                framePoint.getFlag(MagirecoScene0Strategy.FlagIndex.Dialog)
                and not framePoint.getFlag(MagirecoScene0Strategy.FlagIndex.DialogContentValJumpDown)
                and not framePoint.getFlag(MagirecoScene0Strategy.FlagIndex.DialogContentValJumpUpJumpUp)
                and not framePoint.getFlag(MagirecoScene0Strategy.FlagIndex.DialogNameValJump)
            )
        self.fpirPasses["fpirPassBreakDialogJump"] = FPIRPassFramewiseFunctional(
            func=breakDialogJump
        )
        self.fpirPasses["fpirPassRemoveNoiseDialogFalse"] = FPIRPassBooleanRemoveNoise(MagirecoScene0Strategy.FlagIndex.Dialog, False, 2)
        self.fpirPasses["fpirPassRemoveNoiseDialogTrue"] = FPIRPassBooleanRemoveNoise(MagirecoScene0Strategy.FlagIndex.Dialog, True, 10)
        
        self.fpirPasses["fpirPassDetectBalloonFeatureJump"] = FPIRPassDetectFeatureJump(
            featFlag=MagirecoScene0Strategy.FlagIndex.BalloonVal,
            dstFlag=MagirecoScene0Strategy.FlagIndex.BalloonValJump, 
            featOpMean=lambda feats : np.mean(feats, 0),
            featOpDist=lambda lhs, rhs : lhs-rhs, # No abs(), only detects falling edge
            threshDist=3.0
        )
        def breakBalloonJump(framePoint: FramePoint):
            framePoint.setFlag(MagirecoScene0Strategy.FlagIndex.Balloon,
                framePoint.getFlag(MagirecoScene0Strategy.FlagIndex.Balloon)
                and not framePoint.getFlag(MagirecoScene0Strategy.FlagIndex.BalloonValJump)
            )
        self.fpirPasses["fpirPassBreakBalloonJump"] = FPIRPassFramewiseFunctional(
            func=breakBalloonJump
        )
        self.fpirPasses["fpirPassRemoveNoiseBalloonFalse"] = FPIRPassBooleanRemoveNoise(MagirecoScene0Strategy.FlagIndex.Balloon, False, 2)
        self.fpirPasses["fpirPassRemoveNoiseBalloonTrue"] = FPIRPassBooleanRemoveNoise(MagirecoScene0Strategy.FlagIndex.Balloon, True, 10)

        self.fpirToIirPasses = collections.OrderedDict()
        self.fpirToIirPasses["fpirPassBuildIntervals"] = FPIRPassBooleanBuildIntervals(
            MagirecoScene0Strategy.FlagIndex.Blackscreen, 
            MagirecoScene0Strategy.FlagIndex.Dialog, 
            MagirecoScene0Strategy.FlagIndex.Balloon, 
        )

        self.iirPasses = collections.OrderedDict()
        self.iirPasses["iirPassFillGapBlackscreen"] = IIRPassFillGap(MagirecoScene0Strategy.FlagIndex.Blackscreen, 1200)
        self.iirPasses["iirPassFillGapDialog"] = IIRPassFillGap(MagirecoScene0Strategy.FlagIndex.Dialog, 500, meetPoint=1)
        self.iirPasses["iirPassFillGapBalloon"] = IIRPassFillGap(MagirecoScene0Strategy.FlagIndex.Balloon, 500, meetPoint=1)
        
        colourSpace: typing.Dict[str, typing.Tuple[float, float]] = {
            "Shiro":  ( 0.00,  0.00),
            "Mabayu": ( 0.13,  0.55),
            "Homura": (-0.05, -0.17),
            "Madoka": ( 0.27, -0.17),
            "Sayaka": (-0.25, -0.05),
            "Kyoko":  ( 0.40, -0.02),
            "Mami":   ( 0.20,  0.20),
            "Nagisa": ( 0.05, -0.17),
        }
        def classifySpeaker(interval: Interval):
            if not interval.mainFlag == MagirecoScene0Strategy.FlagIndex.Dialog and not interval.mainFlag == MagirecoScene0Strategy.FlagIndex.Balloon:
                return
            meanTextColour = np.mean([interval.framePoints[i].flags[MagirecoScene0Strategy.FlagIndex.MeanTextColour] for i in range(len(interval.framePoints))], 0)
            meanTextColourHSV = cv.cvtColor(np.array([[meanTextColour]], dtype=np.float32), cv.COLOR_BGR2HSV)[0][0]
            
            h = meanTextColourHSV[0]
            s = meanTextColourHSV[1]

            x = s * np.cos(np.deg2rad(h))
            y = s * np.sin(np.deg2rad(h))

            minDist = 1000
            for style, point in colourSpace.items():
                x0 = point[0]
                y0 = point[1]
                dist = (x - x0) * (x - x0) + (y - y0) * (y - y0)
                if dist < minDist:
                    interval.style = style
                    minDist = dist
            if minDist > 0.20:
                interval.style = "Shiranai"
            
            print(formatTimestamp(interval.begin), h, s, x, y, interval.style, minDist)

        self.iirPasses["iirPassClassifySpeaker"] = IIRPassIntervalwiseFunctional(classifySpeaker)

    @classmethod
    def getFlagIndexType(cls) -> typing.Type[AbstractFlagIndex]:
        return cls.FlagIndex

    def getRectangles(self) -> collections.OrderedDict[str, AbstractRectangle]:
        return self.rectangles

    def getCvPasses(self) -> typing.List[typing.Callable[[cv.UMat, FramePoint], bool]]:
        return self.cvPasses

    def getFpirPasses(self) -> collections.OrderedDict[str, FPIRPass]:
        return self.fpirPasses

    def getFpirToIirPasses(self) -> collections.OrderedDict[str, FPIRPassBuildIntervals]:
        return self.fpirToIirPasses

    def getIirPasses(self) -> collections.OrderedDict[str, IIRPass]:
        return self.iirPasses
    
    def getStyles(self) -> typing.List[str]:
        return [
            "Style: Shiro,Microsoft YaHei,40,&H00FFFFFF,&H000000FF,&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,2,2,2,10,10,100,1\n",
            "Style: Shiranai,Microsoft YaHei,40,&H00000000,&H000000FF,&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,2,2,2,10,10,100,1\n",
            "Style: Mabayu,Microsoft YaHei,40,&H0038FCCC,&H000000FF,&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,2,2,2,10,10,100,1\n",
            "Style: Homura,Microsoft YaHei,40,&H00FEC5C4,&H000000FF,&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,2,2,2,10,10,100,1\n",
            "Style: Madoka,Microsoft YaHei,40,&H00C699F6,&H000000FF,&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,2,2,2,10,10,100,1\n",
            "Style: Sayaka,Microsoft YaHei,40,&H00F3DD96,&H000000FF,&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,2,2,2,10,10,100,1\n",
            "Style: Kyoko,Microsoft YaHei,40,&H00738EFC,&H000000FF,&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,2,2,2,10,10,100,1\n",
            "Style: Mami,Microsoft YaHei,40,&H0083E1FE,&H000000FF,&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,2,2,2,10,10,100,1\n",
            "Style: Nagisa,Microsoft YaHei,40,&H00DCAFE3,&H000000FF,&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,2,2,2,10,10,100,1\n",
        ]
    
    def cvPassBlackscreen(self, frame: cv.UMat, framePoint: FramePoint) -> bool:
        roiBlackscreen = self.blackscreenRect.cutRoi(frame)
        roiBlackscreenGray = cv.cvtColor(roiBlackscreen, cv.COLOR_BGR2GRAY)
        _, roiBlackscreenBgBin = cv.threshold(roiBlackscreenGray, 80, 255, cv.THRESH_BINARY)
        _, roiBlackscreenTextBin = cv.threshold(roiBlackscreenGray, 160, 255, cv.THRESH_BINARY)
        meanBlackscreenBgBin: float = cv.mean(roiBlackscreenBgBin)[0]
        meanBlackscreenTextBin: float = cv.mean(roiBlackscreenTextBin)[0]
        hasBlackscreenBg: bool = meanBlackscreenBgBin < 20
        hasBlackscreenText: bool = meanBlackscreenTextBin > 0.07 and meanBlackscreenTextBin < 16

        isValidBlackscreen = hasBlackscreenBg and hasBlackscreenText

        framePoint.setFlag(MagirecoScene0Strategy.FlagIndex.Blackscreen, isValidBlackscreen)
        framePoint.setFlag(MagirecoScene0Strategy.FlagIndex.BlackscreenBg, hasBlackscreenBg)
        framePoint.setFlag(MagirecoScene0Strategy.FlagIndex.BlackscreenText, hasBlackscreenText)
        return isValidBlackscreen

    def cvPassDialog(self, frame: cv.UMat, framePoint: FramePoint) -> bool:
        roiDialog = self.dialogRect.cutRoi(frame)
        roiDialogGray = cv.cvtColor(roiDialog, cv.COLOR_BGR2GRAY)
        # roiDialogHSV = cv.cvtColor(roiDialog, cv.COLOR_BGR2HSV)

        _, roiDialogShade1BinFix = cv.threshold(roiDialogGray, 50, 255, cv.THRESH_BINARY)
        roiDialogShade1BinAdap = cv.adaptiveThreshold(roiDialogGray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 7, 3)
        roiDialogShade1Bin = cv.bitwise_and(roiDialogShade1BinFix, roiDialogShade1BinAdap)
        
        cc1Num, cc1Labels, cc1Stats, cc1Centroids = cv.connectedComponentsWithStatsWithAlgorithm(roiDialogShade1Bin, connectivity=4, ltype=cv.CV_16U, ccltype=cv.CCL_SAUF)
        cc1Stats: cv.UMat = cc1Stats.get()
        cc1Labels: cv.UMat = cc1Labels.get()

        cc1AcceptedLabels = []
        for n in range(cc1Num):
            stat = cc1Stats[n]
            if stat[4] > 20 and stat[4] < 300 and stat[2] < 25 and stat[3] < 25 and (stat[2] > 3 and stat[4] > 3):
                cc1AcceptedLabels.append(n)

        roiDialogText1Bin = np.isin(cc1Labels, cc1AcceptedLabels) * np.uint8(255)
        roiDialogText1Bin = cv.UMat(roiDialogText1Bin)
        roiDialogText1BinDialate = cv.morphologyEx(roiDialogText1Bin, cv.MORPH_DILATE, kernel=cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5)))

        _, roiDialogShade2BinFix = cv.threshold(roiDialogGray, 40, 255, cv.THRESH_BINARY_INV)
        roiDialogShade2BinAdap = cv.adaptiveThreshold(roiDialogGray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 7, 6)
        roiDialogShade2Bin = cv.bitwise_and(roiDialogShade2BinFix, roiDialogShade2BinAdap)
        roiDialogShade2BinFiltered = cv.bitwise_and(roiDialogShade2Bin, roiDialogText1BinDialate)

        roiDialogShade2BinFilteredClose = cv.morphologyEx(roiDialogShade2BinFiltered, cv.MORPH_CLOSE, kernel=cv.getStructuringElement(cv.MORPH_ELLIPSE, (11, 11)))
        
        roiDialogText2Bin = cv.bitwise_and(roiDialogShade2BinFilteredClose, roiDialogText1Bin)

        meanTextColour = cv.mean(roiDialog, mask=roiDialogText2Bin)

        roiDialogNameText2Bin = self.dialogNameRect.cutRoi(roiDialogText2Bin, self.dialogRect)
        roiDialogContentText2Bin = self.dialogContentRect.cutRoi(roiDialogText2Bin, self.dialogRect)

        cc2NameNum, cc2NameLabels, cc2NameStats, cc2NameCentroids = cv.connectedComponentsWithStatsWithAlgorithm(roiDialogNameText2Bin, connectivity=4, ltype=cv.CV_16U, ccltype=cv.CCL_SAUF)
        cc2NameStats: cv.UMat = cc2NameStats.get()

        cc2NameLeagalAreaSum: int = 0
        for n in range(cc2NameNum):
            stat = cc2NameStats[n]
            if (stat[4] > 20 and stat[4] < 300 and stat[2] < 25 and stat[3] < 25 and (stat[2] > 3 and stat[4] > 3)):
                cc2NameLeagalAreaSum += stat[4]
        cc2NameLeagalAreaRatio: float = cc2NameLeagalAreaSum / self.dialogRect.getArea() * 256

        cc2ContentNum, cc2ContentLabels, cc2ContentStats, cc2ContentCentroids = cv.connectedComponentsWithStatsWithAlgorithm(roiDialogContentText2Bin, connectivity=4, ltype=cv.CV_16U, ccltype=cv.CCL_SAUF)
        cc2ContentStats: cv.UMat = cc2ContentStats.get()

        roiDialogContentText3Bin = roiDialogContentText2Bin
        cc2ContentLeagalAreaSum: int = 0
        for n in range(cc2ContentNum):
            stat = cc2ContentStats[n]
            if (stat[4] > 20 and stat[4] < 300 and stat[2] < 25 and stat[3] < 25 and (stat[2] > 3 and stat[4] > 3)):
                cc2ContentLeagalAreaSum += stat[4]
            # if not (stat[4] > 20 and stat[4] < 300 and stat[2] < 25 and stat[3] < 25 and (stat[2] > 3 and stat[4] > 3)):
            #     roiDialogText3Bin[cc2ContentLabels == n] = 0
        cc2ContentLeagalAreaRatio: float = cc2ContentLeagalAreaSum / self.dialogRect.getArea() * 256

        hasDialog: bool = cc2NameLeagalAreaRatio > 1.0 or cc2ContentLeagalAreaRatio > 3.0
        hasDialogStrict: bool = cc2NameLeagalAreaRatio > 1.0 and cc2ContentLeagalAreaRatio > 5.0

        if hasDialog:
            framePoint.setFlag(MagirecoScene0Strategy.FlagIndex.MeanTextColour, meanTextColour)

        # framePoint.setDebugFlag(cc2NameLeagalAreaRatio, cc2ContentLeagalAreaRatio)
        # framePoint.setDebugFlag(meanDialogText1BinOpen, num, stats)

        framePoint.setFlag(MagirecoScene0Strategy.FlagIndex.Dialog, hasDialog)
        framePoint.setFlag(MagirecoScene0Strategy.FlagIndex.DialogNameVal, cc2NameLeagalAreaRatio)
        framePoint.setFlag(MagirecoScene0Strategy.FlagIndex.DialogContentVal, cc2ContentLeagalAreaRatio)
        return hasDialogStrict

    def cvPassBalloon(self, frame: cv.UMat, framePoint: FramePoint) -> bool:
        roiBolloon = self.balloonRect.cutRoi(frame)
        roiBolloonGray = cv.cvtColor(roiBolloon, cv.COLOR_BGR2GRAY)

        _, roiBolloonShade1BinFix = cv.threshold(roiBolloonGray, 40, 255, cv.THRESH_BINARY_INV)
        roiBolloonShade1BinAdap = cv.adaptiveThreshold(roiBolloonGray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 13, 33)
        roiBalloonShade1Bin = cv.bitwise_and(roiBolloonShade1BinFix, roiBolloonShade1BinAdap)
        
        roiBalloonShade1BinClose = cv.morphologyEx(roiBalloonShade1Bin, cv.MORPH_CLOSE, kernel=cv.getStructuringElement(cv.MORPH_ELLIPSE, (11, 11)))

        _, roiBolloonText1Bin = cv.threshold(roiBolloonGray, 100, 255, cv.THRESH_BINARY)
        roiBolloonText2Bin = cv.bitwise_and(roiBolloonText1Bin, roiBalloonShade1BinClose)

        roiBolloonText2BinOpen = cv.morphologyEx(roiBolloonText2Bin, cv.MORPH_OPEN, kernel=cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3)))

        # roiBolloonText1BinDialate = cv.morphologyEx(roiBolloonText1Bin, cv.MORPH_DILATE, kernel=cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3)))
        # roiBalloonShade2Bin = cv.bitwise_and(roiBalloonShade1Bin, roiBolloonText1BinDialate)

        roiBalloonBlur = cv.blur(roiBolloonText2BinOpen, (301, 101))

        _, maxBalloonBlur, _, maxBalloonBlurPoint = cv.minMaxLoc(roiBalloonBlur)

        if maxBalloonBlur < 3:
            framePoint.setFlag(MagirecoScene0Strategy.FlagIndex.Balloon, False)
            framePoint.setFlag(MagirecoScene0Strategy.FlagIndex.BalloonVal, 0)
            return False

        floatBalloonCentralY = maxBalloonBlurPoint[0]
        floatBalloonCentralX = maxBalloonBlurPoint[1]
        balloonRectWidth, balloonRectHeight = self.balloonRect.getSizeInt()
        floatBalloonCentralYRatio = floatBalloonCentralY / balloonRectWidth
        floatBalloonCentralXRatio = floatBalloonCentralX / balloonRectHeight
        floatBalloonLeftRatio = max(0, floatBalloonCentralYRatio - 0.2)
        floatBalloonRightRatio = min(1, floatBalloonCentralYRatio + 0.2)
        floatBalloonTopRatio = max(0, floatBalloonCentralXRatio - 0.15)
        floatBalloonBottomRatio = min(1, floatBalloonCentralXRatio + 0.15)
        
        # Guarantee constant shape in corners
        if floatBalloonLeftRatio == 0:
            floatBalloonRightRatio = 0.4
        if floatBalloonRightRatio == 1:
            floatBalloonLeftRatio = 0.6
        if floatBalloonTopRatio == 0:
            floatBalloonBottomRatio = 0.3
        if floatBalloonBottomRatio == 1:
            floatBalloonTopRatio = 0.7

        self.rectangles["floatingBalloonRect"] = RatioRectangle(self.balloonRect, floatBalloonLeftRatio, floatBalloonRightRatio, floatBalloonTopRatio, floatBalloonBottomRatio)
        self.floatingBalloonRect = self.rectangles["floatingBalloonRect"]

        roiFB = self.floatingBalloonRect.cutRoi(frame)
        roiFBGray = cv.cvtColor(roiFB, cv.COLOR_BGR2GRAY)

        _, roiFBShade1BinFix = cv.threshold(roiFBGray, 50, 255, cv.THRESH_BINARY)
        roiFBShade1BinAdap = cv.adaptiveThreshold(roiFBGray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 7, 3)
        roiFBShade1Bin = cv.bitwise_and(roiFBShade1BinFix, roiFBShade1BinAdap)
        
        cc1Num, cc1Labels, cc1Stats, cc1Centroids = cv.connectedComponentsWithStatsWithAlgorithm(roiFBShade1Bin, connectivity=4, ltype=cv.CV_16U, ccltype=cv.CCL_SAUF)
        cc1Labels: cv.UMat = cc1Labels.get()
        cc1Stats: cv.UMat = cc1Stats.get()

        cc1AcceptedLabels = []
        for n in range(cc1Num):
            stat = cc1Stats[n]
            if stat[4] > 20 and stat[4] < 300 and stat[2] < 25 and stat[3] < 25 and (stat[2] > 3 and stat[4] > 3):
                cc1AcceptedLabels.append(n)

        roiFBText1Bin = np.isin(cc1Labels, cc1AcceptedLabels) * np.uint8(255)
        roiFBText1Bin = cv.UMat(roiFBText1Bin)

        roiFBText1BinDialate = cv.morphologyEx(roiFBText1Bin, cv.MORPH_DILATE, kernel=cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5)))

        _, roiFBShade2BinFix = cv.threshold(roiFBGray, 40, 255, cv.THRESH_BINARY_INV)
        roiFBShade2BinAdap = cv.adaptiveThreshold(roiFBGray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 7, 6)
        roiFBShade2Bin = cv.bitwise_and(roiFBShade2BinFix, roiFBShade2BinAdap)
        roiFBShade2BinFiltered = cv.bitwise_and(roiFBShade2Bin, roiFBText1BinDialate)

        roiFBShade2BinFilteredClose = cv.morphologyEx(roiFBShade2BinFiltered, cv.MORPH_CLOSE, kernel=cv.getStructuringElement(cv.MORPH_ELLIPSE, (11, 11)))
        
        roiFBText2Bin = cv.bitwise_and(roiFBShade2BinFilteredClose, roiFBText1Bin)

        meanTextColour= cv.mean(roiFB, mask=roiFBText2Bin)

        cc2Num, cc2Labels, cc2Stats, cc2Centroids = cv.connectedComponentsWithStatsWithAlgorithm(roiFBText2Bin, connectivity=4, ltype=cv.CV_16U, ccltype=cv.CCL_SAUF)
        cc2Stats: cv.UMat = cc2Stats.get()
        
        cc2LeagalAreaSum: int = 0
        for n in range(cc2Num):
            stat = cc2Stats[n]
            if (stat[4] > 20 and stat[4] < 300 and stat[2] < 25 and stat[3] < 25 and (stat[2] > 3 and stat[4] > 3)):
                cc2LeagalAreaSum += stat[4]
        cc2LeagalAreaRatio: float = cc2LeagalAreaSum / self.floatingBalloonRect.getArea() * 256

        hasFloatingBalloon = cc2LeagalAreaRatio > 2.5

        if hasFloatingBalloon:
            framePoint.setFlag(MagirecoScene0Strategy.FlagIndex.MeanTextColour, meanTextColour)

        framePoint.setFlag(MagirecoScene0Strategy.FlagIndex.Balloon, hasFloatingBalloon)
        framePoint.setFlag(MagirecoScene0Strategy.FlagIndex.BalloonVal, cc2LeagalAreaRatio)

        return hasFloatingBalloon