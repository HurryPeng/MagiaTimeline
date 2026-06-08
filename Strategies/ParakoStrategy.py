import typing
import enum
import collections

from Util import *
from Strategies.AbstractStrategy import *
from AbstractFlagIndex import *
from Rectangle import *
from IR import *

class ParakoStrategy(AbstractFramewiseStrategy, AbstractSpeculativeStrategy):
    class FlagIndex(AbstractFlagIndex):
        Dialog = enum.auto()
        DialogFeat = enum.auto()
        DialogFeatJump = enum.auto()

        @classmethod
        def getDefaultFlagsImpl(cls) -> typing.List[typing.Any]:
            return [False, np.zeros(64), False]

    def __init__(self, config: dict, contentRect: AbstractRectangle) -> None:
        AbstractStrategy.__init__(self, contentRect)
        AbstractSpeculativeStrategy.__init__(self)
        self.rectangles: collections.OrderedDict[str, AbstractRectangle] = collections.OrderedDict()
        for k, v in config.items():
            self.rectangles[k] = RatioRectangle(contentRect, *v)

        self.dialogRect = self.rectangles["dialogRect"]

        self.cvPasses = [self.cvPassDialog]

        self.fpirPasses = collections.OrderedDict()

        self.fpirPasses["fpirPassDetectDialogJump"] = FPIRPassDetectFeatureJump(
            featFlag=ParakoStrategy.FlagIndex.DialogFeat,
            dstFlag=ParakoStrategy.FlagIndex.DialogFeatJump, 
            featOpMean=lambda feats : np.mean(feats, axis=0),
            featOpDist=lambda lhs, rhs : np.linalg.norm(lhs - rhs),
            threshDist=0.1,
            windowSize=5,
            featOpStd=lambda feats: np.mean(np.std(feats, axis=0)),
            threshStd=0.005
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
        self.iirPasses["iirPassFillGapDialog"] = IIRPassFillGap(ParakoStrategy.FlagIndex.Dialog.name, 300, meetPoint=1.0)

        self.specIirPasses = collections.OrderedDict()
        self.specIirPasses["iirPassMerge"] = IIRPassMerge(
            lambda iir, interval0, interval1:
                self.decideFeatureMerge(
                    [interval0.getAttachment(AbstractSpeculativeStrategy.AggregatedFeatureKey)],
                    [interval1.getAttachment(AbstractSpeculativeStrategy.AggregatedFeatureKey)]
                )
        )
        self.specIirPasses["iirPassDenoise"] = IIRPassDenoise(ParakoStrategy.FlagIndex.Dialog.name, 100)
        self.specIirPasses["iirPassMerge2"] = self.specIirPasses["iirPassMerge"]

    @classmethod
    def getFlagIndexType(cls) -> typing.Type[AbstractFlagIndex]:
        return cls.FlagIndex
    

    def decideFeatureMerge(self, oldFeatures: typing.List[typing.Any], newFeatures: typing.List[typing.Any]) -> bool:
        return bool(np.linalg.norm(np.mean(oldFeatures, axis=0) - np.mean(newFeatures, axis=0)) < 0.1)
    
    def aggregateFeatures(self, features: typing.List[np.ndarray]) -> np.ndarray:
        return np.mean(features, axis=0)

    def isFpNonEmpty(self, fp: FramePoint) -> bool:
        return fp.getFlag(ParakoStrategy.FlagIndex.Dialog)

    def getFpFeature(self, fp: FramePoint) -> np.ndarray:
        return fp.getFlag(ParakoStrategy.FlagIndex.DialogFeat)

    def freeFpFeature(self, fp: FramePoint) -> None:
        fp.setFlag(ParakoStrategy.FlagIndex.DialogFeat, None)

    def cvPassDialog(self, frame: cv.Mat, framePoint: FramePoint) -> bool:
        roiDialog = self.dialogRect.cutRoi(frame)
        roiDialogHSV = cv.cvtColor(roiDialog, cv.COLOR_BGR2HSV)

        # Select out the whitest part of the dialog
        roiDialogShade = cv.inRange(roiDialogHSV, (0, 0, 240), (180, 16, 255))

        meanStdDevDialogShade = cv.meanStdDev(roiDialogShade)
        meanDialogShade = meanStdDevDialogShade[0][0][0]

        roiDialogShadeResized = cv.resize(roiDialogShade, (150, 50))
        dctFeat = dctDescriptor(roiDialogShadeResized, 8, 8)

        framePoint.setFlag(ParakoStrategy.FlagIndex.Dialog, meanDialogShade > 1.0)
        framePoint.setFlag(ParakoStrategy.FlagIndex.DialogFeat, dctFeat)

        return False
