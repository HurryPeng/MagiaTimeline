import typing
import enum
import collections
import paddleocr

from Util import *
from Strategies.AbstractStrategy import *
from AbstractFlagIndex import *
from Rectangle import *
from IR import *

class DiffTextDetectionStrategy(AbstractFramewiseStrategy, AbstractSpeculativeStrategy, AbstractOcrStrategy):
    class FlagIndex(AbstractFlagIndex):
        Dialog = enum.auto()
        DialogVal = enum.auto()
        DialogFeat = enum.auto() # (frame, mask)
        DialogFeatJump = enum.auto()
        OcrFrame = enum.auto()

        @classmethod
        def getDefaultFlagsImpl(cls) -> typing.List[typing.Any]:
            return [
                False,
                0.0,
                (None, None),
                False,
                None
            ]

    def __init__(self, config: dict, contentRect: AbstractRectangle) -> None:
        AbstractStrategy.__init__(self, contentRect)
        AbstractSpeculativeStrategy.__init__(self)

        self.ocr = paddleocr.PaddleOCR(
            det=True, rec=False, cls=False, use_angle_cls=False, show_log=False,
            det_model_dir="./PaddleOCRModels/ch_PP-OCRv4_det_infer/",
            rec_model_dir="./PaddleOCRModels/ch_PP-OCRv4_rec_infer/",
            cls_model_dir="./PaddleOCRModels/ch_ppocr_mobile_v2.0_cls_infer/"
        )

        self.rectangles: collections.OrderedDict[str, AbstractRectangle] = collections.OrderedDict()
        self.rectangles["dialogRect"] = RatioRectangle(contentRect, *config["dialogRect"])

        self.featureThreshold: float = config["featureThreshold"]
        self.boxVerticalExpansion: float = config["boxVerticalExpansion"]
        self.nonMajorBoxSuppressionMaxRatio: float = config["nonMajorBoxSuppressionMaxRatio"]
        self.nonMajorBoxSuppressionMinRank: int = config["nonMajorBoxSuppressionMinRank"]
        self.colourTolerance: int = config["colourTolerance"]
        self.minIou: float = 0.7
        self.iirPassDenoiseMinTime: int = config["iirPassDenoiseMinTime"]
        self.debugLevel: int = config["debugLevel"]

        self.dialogRect = self.rectangles["dialogRect"]

        self.cvPasses = [self.cvPassDialog]

        self.fpirPasses = collections.OrderedDict()

        # self.fpirPasses["fpirPassDetectDialogJump"] = FPIRPassDetectFeatureJump(
        #     featFlag=DiffOcrBooleanStrategy.FlagIndex.DialogFeat,
        #     dstFlag=DiffOcrBooleanStrategy.FlagIndex.DialogFeatJump, 
        #     featOpMean=lambda feats : np.mean(feats, axis=0),
        #     featOpDist=lambda lhs, rhs : np.linalg.norm(lhs - rhs),
        #     threshDist=0.1,
        #     windowSize=5,
        #     featOpStd=lambda feats: np.mean(np.std(feats, axis=0)),
        #     threshStd=0.005
        # )


        # def breakDialogJump(framePoint: FramePoint):
        #     framePoint.setFlag(DiffOcrBooleanStrategy.FlagIndex.Dialog,
        #         framePoint.getFlag(DiffOcrBooleanStrategy.FlagIndex.Dialog)
        #         and not framePoint.getFlag(DiffOcrBooleanStrategy.FlagIndex.DialogFeatJump)
        #     )
        # self.fpirPasses["fpirPassBreakDialogJump"] = FPIRPassFramewiseFunctional(
        #     func=breakDialogJump
        # )
        
        self.fpirToIirPasses = collections.OrderedDict()
        self.fpirToIirPasses["fpirPassBuildIntervals"] = FPIRPassBooleanBuildIntervals(
            DiffTextDetectionStrategy.FlagIndex.Dialog
        )

        self.iirPasses = collections.OrderedDict()
        # self.iirPasses["iirPassFillGapDialog"] = IIRPassFillGap(DiffOcrBooleanStrategy.FlagIndex.Dialog, self.iirPassDenoiseMinTime, meetPoint=1.0)
        
        self.specIirPasses = collections.OrderedDict()
        # self.specIirPasses["iirPassMerge"] = IIRPassMerge(
        #     lambda interval0, interval1:
        #         self.decideFeatureMerge(
        #             interval0.getFlag(self.getFeatureFlagIndex()),
        #             interval1.getFlag(self.getFeatureFlagIndex())
        #         )
        # )
        self.specIirPasses["iirPassDenoise"] = IIRPassDenoise(DiffTextDetectionStrategy.FlagIndex.Dialog, self.iirPassDenoiseMinTime)
        # self.specIirPasses["iirPassMerge2"] = self.specIirPasses["iirPassMerge"]

    @classmethod
    def getFlagIndexType(cls) -> typing.Type[AbstractFlagIndex]:
        return cls.FlagIndex
    
    @classmethod
    def getMainFlagIndex(cls) -> AbstractFlagIndex:
        return cls.FlagIndex.Dialog

    @classmethod
    def getFeatureFlagIndex(cls) -> AbstractFlagIndex:
        return cls.FlagIndex.DialogFeat
    
    @classmethod
    def getOcrFrameFlagIndex(cls) -> AbstractFlagIndex:
        return cls.FlagIndex.OcrFrame
    
    @classmethod
    def isEmptyFeature(cls, feature: typing.Tuple[cv.Mat, cv.Mat]) -> bool:
        return feature[0] is None or feature[1] is None

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
    
    def getSpecIirPasses(self) -> collections.OrderedDict[str, IIRPass]:
        return self.specIirPasses
    
    def decideFeatureMerge(self, oldFeatures: typing.List[typing.Any], newFeatures: typing.List[typing.Any]) -> bool:
        oldFeature = oldFeatures[0]
        newFeature = newFeatures[0]

        if self.isEmptyFeature(oldFeature) and self.isEmptyFeature(newFeature):
            return True
        if self.isEmptyFeature(oldFeature) or self.isEmptyFeature(newFeature):
            return False

        oldImage, oldMask = oldFeature
        newImage, newMask = newFeature

        # Quick mask iou check before performing ocr on the intersection of the two images
        intersectMask = cv.bitwise_and(oldMask, newMask)
        unionMask = cv.bitwise_or(oldMask, newMask)
        oldArea = np.sum(oldMask) / 255
        newArea = np.sum(newMask) / 255
        intersectArea = np.sum(intersectMask) / 255
        unionArea = np.sum(unionMask) / 255

        # debugFrameNew = cv.addWeighted(newImage, 0.5, cv.cvtColor(newMask, cv.COLOR_GRAY2BGR), 0.5, 0)
        # debugFrameOld = cv.addWeighted(oldImage, 0.5, cv.cvtColor(oldMask, cv.COLOR_GRAY2BGR), 0.5, 0)
        # debugFrame = cv.addWeighted(debugFrameNew, 0.5, debugFrameOld, 0.5, 0)
        
        if unionArea == 0:
            return False
        
        iou = intersectArea / unionArea
        # print("intersectArea:", intersectArea)
        # print("unionArea:", unionArea)
        # print("iou:", iou)

        # if iou < self.minIou:
        #     print("NO")
        # else:
        #     print("WAIT")
        
        if iou < self.minIou:
            return False
        
        # cv.imshow("DebugFrame", debugFrame)
        # cv.waitKey(0)
        
        diffImage = cv.absdiff(oldImage, newImage)
        diffImageGrey = cv.cvtColor(diffImage, cv.COLOR_BGR2GRAY)
        _, diffMask = cv.threshold(diffImageGrey, self.colourTolerance, 255, cv.THRESH_BINARY)
        # The rate of pixels that are close enough
        diffRate = np.average(diffMask) / 255
        # print("diffRate:", diffRate)
        if diffRate < 0.001:
            # print("YES IDENTICAL")
            return True
        # cv.imshow("DebugFrame", diffMask)
        # cv.waitKey(0)
        inpaintMask = cv.bitwise_and(cv.bitwise_not(diffMask), unionMask)
        # cv.imshow("DebugFrame", inpaintMask)
        # cv.waitKey(0)
        biggerImage = oldImage if oldArea > newArea else newImage
        biggerImageInpaint = cv.inpaint(biggerImage, inpaintMask, 3, cv.INPAINT_TELEA)
        # noisifiedImage = cv.bitwise_and(noisifiedImage, noisifiedImage, mask=intersectMask)
        # cv.imshow("DebugFrame", newImageInpaint)
        # cv.waitKey(0)
        
        # Perform ocr on the noisified image and recompute iou
        ocrMask, _, _ = self.ocrPass(biggerImageInpaint)

        ocrIntersectMask = cv.bitwise_and(ocrMask, unionMask)
        ocrIntersectVal = np.mean(ocrIntersectMask)
        ocrIntersectArea = np.sum(ocrIntersectMask) / 255
        ocrIou = ocrIntersectArea / unionArea

        # print("ocrIntersectVal:", ocrIntersectVal)
        # print("unionArea:", unionArea)
        # print("ocrIou:", ocrIou)

        # # After inpainting the common area, detecting no text means the original texts are the same
        # if ocrIntersectVal < self.featureThreshold or ocrIou < 1 - self.minIou:
        #     print("YES")
        # else:
        #     print("NOO")

        # debugFrame = cv.addWeighted(newImageInpaint, 0.5, cv.cvtColor(ocrIntersectMask, cv.COLOR_GRAY2BGR), 0.5, 0)
        # cv.imshow("DebugFrame", debugFrame)
        # cv.waitKey(0)

        return ocrIntersectVal < self.featureThreshold or ocrIou < 1 - self.minIou

    def releaseFeaturesOnHook(self) -> bool:
        return True

    def cutOcrFrame(self, frame: cv.Mat) -> cv.Mat:
        return self.dialogRect.cutRoi(frame)
    
    def cutCleanOcrFrame(self, frame: cv.Mat) -> cv.Mat:
        return self.dialogRect.cutRoi(frame)
    
    def detectTextBoxes(self, frame: cv.Mat, doRec: bool) -> typing.List[typing.Tuple[int, int, int, int]]:
        result = self.ocr.ocr(frame, det=True, cls=False, rec=doRec)

        # print("ocr result:", result)

        imgH, imgW = frame.shape[:2]

        if result[0] is None:
            return []

        boxes = []
        for wordInfo in result[0]:
            if doRec:
                confidence = wordInfo[1][1]
                if confidence < 0.9:
                    continue
                wordInfo = wordInfo[0]
            x0, y0 = wordInfo[0]
            x1, y1 = wordInfo[1]
            x2, y2 = wordInfo[2]
            x3, y3 = wordInfo[3]
            angle0 = np.arctan2(y1 - y0, x1 - x0)
            angle3 = np.arctan2(y2 - y3, x2 - x3)
            angle = (angle0 + angle3) / 2
            if np.abs(angle) > np.pi / 180 * 3:
                continue

            wordInfo = np.array(wordInfo, np.int32).reshape((-1, 1, 2))
            x0, y0, w0, h0 = cv.boundingRect(wordInfo)
            expand = int(h0 * self.boxVerticalExpansion)
            x = max(0, x0 - expand)
            y = max(0, y0 - expand)
            w = min(imgW, w0 + 2 * expand)
            h = min(imgH, h0 + 2 * expand)
            wordInfo = (x, y, w, h)
            boxes.append(wordInfo)

        return boxes

    def cvPassDialog(self, frame: cv.Mat, framePoint: FramePoint) -> bool:

        image = self.cutOcrFrame(frame)

        mask, dialogVal, debugFrame = self.ocrPass(image)

        hasDialog = dialogVal > self.featureThreshold

        feat: typing.Tuple[cv.Mat, cv.Mat] = (None, None)
        if hasDialog:
            feat = (image, mask)

        framePoint.setFlag(DiffTextDetectionStrategy.FlagIndex.Dialog, hasDialog)
        framePoint.setFlag(DiffTextDetectionStrategy.FlagIndex.DialogVal, dialogVal)
        framePoint.setFlag(DiffTextDetectionStrategy.FlagIndex.DialogFeat, feat)

        return False

    def ocrPass(self, frame: cv.Mat, doRec: bool = False) -> typing.Tuple[cv.Mat, float, cv.Mat | None]:
        # returns mask, dialogVal, debugFrame

        boxes = self.detectTextBoxes(frame, doRec)

        mask: cv.Mat = np.zeros_like(frame[:, :, 0])
        boxSizeSum = 0
        for box in boxes:
            x, y, w, h = box
            boxSizeSum += w * h
        boxes = sorted(boxes, key=lambda box: box[2] * box[3], reverse=True)
        for rank, box in enumerate(boxes):
            x, y, w, h = box
            if w * h <= self.nonMajorBoxSuppressionMaxRatio * boxSizeSum and rank >= self.nonMajorBoxSuppressionMinRank:
                break
            mask[y:y+h, x:x+w] = 255

        dialogVal: float = np.mean(mask)

        return mask, dialogVal, mask
