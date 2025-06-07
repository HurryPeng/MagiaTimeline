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
        
    @staticmethod
    def genOcrEngine() -> paddleocr.PaddleOCR:
        return paddleocr.PaddleOCR(
            det=True, rec=False, cls=False, use_angle_cls=False, show_log=False,
            det_model_dir="./PaddleOCRModels/ch_PP-OCRv4_det_infer/",
            rec_model_dir="./PaddleOCRModels/ch_PP-OCRv4_rec_infer/",
            cls_model_dir="./PaddleOCRModels/ch_ppocr_mobile_v2.0_cls_infer/"
        )

    def __init__(self, config: dict, contentRect: AbstractRectangle) -> None:
        AbstractStrategy.__init__(self, contentRect)
        AbstractSpeculativeStrategy.__init__(self)

        self.ocr = DiffTextDetectionStrategy.genOcrEngine()

        self.rectangles: collections.OrderedDict[str, AbstractRectangle] = collections.OrderedDict()
        self.rectangles["dialogRect"] = RatioRectangle(contentRect, *config["dialogRect"])

        self.featureThreshold: float = config["featureThreshold"]
        self.boxVerticalExpansion: float = config["boxVerticalExpansion"]
        self.nonMajorBoxSuppressionMaxRatio: float = config["nonMajorBoxSuppressionMaxRatio"]
        self.nonMajorBoxSuppressionMinRank: int = config["nonMajorBoxSuppressionMinRank"]
        self.colourTolerance: int = config["colourTolerance"]
        self.minMaskIou: float = 0.5
        self.minOcrIou: float = 0.2
        self.iirPassDenoiseMinTime: int = config["iirPassDenoiseMinTime"]
        self.debugLevel: int = config["debugLevel"]

        self.dialogRect = self.rectangles["dialogRect"]

        self.cvPasses = [self.cvPassDialog]

        self.fpirPasses = collections.OrderedDict()

        self.fpirToIirPasses = collections.OrderedDict()
        self.fpirToIirPasses["fpirPassBuildIntervals"] = FPIRPassBooleanBuildIntervals(
            DiffTextDetectionStrategy.FlagIndex.Dialog
        )

        self.iirPasses = collections.OrderedDict()
        
        self.specIirPasses = collections.OrderedDict()
        self.specIirPasses["iirPassMerge"] = IIRPassMerge(
            lambda iir, interval0, interval1:
                self.decideFeatureMerge(
                    [interval0.getFlag(self.getFeatureFlagIndex())],
                    [interval1.getFlag(self.getFeatureFlagIndex())]
                ) and iir.ms2Timestamp(self.iirPassDenoiseMinTime) > interval0.dist(interval1)
        )
        self.specIirPasses["iirPassDenoise"] = IIRPassDenoise(DiffTextDetectionStrategy.FlagIndex.Dialog, self.iirPassDenoiseMinTime)
        self.specIirPasses["iirPassMerge2"] = self.specIirPasses["iirPassMerge"]

        self.statDecideFeatureMerge = 0
        self.statDecideFeatureMergeDiff = 0
        self.statDecideFeatureMergeComputeECC = 0
        self.statDecideFeatureMergeFindTransformECC = 0
        self.statDecideFeatureMergeInpaint = 0
        self.statDecideFeatureMergeOCR = 0
        # self.log = open("log.csv", "w")

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
        self.statDecideFeatureMerge += 1

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
        intersectArea = np.sum(intersectMask) / 255
        unionArea = np.sum(unionMask) / 255
        
        if unionArea == 0:
            return False
        
        maskIou = intersectArea / unionArea
        if self.debugLevel >= 1:
            print("intersectArea:", intersectArea)
            print("unionArea:", unionArea)
            print("iou:", maskIou)

        if self.debugLevel >= 1:
            if maskIou < self.minMaskIou:
                print("NO")
            else:
                print("WAIT")
        
        if maskIou < self.minMaskIou:
            return False
        
        self.statDecideFeatureMergeDiff += 1
        
        diffMask: cv.Mat = rgbDiffMask(oldImage, newImage, self.colourTolerance)
        # The rate of pixels that are close enough
        diffArea = np.sum(diffMask) / 255
        diffRate = diffArea / unionArea
        if self.debugLevel >= 1:
            print("diffRate:", diffRate)
        if diffRate < 0.1:
            return True
        
        # ECC
        self.statDecideFeatureMergeComputeECC += 1

        oldImageGrey = cv.cvtColor(oldImage, cv.COLOR_BGR2GRAY)
        newImageGrey = cv.cvtColor(newImage, cv.COLOR_BGR2GRAY)

        warp = np.eye(2, 3, dtype=np.float32)
        ccInit: float = cv.computeECC(
            templateImage=newImageGrey,
            inputImage=oldImageGrey,
            inputMask=unionMask,
        )
        cc = ccInit

        oldImageGreyMasked = cv.bitwise_and(oldImageGrey, oldImageGrey, mask=oldMask)
        newImageGreyMasked = cv.bitwise_and(newImageGrey, newImageGrey, mask=newMask)
        oldImageGreyMaskedF32 = np.float32(oldImageGreyMasked)
        newImageGreyMaskedF32 = np.float32(newImageGreyMasked)
        (shiftX, shiftY), response = phaseCorrelateMaxRes(
            src1=newImageGreyMaskedF32,
            src2=oldImageGreyMaskedF32,
            maxResolution=1800
        )
        if response > 0.1:
            warp = np.array([[1, 0, shiftX], [0, 1, shiftY]], dtype=np.float32)
        pcWarpDist = np.linalg.norm(warp[0:2, 2])
        if pcWarpDist > 0.5 and pcWarpDist < 50:
            try:
                self.statDecideFeatureMergeFindTransformECC += 1
                cc, warp = cv.findTransformECC(
                    templateImage=newImageGrey,
                    inputImage=oldImageGrey,
                    warpMatrix=warp,
                    motionType=cv.MOTION_TRANSLATION,
                    criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.01),
                    inputMask=unionMask,
                )
            except cv.error:
                cc = 0
                warp = np.eye(2, 3, dtype=np.float32)

        warpedImage = newImage
        warpDist = np.linalg.norm(warp[0:2, 2])
        if warpDist > 0.5 and warpDist < 50 and cc > ccInit:
            warpedImage = cv.warpAffine(newImage, warp, (newImage.shape[1], newImage.shape[0]), flags=cv.INTER_LINEAR)

        if self.debugLevel >= 1:
            print("cc:", cc)
            print("warp:", warp)
            print("warpDist:", warpDist)

        if self.debugLevel >= 2:
            combinedImage = cv.addWeighted(oldImage, 0.5, warpedImage, 0.5, 0)
            combinedImage = cv.bitwise_and(combinedImage, combinedImage, mask=unionMask)
            cv.imshow("DebugFrame", combinedImage)
            cv.waitKey(0)
        
        # Sobel Iou Filtering

        oldImageSobel = rgbSobel(oldImage, 1)
        warpedImageSobel = rgbSobel(warpedImage, 1)
        oldImageSobelBin = cv.threshold(oldImageSobel, 32, 255, cv.THRESH_BINARY)[1]
        warpedImageSobelBin = cv.threshold(warpedImageSobel, 32, 255, cv.THRESH_BINARY)[1]
        oldImageSobelBinDilate = cv.morphologyEx(oldImageSobelBin, cv.MORPH_DILATE, cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3)))
        warpedImageSobelBinDilate = cv.morphologyEx(warpedImageSobelBin, cv.MORPH_DILATE, cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3)))
        intersectSobelBin = cv.bitwise_and(oldImageSobelBinDilate, warpedImageSobelBinDilate)
        intersectSobelBinMasked = cv.bitwise_and(intersectSobelBin, unionMask)
        unionSobelBin = cv.bitwise_or(oldImageSobelBinDilate, warpedImageSobelBinDilate)
        unionSobelBinMasked = cv.bitwise_and(unionSobelBin, unionMask)
        sobelIou = np.sum(intersectSobelBinMasked) / np.sum(unionSobelBinMasked)

        # Inpainting

        self.statDecideFeatureMergeInpaint += 1

        diffMask = rgbDiffMask(oldImage, warpedImage, self.colourTolerance)

        if self.debugLevel >= 2:
            cv.imshow("DebugFrame", diffMask)
            cv.waitKey(0)

        inpaintMask = cv.bitwise_and(cv.bitwise_not(diffMask), unionMask)
        inpaintMaskGradient = cv.morphologyEx(inpaintMask, cv.MORPH_GRADIENT, cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3)))
        inpaintMaskGradientAndCommonSobel = cv.bitwise_and(inpaintMaskGradient, intersectSobelBinMasked)
        cv.copyTo(src=inpaintMaskGradientAndCommonSobel, dst=inpaintMask, mask=inpaintMaskGradient)

        warpedImageInpaint = cv.inpaint(warpedImage, inpaintMask, 1, cv.INPAINT_TELEA)

        # Post-inpaint Sobel Iou Filtering

        warpedImageInpaintSobel = rgbSobel(warpedImageInpaint, 1)
        warpedImageInpaintSobelBin = cv.threshold(warpedImageInpaintSobel, 32, 255, cv.THRESH_BINARY)[1]
        warpedImageInpaintSobelBinDilate = cv.morphologyEx(warpedImageInpaintSobelBin, cv.MORPH_DILATE, cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3)))
        warpedImageInpaintSobelBinDilateMasked = cv.bitwise_and(warpedImageInpaintSobelBinDilate, unionMask)
        intersectPostInpaintSobel = cv.bitwise_and(warpedImageInpaintSobelBinDilateMasked, intersectSobelBinMasked)
        unionPostInpaintSobel = cv.bitwise_or(warpedImageInpaintSobelBinDilateMasked, unionSobelBinMasked)
        postInpaintSobelIou = np.sum(intersectPostInpaintSobel) / np.sum(unionPostInpaintSobel)

        sobelIouDiff = sobelIou - postInpaintSobelIou
        if self.debugLevel >= 1:
            print("sobelIou:", sobelIou)
            print("postInpaintSobelIou:", postInpaintSobelIou)
            print("sobelIouDiff:", sobelIouDiff)

        if sobelIouDiff > 0.8:
            return True
        if sobelIouDiff < 0.2:
            return False

        if self.debugLevel >= 2:
            cv.imshow("DebugFrame", inpaintMask)
            cv.waitKey(0)
        if self.debugLevel >= 2:
            cv.imshow("DebugFrame", warpedImageInpaint)
            cv.waitKey(0)
        
        # OCR
        self.statDecideFeatureMergeOCR += 1

        ocrMask, _, _ = self.ocrPass(warpedImageInpaint)

        ocrIntersectMask = cv.bitwise_and(ocrMask, unionMask)
        ocrIntersectVal = np.mean(ocrIntersectMask)
        ocrIntersectArea = np.sum(ocrIntersectMask) / 255
        ocrIou = ocrIntersectArea / unionArea

        if self.debugLevel >= 1:
            print("ocrIntersectVal:", ocrIntersectVal)
            print("unionArea:", unionArea)
            print("ocrIou:", ocrIou)

        # After inpainting the common area, detecting no text means the original texts are the same
        if self.debugLevel >= 1:
            if ocrIntersectVal < self.featureThreshold or ocrIou < self.minOcrIou:
                print("YES")
            else:
                print("NOO")
        if self.debugLevel >= 2:
            debugFrame = cv.addWeighted(warpedImageInpaint, 0.5, cv.cvtColor(ocrIntersectMask, cv.COLOR_GRAY2BGR), 0.5, 0)
            cv.imshow("DebugFrame", debugFrame)
            cv.waitKey(0)

        # self.log.write(f"{sobelIou},{postInpaintSobelIou},{ocrIou},{1 if ocrIntersectVal < self.featureThreshold or ocrIou < self.minOcrIou else 0}\n")
        # self.log.flush()

        # if not (ocrIntersectVal < self.featureThreshold or ocrIou < self.minOcrIou):
        #     print(f"sobelIou: {sobelIou}, postInpaintSobelIou: {postInpaintSobelIou}, ocrIntersectVal: {ocrIntersectVal}, ocrIou: {ocrIou}")
        #     cv.imshow("DebugFrame", cv.addWeighted(oldImage, 0.5, warpedImage, 0.5, 0))
        #     cv.waitKey(0)
        #     cv.imshow("DebugFrame", oldImage)
        #     cv.waitKey(0)
        #     cv.imshow("DebugFrame", warpedImage)
        #     cv.waitKey(0)
        #     cv.imshow("DebugFrame", oldImageSobelBinDilate)
        #     cv.waitKey(0)
        #     cv.imshow("DebugFrame", warpedImageSobelBinDilate)
        #     cv.waitKey(0)
        #     cv.imshow("DebugFrame", intersectSobelBinMasked)
        #     cv.waitKey(0)
        #     cv.imshow("DebugFrame", inpaintMask)
        #     cv.waitKey(0)
        #     cv.imshow("DebugFrame", warpedImageInpaint)
        #     cv.waitKey(0)
        #     cv.imshow("DebugFrame", warpedImageInpaintSobelBinDilateMasked)
        #     cv.waitKey(0)
        #     cv.imshow("DebugFrame", cv.addWeighted(warpedImageInpaint, 0.5, cv.cvtColor(ocrIntersectMask, cv.COLOR_GRAY2BGR), 0.5, 0))
        #     cv.waitKey(0)
        #     cv.imshow("DebugFrame", intersectPostInpaintSobel)
        #     cv.waitKey(0)

        return ocrIntersectVal < self.featureThreshold or ocrIou < self.minOcrIou
    
    def aggregateFeatures(self, features: typing.List[typing.Any]) -> typing.Any:
        # simply return the last feature
        return features[-1]

    def cutOcrFrame(self, frame: cv.Mat) -> cv.Mat:
        return self.dialogRect.cutRoi(frame)
    
    def cutCleanOcrFrame(self, frame: cv.Mat) -> cv.Mat:
        return self.dialogRect.cutRoi(frame)
    
    def detectTextBoxes(self, frame: cv.Mat, maxResolution: int = 1800) -> typing.List[typing.Tuple[int, int, int, int]]:
        imgH, imgW = frame.shape[:2]
        frameScaled, scale = maxResScaleDown(frame, maxResolution)

        result = self.ocr.ocr(frameScaled, det=True, cls=False, rec=False)

        if result[0] is None:
            return []

        boxes = []
        for wordInfo in result[0]:
            wordInfo = np.array(wordInfo, np.int32)
            wordInfo *= scale
            x0, y0 = wordInfo[0]
            x1, y1 = wordInfo[1]
            x2, y2 = wordInfo[2]
            x3, y3 = wordInfo[3]
            angle0 = np.arctan2(y1 - y0, x1 - x0)
            angle3 = np.arctan2(y2 - y3, x2 - x3)
            angle = (angle0 + angle3) / 2
            if np.abs(angle) > np.pi / 180 * 3:
                continue

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

    def ocrPass(self, frame: cv.Mat) -> typing.Tuple[cv.Mat, float, cv.Mat | None]:
        # returns mask, dialogVal, debugFrame

        boxes = self.detectTextBoxes(frame)

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
