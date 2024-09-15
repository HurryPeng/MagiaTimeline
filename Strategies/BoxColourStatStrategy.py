import typing
import enum
import collections
import paddleocr
import scipy.cluster
import scipy.spatial
import sklearn
import scipy
import sklearn.preprocessing

from Util import *
from Strategies.AbstractStrategy import *
from AbstractFlagIndex import *
from Rectangle import *
from IR import *

class BoxColourStatStrategy(AbstractFramewiseStrategy, AbstractSpeculativeStrategy, AbstractOcrStrategy):
    class FlagIndex(AbstractFlagIndex):
        Dialog = enum.auto()
        DialogFeat = enum.auto()
        DialogFeatJump = enum.auto()
        OcrFrame = enum.auto()

        @classmethod
        def getDefaultFlagsImpl(cls) -> typing.List[typing.Any]:
            return [False, np.zeros(64), False, None]

    def __init__(self, config: dict, contentRect: AbstractRectangle) -> None:
        AbstractStrategy.__init__(self, contentRect)
        AbstractSpeculativeStrategy.__init__(self)

        self.ocr = paddleocr.PaddleOCR(use_angle_cls=False, det_algorithm="DB", show_log=False)

        self.rectangles: collections.OrderedDict[str, AbstractRectangle] = collections.OrderedDict()
        self.rectangles["dialogRect"] = RatioRectangle(contentRect, *config["dialogRect"])

        self.sobelThreshold: int = config["sobelThreshold"]
        self.featureThreshold: float = config["featureThreshold"]
        self.featureJumpThreshold: float = config["featureJumpThreshold"]
        self.featureJumpStddevThreshold: float = config["featureJumpStddevThreshold"]
        self.minCcAreaRatio: float = config["minCcAreaRatio"]
        self.maxCcAreaRatio: float = config["maxCcAreaRatio"]
        self.minCcFinalMean: float = config["minCcFinalMean"]
        self.maxCcStddev: float = config["maxCcStddev"]
        self.colourTolerance: int = config["colourTolerance"]
        self.clusterThreshold: float = config["clusterThreshold"]
        self.minColourAreaRatio: float = config["minColourAreaRatio"]
        self.maxGreyscalePenalty: float = config["maxGreyscalePenalty"]

        self.dialogRect = self.rectangles["dialogRect"]

        self.cvPasses = [self.cvPassDialog]

        self.fpirPasses = collections.OrderedDict()

        self.fpirPasses["fpirPassDetectDialogJump"] = FPIRPassDetectFeatureJump(
            featFlag=BoxColourStatStrategy.FlagIndex.DialogFeat,
            dstFlag=BoxColourStatStrategy.FlagIndex.DialogFeatJump, 
            featOpMean=lambda feats : np.mean(feats, axis=0),
            featOpDist=lambda lhs, rhs : np.linalg.norm(lhs - rhs),
            threshDist=0.1,
            windowSize=5,
            featOpStd=lambda feats: np.mean(np.std(feats, axis=0)),
            threshStd=0.005
        )


        def breakDialogJump(framePoint: FramePoint):
            framePoint.setFlag(BoxColourStatStrategy.FlagIndex.Dialog,
                framePoint.getFlag(BoxColourStatStrategy.FlagIndex.Dialog)
                and not framePoint.getFlag(BoxColourStatStrategy.FlagIndex.DialogFeatJump)
            )
        self.fpirPasses["fpirPassBreakDialogJump"] = FPIRPassFramewiseFunctional(
            func=breakDialogJump
        )
        
        self.fpirToIirPasses = collections.OrderedDict()
        self.fpirToIirPasses["fpirPassBuildIntervals"] = FPIRPassBooleanBuildIntervals(
            BoxColourStatStrategy.FlagIndex.Dialog
        )

        self.iirPasses = collections.OrderedDict()
        self.iirPasses["iirPassFillGapDialog"] = IIRPassFillGap(BoxColourStatStrategy.FlagIndex.Dialog, 300, meetPoint=1.0)
        
        self.specIirPasses = collections.OrderedDict()
        self.specIirPasses["iirPassMerge"] = IIRPassMerge(
            lambda interval0, interval1:
                self.decideFeatureMerge(
                    [framePoint.getFlag(self.getFeatureFlagIndex()) for framePoint in interval0.framePoints],
                    [framePoint.getFlag(self.getFeatureFlagIndex()) for framePoint in interval1.framePoints]
                )
        )
        self.specIirPasses["iirPassDenoise"] = IIRPassDenoise(BoxColourStatStrategy.FlagIndex.Dialog, 300)
        self.specIirPasses["iirPassMerge2"] = self.specIirPasses["iirPassMerge"]

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
    def isEmptyFeature(cls, feature: np.ndarray) -> bool:
        return np.all(feature == 0)

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
        return bool(np.linalg.norm(np.mean(oldFeatures, axis=0) - np.mean(newFeatures, axis=0)) < self.featureJumpThreshold)
    
    def cutOcrFrame(self, frame: cv.Mat) -> cv.Mat:
        return self.dialogRect.cutRoi(frame)
    
    def cutCleanOcrFrame(self, frame: cv.Mat) -> cv.Mat:
        return self.ocrPass(frame)[0]
    
    def detectTextBoxes(self, frame: cv.Mat) -> typing.List[typing.Tuple[int, int, int, int]]:
        result = self.ocr.ocr(frame, det=True, cls=False, rec=False)

        imgH, imgW = frame.shape[:2]

        if result[0] is None:
            return []

        boxes = []
        for wordInfo in result[0]:
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
            expand = int(h0 * 0.10)
            x = max(0, x0 - expand)
            y = max(0, y0 - expand)
            w = min(imgW, w0 + 2 * expand)
            h = min(imgH, h0 + 2 * expand)
            wordInfo = (x, y, w, h)
            boxes.append(wordInfo)

        return boxes
    
    def filterText(self, image: cv.Mat) -> cv.Mat:
        imageSobel = rgbSobel(image, 1)
        imageSobelBin = cv.threshold(imageSobel, 32, 255, cv.THRESH_BINARY_INV)[1]

        minCcAreaRatio = self.minCcAreaRatio
        maxCcAreaRatio = self.maxCcAreaRatio

        minCcFinalMean = self.minCcFinalMean
        maxCcStddev = self.maxCcStddev

        colourTolerance = self.colourTolerance

        area = image.shape[0] * image.shape[1]
        minCcArea = minCcAreaRatio * area
        if minCcArea < 10:
            minCcArea = 10
        maxCcArea = maxCcAreaRatio * area

        nLabels, labels, stats, centroids = cv.connectedComponentsWithStats(imageSobelBin, connectivity=4, ltype=cv.CV_32S)
        acceptedMask = np.zeros_like(imageSobelBin)
        acceptedIds = []
        ccMeans = []
        ccStds = []
        acceptedArea = 0
        for i in range(0, nLabels):
            if stats[i][cv.CC_STAT_AREA] >= minCcArea and stats[i][cv.CC_STAT_AREA] <= maxCcArea:
                mask = np.where(labels == i, 255, 0).astype(np.uint8)
                mean, std = cv.meanStdDev(image, mask=mask)
                std = np.mean(std)
                if std < maxCcStddev:
                    acceptedMask = np.where(labels == i, 255, acceptedMask)
                    acceptedIds.append(i)
                    ccMeans.append(mean)
                    ccStds.append(std)
                    acceptedArea += stats[i][cv.CC_STAT_AREA]

        if len(acceptedIds) <= 1:
            return np.zeros_like(image[:, :, 0])

        ccMeans = np.array(ccMeans).reshape(len(ccMeans), -1)
        ccAreas = np.array([stats[i][cv.CC_STAT_AREA] for i in acceptedIds]).reshape(-1, 1)
        scaler = sklearn.preprocessing.StandardScaler()
        ccMeansScaled = scaler.fit_transform(ccMeans)
        # weightedMeans = meansScaled * np.sqrt(areas)
        distMat = scipy.spatial.distance.pdist(ccMeansScaled, metric='euclidean')
        Z = scipy.cluster.hierarchy.linkage(distMat, method='weighted')

        clusterThreshold = self.clusterThreshold
        minColourAreaRatio = self.minColourAreaRatio
        maxGreyscalePenalty = self.maxGreyscalePenalty
        clusters = scipy.cluster.hierarchy.fcluster(Z, clusterThreshold, criterion="distance")

        clusterColours = {}
        clusterAreas = {}

        for i, clusterId in enumerate(clusters):
            if clusterId not in clusterColours:
                clusterColours[clusterId] = []
                clusterAreas[clusterId] = 0
            clusterColours[clusterId].append((ccMeans[i], ccAreas[i][0]))
            clusterAreas[clusterId] += ccAreas[i][0]

        finalClusters = []

        for clusterId, colors in clusterColours.items():
            totalArea = clusterAreas[clusterId]
            weightedColourSum = np.sum([mean * area for mean, area in colors], axis=0)
            avgColour = weightedColourSum / totalArea
            
            areaRatio = totalArea / acceptedArea

            if areaRatio >= minColourAreaRatio:
                score = areaRatio
                # Adjust score based on how close the colour is to grayscale, in L1 distance
                b, g, r = avgColour
                t = (b + g + r) / 3
                dist = np.abs(b - t) + np.abs(g - t) + np.abs(r - t)
                penalty = dist / 340
                
                avgColour = np.round(avgColour).astype(np.uint8)
                score *= (1 - maxGreyscalePenalty) + maxGreyscalePenalty * penalty
                finalClusters.append((avgColour, score, areaRatio))

        finalClusters.sort(key=lambda x: x[1], reverse=True)

        pickedColour = finalClusters[0][0]

        lowerBound = np.array([max(pickedColour[0] - colourTolerance, 0),
                                max(pickedColour[1] - colourTolerance, 0),
                                max(pickedColour[2] - colourTolerance, 0)])

        upperBound = np.array([min(pickedColour[0] + colourTolerance, 255),
                                min(pickedColour[1] + colourTolerance, 255),
                                min(pickedColour[2] + colourTolerance, 255)])
        
        colourMask = cv.inRange(image, lowerBound, upperBound)

        meanfinalMask = cv.mean(colourMask)[0]
        hasDialog = meanfinalMask > minCcFinalMean
        
        if hasDialog:
            return colourMask
        else:
            return np.zeros_like(image[:, :, 0])

    def cvPassDialog(self, frame: cv.Mat, framePoint: FramePoint) -> bool:
        roiDialogText, hasDialog, debugFrame = self.ocrPass(frame)

        framePoint.setDebugFrame(debugFrame)

        roiDialogTextResized = cv.resize(roiDialogText, (150, 50))
        dctFeat = dctDescriptor(roiDialogTextResized, 8, 8)

        # inverseDctFeat = inverseDctDescriptor(dctFeat, 150, 50, 8, 8)
        # debugFrame = inverseDctFeat
        # debugFrame = cv.resize(inverseDctFeat, (roiDialogText.shape[1], roiDialogText.shape[0]))

        framePoint.setDebugFrame(debugFrame)

        framePoint.setFlag(BoxColourStatStrategy.FlagIndex.Dialog, hasDialog)
        framePoint.setFlag(BoxColourStatStrategy.FlagIndex.DialogFeat, dctFeat)

        return False

    def ocrPass(self, frame: cv.Mat) -> typing.Tuple[cv.Mat, bool, cv.Mat]:
        noFilterMode = False
        nonMajonSuppression = True

        image = self.dialogRect.cutRoi(frame)

        boxes = self.detectTextBoxes(image)

        finalMask: cv.Mat = np.zeros_like(image[:, :, 0])

        boxSizeSum = 0
        for box in boxes:
            x, y, w, h = box
            boxSizeSum += w * h

        boxes = sorted(boxes, key=lambda box: box[2] * box[3], reverse=True)

        for rank, box in enumerate(boxes):
            x, y, w, h = box
            if nonMajonSuppression:
                if w * h < 0.2 * boxSizeSum and rank >= 1:
                    break
            roi = image[y:y+h, x:x+w]
            if noFilterMode:
                mask = np.ones_like(roi[:, :, 0]) * 255
            else:
                mask = self.filterText(roi)
            finalMask[y:y+h, x:x+w] = mask

        hasDialog = np.mean(finalMask) > self.featureThreshold

        debugFrame = finalMask

        return finalMask, hasDialog, debugFrame
