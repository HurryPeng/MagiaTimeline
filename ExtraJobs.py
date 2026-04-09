from __future__ import annotations
import pytesseract
import paddleocr
import typing

import scipy.cluster.hierarchy
import scipy.spatial.distance
import sklearn.preprocessing

from IR import IIR
from Util import *
from AbstractFlagIndex import *
from IR import *
from Strategies.AbstractStrategy import *

class TextDetectionResult(AttachmentKey):
    """Stores text detection boxes for an image region.
    Crops are not stored; call cropTextFromImage(image) to produce them on demand.
    This class also serves as its own attachment key: use TextDetectionResult (the class
    itself) as the key when calling setAttachment/getAttachment."""
    def __init__(self, boxes: typing.Optional[typing.List[typing.Tuple[int, int, int, int]]] = None):
        self.boxes: typing.List[typing.Tuple[int, int, int, int]] = boxes if boxes is not None else []

    def cropTextFromImage(self, image: cv.Mat) -> typing.List[np.ndarray]:
        return [image[y:y + h, x:x + w].copy() for x, y, w, h in self.boxes]

class IIRTextDetectionPreprocessPass(IIRPass):
    def __init__(self, frameKey: type, config: dict):
        self.frameKey: type = frameKey
        self.nonMajorBoxSuppressionMaxRatio: float = config["nonMajorBoxSuppressionMaxRatio"]
        self.nonMajorBoxSuppressionMinRank: int = config["nonMajorBoxSuppressionMinRank"]
        suppressPaddleWarnings()
        self.detector = paddleocr.TextDetection(
            model_name="PP-OCRv4_mobile_det",
            model_dir="./PaddleOCRModels/official_models/PP-OCRv4_mobile_det",
            thresh=0.2,
            box_thresh=0.3,
            device="cpu",
            enable_mkldnn=True
        )

    def apply(self, iir: IIR):
        print(f"IIRTextDetectionPreprocessPass: processing {len(iir.intervals)} intervals")
        for i, interval in enumerate(iir.intervals):
            image: cv.Mat = interval.getAttachment(self.frameKey)
            if image is None:
                continue
            imgH, imgW = image.shape[:2]

            result = self.detector.predict(image)
            result = result[0]
            dtPolys: typing.List[np.ndarray] = result["dt_polys"]
            n = len(dtPolys)

            rawBoxes = []
            for j in range(n):
                poly = np.array(dtPolys[j], np.int32)
                x0, y0 = poly[0]
                x1, y1 = poly[1]
                x2, y2 = poly[2]
                x3, y3 = poly[3]
                angle0 = np.arctan2(y1 - y0, x1 - x0)
                angle3 = np.arctan2(y2 - y3, x2 - x3)
                angle = (angle0 + angle3) / 2
                if np.abs(angle) > np.pi / 180 * 3:
                    continue
                bx, by, bw, bh = cv.boundingRect(poly)
                bx = max(0, bx)
                by = max(0, by)
                bw = min(imgW - bx, bw)
                bh = min(imgH - by, bh)
                rawBoxes.append((bx, by, bw, bh))

            rawBoxes.sort(key=lambda b: b[2] * b[3], reverse=True)
            boxSizeSum = sum(b[2] * b[3] for b in rawBoxes)
            filteredBoxes = []
            for rank, box in enumerate(rawBoxes):
                if box[2] * box[3] <= self.nonMajorBoxSuppressionMaxRatio * boxSizeSum and rank >= self.nonMajorBoxSuppressionMinRank:
                    break
                filteredBoxes.append(box)
            # Sort boxes in reading order: group into rows by center-y proximity
            # (threshold = half the average box height), then sort each row by left-x.
            if filteredBoxes:
                avgH = sum(b[3] for b in filteredBoxes) / len(filteredBoxes)
                rowThreshold = avgH * 0.5
                filteredBoxes.sort(key=lambda b: b[1] + b[3] / 2)
                rows: typing.List[typing.List[typing.Tuple[int, int, int, int]]] = []
                currentRow: typing.List[typing.Tuple[int, int, int, int]] = [filteredBoxes[0]]
                currentRowBaseY: float = filteredBoxes[0][1] + filteredBoxes[0][3] / 2
                for box in filteredBoxes[1:]:
                    centerY = box[1] + box[3] / 2
                    if abs(centerY - currentRowBaseY) < rowThreshold:
                        currentRow.append(box)
                    else:
                        rows.append(currentRow)
                        currentRow = [box]
                        currentRowBaseY = centerY
                rows.append(currentRow)
                filteredBoxes = []
                for row in rows:
                    row.sort(key=lambda b: b[0])
                    filteredBoxes.extend(row)

            interval.setAttachment(TextDetectionResult, TextDetectionResult(filteredBoxes))

            if i % 10 == 0:
                print(interval.getName(i))


class IIROcrPass(IIRPass):
    def __init__(self, config: dict, dest: str, strategy: AbstractExtraJobStrategy):
        self.config: dict = config
        self.dest: str = dest
        self.strategy: AbstractExtraJobStrategy = strategy
        self.standaloneOutput: bool = config["standaloneOutput"]
        self.standaloneOutputSuffix: str = config["standaloneOutputSuffix"]
        self.separator: str = config["separator"]
        self.doPaddle: bool = config["doPaddle"]
        self.nonMajorBoxSuppressionMaxRatio: float = config["nonMajorBoxSuppressionMaxRatio"]
        self.nonMajorBoxSuppressionMinRank: int = config["nonMajorBoxSuppressionMinRank"]
        self.doTeseract: bool = config["doTesseract"]
        self.tesseractLang: str = config["tesseractLang"]

    def apply(self, iir: IIR):
        file = None
        if self.standaloneOutput:
            filename = self.dest + self.standaloneOutputSuffix
            print(f"Standalone output enabled. Writing to {filename}")
            file = open(filename, "w", encoding="utf-8")
        else:
            print("Standalone output disabled. Writing to ass file.")

        paddle = None
        if self.doPaddle:
            suppressPaddleWarnings()
            paddle = paddleocr.PaddleOCR(
                text_detection_model_name="PP-OCRv4_mobile_det",
                text_detection_model_dir="./PaddleOCRModels/official_models/PP-OCRv4_mobile_det",
                text_recognition_model_name="PP-OCRv5_mobile_rec",
                text_recognition_model_dir="./PaddleOCRModels/official_models/PP-OCRv5_mobile_rec",
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=False,
                device="cpu",
                enable_mkldnn=True
            )

        frameKey: type = self.strategy.getExtraJobFrameKey()

        for i, interval in enumerate(iir.intervals):
            buff: str = ""
            name: str = interval.getName(i)
            image: cv.Mat = interval.getAttachment(frameKey)

            if self.doPaddle and paddle is not None and image is not None:
                result = paddle.predict(image)
                result = result[0]
                recTexts: typing.List[str] = result["rec_texts"]
                recBoxes: np.ndarray = result["rec_boxes"]
                recPolys: typing.List[np.ndarray] = result["rec_polys"]
                recBoxSizes = [
                    (int(b[2]) - int(b[0])) * (int(b[3]) - int(b[1]))
                    for b in recBoxes
                ]
                boxSizeSum = sum(recBoxSizes)

                passesAngle: typing.List[bool] = []
                for poly in recPolys:
                    x0, y0 = poly[0]; x1, y1 = poly[1]
                    x2, y2 = poly[2]; x3, y3 = poly[3]
                    angle = (np.arctan2(y1 - y0, x1 - x0) + np.arctan2(y2 - y3, x2 - x3)) / 2
                    passesAngle.append(bool(np.abs(angle) <= np.pi / 180 * 3))

                sortedIdx = sorted(
                    range(len(recBoxSizes)),
                    key=lambda j: recBoxSizes[j] if passesAngle[j] else 0,
                    reverse=True
                )
                rankMap = [0] * len(recBoxSizes)
                for rank, origIdx in enumerate(sortedIdx):
                    rankMap[origIdx] = rank

                paddleText = ""
                for j, (line, boxSize) in enumerate(zip(recTexts, recBoxSizes)):
                    if not passesAngle[j]:
                        continue
                    if boxSize > self.nonMajorBoxSuppressionMaxRatio * boxSizeSum \
                            or rankMap[j] < self.nonMajorBoxSuppressionMinRank:
                        paddleText += line + " "
                buff += paddleText.strip()

            if self.doTeseract:
                if buff != "":
                    buff += self.separator
                if image is not None:
                    tesseractText: str = pytesseract.image_to_string(
                        ensureMat(image), config=f"-l {self.tesseractLang} --psm 6"
                    )
                    buff += tesseractText[:-1].replace("\n", "")

            if file is not None:
                file.write(f"{name},{buff}\n")
            else:
                interval.text = buff

            if i % 10 == 0:
                print(name)

        if file is not None:
            file.close()
            print(f"Output written to {file.name}")

def extractColourClusters(
    roi: cv.Mat,
    sobelThreshold: int,
    minCcAreaRatio: float,
    maxCcAreaRatio: float,
    maxCcStddev: float,
    clusterThreshold: float,
    minColourAreaRatio: float,
) -> typing.List[typing.Tuple[np.ndarray, float]]:
    """Extract colour clusters from a text box ROI.
    Returns [(avgColourBGR, areaRatio), ...] sorted descending by area ratio.
    avgColourBGR is a shape (3,) float array;
    areaRatio is the cluster's share of total accepted CC area (0.0~1.0).
    Returns empty list if no valid regions are found."""

    imageSobel = rgbSobel(roi, 1)
    imageSobelBin = cv.threshold(imageSobel, sobelThreshold, 255, cv.THRESH_BINARY_INV)[1]

    area = roi.shape[0] * roi.shape[1]
    minCcArea = max(minCcAreaRatio * area, 10)
    maxCcArea = maxCcAreaRatio * area

    nLabels, labels, stats, centroids = cv.connectedComponentsWithStats(imageSobelBin, connectivity=4, ltype=cv.CV_32S)

    acceptedIds = []
    ccMeans = []
    acceptedArea = 0

    for i in range(nLabels):
        ccArea = stats[i][cv.CC_STAT_AREA]
        if ccArea >= minCcArea and ccArea <= maxCcArea:
            mask = np.where(labels == i, 255, 0).astype(np.uint8)
            mean, std = cv.meanStdDev(roi, mask=mask)
            std = np.mean(std)
            if std < maxCcStddev:
                acceptedIds.append(i)
                ccMeans.append(mean)
                acceptedArea += ccArea

    if len(acceptedIds) == 0:
        return []

    if len(acceptedIds) == 1:
        return [(ccMeans[0].flatten(), 1.0)]

    ccMeans = np.array(ccMeans).reshape(len(ccMeans), -1)
    ccAreas = np.array([stats[i][cv.CC_STAT_AREA] for i in acceptedIds]).reshape(-1, 1)

    scaler = sklearn.preprocessing.StandardScaler()
    ccMeansScaled = scaler.fit_transform(ccMeans)

    distMat = scipy.spatial.distance.pdist(ccMeansScaled, metric='euclidean')
    Z = scipy.cluster.hierarchy.linkage(distMat, method='weighted')
    clusters = scipy.cluster.hierarchy.fcluster(Z, clusterThreshold, criterion='distance')

    clusterColours: typing.Dict[int, typing.List[typing.Tuple[np.ndarray, float]]] = {}
    clusterAreas: typing.Dict[int, float] = {}

    for i, clusterId in enumerate(clusters):
        if clusterId not in clusterColours:
            clusterColours[clusterId] = []
            clusterAreas[clusterId] = 0
        clusterColours[clusterId].append((ccMeans[i], ccAreas[i][0]))
        clusterAreas[clusterId] += ccAreas[i][0]

    result: typing.List[typing.Tuple[np.ndarray, float]] = []
    for clusterId, colours in clusterColours.items():
        totalArea = clusterAreas[clusterId]
        weightedColourSum = np.sum([mean * a for mean, a in colours], axis=0)
        avgColour = weightedColourSum / totalArea
        areaRatio = totalArea / acceptedArea
        if areaRatio >= minColourAreaRatio:
            result.append((avgColour.flatten(), areaRatio))

    result.sort(key=lambda x: x[1], reverse=True)
    return result

class IIRStyleClassifyPass(IIRPass):
    def __init__(self, config: dict):
        # Soft histogram parameters
        self.histBins: int = config["histBins"]

        # extractColourClusters parameters
        self.sobelThreshold: int = config["sobelThreshold"]
        self.minCcAreaRatio: float = config["minCcAreaRatio"]
        self.maxCcAreaRatio: float = config["maxCcAreaRatio"]
        self.maxCcStddev: float = config["maxCcStddev"]
        self.clusterThreshold: float = config["clusterThreshold"]
        self.minColourAreaRatio: float = config["minColourAreaRatio"]

        # Inter-interval clustering parameters
        self.clusterDistThreshold: float = config["clusterDistThreshold"]
        self.minIntervalCount: int = config["minIntervalCount"]

        # Feature type: "hsvCone2d" (default, 2D, easy to visualise) or "softHistogram" (8x8 grid, archived)
        self.featureType: str = config.get("featureType", "hsvCone2d")

        # Output parameters
        self.styleNames: typing.List[str] = config.get("styleNames", [])
        self.baseStyleTemplate: str = config.get(
            "baseStyleTemplate",
            "Style: {name},Microsoft YaHei,80,{primaryColour},&H000000FF,"
            "&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,2,2,2,10,10,200,1"
        )

    def bgrToHsvCone(self, bgr: np.ndarray) -> typing.Tuple[float, float]:
        """Convert a BGR colour to HSV cone Cartesian coordinates (x, y) in [-1, 1]."""
        pixel = np.array([[bgr]], dtype=np.uint8)
        hsv = cv.cvtColor(pixel, cv.COLOR_BGR2HSV)[0][0]
        h, s, _ = hsv
        h_rad = np.deg2rad(float(h)) * 2  # OpenCV H is [0, 180), actual [0, 360)
        s_01 = float(s) / 255.0
        x = s_01 * np.cos(h_rad)
        y = s_01 * np.sin(h_rad)
        return x, y

    def buildFeatureSoftHistogram(self, colourClusters: typing.List[typing.Tuple[np.ndarray, float]]) -> np.ndarray:
        """[Archived] Build an 8x8 HSV-cone soft histogram from colour clusters via bilinear interpolation.
        Returns a flattened histBins*histBins vector."""
        G = self.histBins
        hist = np.zeros((G, G), dtype=np.float64)

        for avgColour, areaRatio in colourClusters:
            x, y = self.bgrToHsvCone(avgColour)

            # Map from [-1, 1] to [0, G-1]
            gx = (x + 1.0) / 2.0 * (G - 1)
            gy = (y + 1.0) / 2.0 * (G - 1)

            # Bilinear soft assignment
            gx0 = int(np.floor(gx))
            gy0 = int(np.floor(gy))
            gx1 = gx0 + 1
            gy1 = gy0 + 1

            fx = gx - gx0
            fy = gy - gy0

            gx0 = max(0, min(gx0, G - 1))
            gx1 = max(0, min(gx1, G - 1))
            gy0 = max(0, min(gy0, G - 1))
            gy1 = max(0, min(gy1, G - 1))

            hist[gy0, gx0] += areaRatio * (1 - fx) * (1 - fy)
            hist[gy0, gx1] += areaRatio * fx * (1 - fy)
            hist[gy1, gx0] += areaRatio * (1 - fx) * fy
            hist[gy1, gx1] += areaRatio * fx * fy

        return hist.flatten()

    def buildFeatureMeanHsvCone(self, colourClusters: typing.List[typing.Tuple[np.ndarray, float]]) -> np.ndarray:
        """Build a 2D HSV-cone feature from colour clusters by area-weighted mean.
        Maps each colour to (s·cos(2h), s·sin(2h)), then computes a weighted average.
        Achromatic colours (black / white / grey) have s≈0 and naturally converge to the
        origin, so they contribute little to the mean without any explicit penalty.
        Returns a shape (2,) float64 vector."""
        x_acc, y_acc = 0.0, 0.0
        w_acc = 0.0
        for avgColour, areaRatio in colourClusters:
            x, y = self.bgrToHsvCone(avgColour)
            x_acc += x * areaRatio
            y_acc += y * areaRatio
            w_acc += areaRatio
        if w_acc > 0:
            return np.array([x_acc / w_acc, y_acc / w_acc], dtype=np.float64)
        return np.zeros(2, dtype=np.float64)

    def buildFeature(self, colourClusters: typing.List[typing.Tuple[np.ndarray, float]]) -> np.ndarray:
        """Dispatch to the configured feature builder.
        Currently supported featureType values:
          "hsvCone2d"     -- 2D area-weighted mean in HSV cone space (default, easy to visualise)
          "softHistogram" -- 8x8 HSV-cone soft histogram (archived, higher-dimensional)
        """
        if self.featureType == "softHistogram":
            return self.buildFeatureSoftHistogram(colourClusters)
        return self.buildFeatureMeanHsvCone(colourClusters)

    def computeAllFeatures(self, iir: IIR) -> typing.Tuple[
        typing.List[typing.Optional[np.ndarray]],
        typing.List[typing.Optional[np.ndarray]]
    ]:
        """Compute per-interval features and representative colours for all intervals.
        Returns (features, repColours) where each is a list parallel to iir.intervals.
        None entries indicate intervals with no valid colour data."""
        features: typing.List[typing.Optional[np.ndarray]] = []
        repColours: typing.List[typing.Optional[np.ndarray]] = []

        featureDim: typing.Optional[int] = None  # inferred from first valid feature

        for i, interval in enumerate(iir.intervals):
            image: cv.Mat = interval.getAttachment(ExtraJobFrameKey)
            tdf: TextDetectionResult = interval.getAttachment(TextDetectionResult)

            if image is None or tdf is None or not tdf.boxes:
                features.append(None)
                repColours.append(None)
                continue

            # Accumulate per-box features weighted by crop area
            crops = tdf.cropTextFromImage(image)
            aggregatedFeature: typing.Optional[np.ndarray] = None
            totalBoxArea = 0
            bestRepColour: typing.Optional[np.ndarray] = None
            bestRepArea = 0.0  # absolute area of the best representative colour

            for crop in crops:
                if crop.size == 0:
                    continue

                clusters = extractColourClusters(
                    np.asarray(crop),
                    self.sobelThreshold,
                    self.minCcAreaRatio,
                    self.maxCcAreaRatio,
                    self.maxCcStddev,
                    self.clusterThreshold,
                    self.minColourAreaRatio,
                )

                if not clusters:
                    continue

                boxArea = crop.shape[1] * crop.shape[0]
                boxFeature = self.buildFeature(clusters)
                if featureDim is None:
                    featureDim = len(boxFeature)
                if aggregatedFeature is None:
                    aggregatedFeature = np.zeros(featureDim, dtype=np.float64)
                aggregatedFeature += boxFeature * boxArea
                totalBoxArea += boxArea

                # Track representative colour: top-1 colour from the box with largest absolute area contribution
                topColour, topAreaRatio = clusters[0]
                absArea = topAreaRatio * boxArea
                if absArea > bestRepArea:
                    bestRepArea = absArea
                    bestRepColour = topColour

            if totalBoxArea == 0 or aggregatedFeature is None:
                features.append(None)
                repColours.append(None)
                continue

            aggregatedFeature /= totalBoxArea  # weighted mean (not sum) across boxes

            # L2 normalise only for softHistogram; hsvCone2d is already a bounded 2D point
            if self.featureType == "softHistogram":
                norm = np.linalg.norm(aggregatedFeature)
                if norm > 0:
                    aggregatedFeature /= norm

            features.append(aggregatedFeature)
            repColours.append(bestRepColour)

            if i % 10 == 0:
                print(interval.getName(i))

        return features, repColours

    def cluster(self, features: typing.List[typing.Optional[np.ndarray]], validIndices: typing.List[int]) -> typing.List[int]:
        """Cluster valid intervals by their features.
        Uses euclidean distance for hsvCone2d (2D point cloud) and cosine for softHistogram.
        Returns cluster assignments parallel to validIndices."""
        if len(validIndices) == 1:
            return [0]

        featureMat = np.array([features[i] for i in validIndices])
        # hsvCone2d features are 2D Euclidean points; softHistogram features are
        # L2-normalised high-dim vectors where cosine distance is more appropriate.
        metric = 'euclidean' if self.featureType == 'hsvCone2d' else 'cosine'
        distMat = scipy.spatial.distance.pdist(featureMat, metric=metric)
        # Replace NaN distances (from zero vectors) with max distance
        distMat = np.nan_to_num(distMat, nan=1.0)
        Z = scipy.cluster.hierarchy.linkage(distMat, method='average')
        rawLabels = scipy.cluster.hierarchy.fcluster(Z, self.clusterDistThreshold, criterion='distance')
        return rawLabels.tolist()

    def computeClusterColour(self, repColours: typing.List[typing.Optional[np.ndarray]], memberIndices: typing.List[int]) -> np.ndarray:
        """Average the representative colours of cluster members."""
        validColours = [repColours[i] for i in memberIndices if repColours[i] is not None]
        if len(validColours) > 0:
            return np.mean(np.array(validColours), axis=0).astype(np.uint8)
        return np.array([255, 255, 255], dtype=np.uint8)  # fallback: white

    def apply(self, iir: IIR):
        print(f"IIRStyleClassifyPass: processing {len(iir.intervals)} intervals")

        # 1. Compute features and representative colours
        features, repColours = self.computeAllFeatures(iir)

        # 2. Filter intervals with valid features
        validIndices = [i for i, f in enumerate(features) if f is not None]
        print(f"IIRStyleClassifyPass: {len(validIndices)} intervals have valid colour features")

        if len(validIndices) < 2:
            print("IIRStyleClassifyPass: not enough valid intervals for clustering, skipping")
            return

        # 3. Agglomerative clustering
        clusterAssignments = self.cluster(features, validIndices)

        # 4. Reorder clusters by member count (largest first) for stable naming
        from collections import Counter
        clusterCounts = Counter(clusterAssignments)
        sizeOrder = [cid for cid, _ in clusterCounts.most_common()]
        remapping = {old: new for new, old in enumerate(sizeOrder)}
        clusterAssignments = [remapping[c] for c in clusterAssignments]

        nClusters = len(set(clusterAssignments))
        print(f"IIRStyleClassifyPass: {nClusters} clusters found")

        # 5. Generate style declarations and assign styles to intervals
        clusterStyleNames: typing.Dict[int, str] = {}
        newStyleLines: typing.List[str] = []

        for clusterId in sorted(set(clusterAssignments)):
            memberIndices = [validIndices[i] for i, c in enumerate(clusterAssignments) if c == clusterId]
            if len(memberIndices) < self.minIntervalCount:
                continue

            repColour = self.computeClusterColour(repColours, memberIndices)

            if self.styleNames and clusterId < len(self.styleNames):
                styleName = self.styleNames[clusterId]
            else:
                styleName = f"StyleClass_{clusterId}"

            b, g, r = repColour
            assColour = f"&H00{b:02X}{g:02X}{r:02X}"
            styleLine = self.baseStyleTemplate.format(name=styleName, primaryColour=assColour)
            newStyleLines.append(styleLine)
            clusterStyleNames[clusterId] = styleName

        # Atomically: append style declarations AND assign interval styles
        iir.styles.extend(newStyleLines)

        for i, clusterId in zip(validIndices, clusterAssignments):
            if clusterId in clusterStyleNames:
                iir.intervals[i].style = clusterStyleNames[clusterId]

        print(f"IIRStyleClassifyPass: assigned {len(clusterStyleNames)} styles, "
              f"declared {len(newStyleLines)} style lines")
