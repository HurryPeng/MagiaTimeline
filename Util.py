from __future__ import annotations
import datetime
import typing
import cv2 as cv
import numpy as np
import fractions
import av.frame
import tempfile
import threading
import uuid
import diskcache
import atexit
import pickle
import lz4
import concurrent.futures
import os
import warnings
import dataclasses
import paddleocr

class CompressedDisk(diskcache.Disk):
    """Cache key and value using lz4 compression."""

    def __init__(self, directory, compress_level=4, **kwargs):
        self.compress_level = compress_level
        super().__init__(directory, **kwargs)

    def put(self, key):
        pickle_bytes = pickle.dumps(key)
        data = lz4.frame.compress(pickle_bytes, compression_level=self.compress_level)
        return super().put(data)

    def get(self, key, raw):
        data = super().get(key, raw)
        return pickle.loads(lz4.frame.decompress(data)) if data else None

    def store(self, value, read, key):
        if not read:
            pickle_bytes = pickle.dumps(value)
            value = lz4.frame.compress(pickle_bytes, compression_level=self.compress_level)
        return super().store(value, read, key=key)

    def fetch(self, mode, filename, value, read):
        data = super().fetch(mode, filename, value, read)
        if not read:
            data = pickle.loads(lz4.frame.decompress(data))
        return data

# Global thread pool for disk cache and async writes
_threadPool: concurrent.futures.ThreadPoolExecutor = concurrent.futures.ThreadPoolExecutor(max_workers=4)
def getThreadPool() -> concurrent.futures.ThreadPoolExecutor:
    global _threadPool
    return _threadPool

# Private globals for cache initialization
_tempLock = threading.Lock()
_tempDir: typing.Optional[tempfile.TemporaryDirectory] = None
_tempDirPath: typing.Optional[str] = None
_diskCache: typing.Optional[diskcache.Cache] = None

def initDiskCache(tempDirPath: typing.Optional[str] = None):
    # If tempDirPath is provided, use it and whoever provided it is responsible for cleaning it up.
    # If not, create a temporary directory that will be cleaned up automatically.
    global _tempLock, _tempDir, _tempDirPath, _diskCache, _threadPool

    if _diskCache is not None:
        return

    def _cleanupCache():
        _threadPool.shutdown(wait=True)
        if _diskCache is not None:
            _diskCache.close()
        if _tempDir is not None:
            _tempDir.cleanup()

    atexit.register(_cleanupCache)

    with _tempLock:
        # Initialize the temporary directory
        if tempDirPath is not None:
            _tempDirPath = tempDirPath
        else:
            _tempDir = tempfile.TemporaryDirectory(prefix="MagiaTimeline_", delete=False)
            _tempDirPath = _tempDir.name
        _diskCache = diskcache.Cache(_tempDirPath, eviction_policy='none', disk=CompressedDisk)
        print(f"Disk cache initialized at {_tempDirPath}")

def clearDiskCache():
    global _diskCache
    assert _diskCache is not None
    _diskCache.clear()

def getDiskCache() -> diskcache.Cache:
    global _diskCache
    assert _diskCache is not None
    return _diskCache

class DiskCacheHandle:
    def __init__(self, value: typing.Any):
        global _diskCache, _threadPool
        self.key = uuid.uuid4().hex
        def writeTask():
            assert _diskCache is not None
            _diskCache[self.key] = value
            _diskCache.close() # Close for this thread
        self.future = _threadPool.submit(writeTask)

    def get(self) -> typing.Any:
        global _diskCache
        assert _diskCache is not None
        if not self.future.done():
            self.future.result()
        value = _diskCache[self.key]
        assert value is not None
        return value

def imwriteAsync(filename: str, frame: cv.Mat) -> concurrent.futures.Future:
    global _threadPool
    future = _threadPool.submit(cv.imwrite, filename, frame)
    return future

def containsLargeNdarray(obj: typing.Any) -> bool:
    """
    Recursively check whether `obj` (which may be a list or tuple
    of arbitrary nesting) contains at least one numpy.ndarray whose
    size is greater than 1kB.
    Returns True on the first array found; otherwise False.
    """
    # Base case: found an ndarray
    if isinstance(obj, np.ndarray) and obj.size > 1024:
        return True
    # If it's a list or tuple, recurse into each element
    if isinstance(obj, (list, tuple)):
        for item in obj:
            if containsLargeNdarray(item):
                return True
    # All other types are ignored
    return False

def formatTimestamp(timeBase: fractions.Fraction, timestamp: int) -> str:
    dTimestamp = datetime.datetime.fromtimestamp(float(timestamp * timeBase), datetime.timezone(datetime.timedelta()))
    # strftime("%f") produces 6-digit microseconds; [:-3] trims to milliseconds; [:-1] trims to centiseconds for ASS format
    timeStr = dTimestamp.strftime("%H:%M:%S.%f")[:-3]
    return timeStr[:-1]

def formatTimestampSrt(timeBase: fractions.Fraction, timestamp: int) -> str:
    dTimestamp = datetime.datetime.fromtimestamp(float(timestamp * timeBase), datetime.timezone(datetime.timedelta()))
    return dTimestamp.strftime("%H:%M:%S,%f")[:-3]

def inRange(frame, lower: typing.List[int], upper: typing.List[int]):
    # just a syntactic sugar
    return cv.inRange(frame, np.array(lower), np.array(upper))

def cosineSimilarity(lhs: np.ndarray, rhs: np.ndarray):
    return np.dot(lhs, rhs) / (np.linalg.norm(lhs) * np.linalg.norm(rhs))

def dctDescriptor(image: cv.Mat, dctWeight=8, dctHeight=8) -> np.ndarray:
    dct: cv.Mat = cv.dct(np.float32(image))
    dctLowFreq = dct[:dctHeight, :dctWeight]
    # Double the weight of the top-left quarter of the DCT
    # dctLowFreq[:dctHeight//2, :dctWeight//2] *= 2
    dctVec = dctLowFreq.flatten()
    # Handle when the image is all black
    if np.linalg.norm(dctVec) == 0:
        return dctVec
    dctVec /= np.linalg.norm(dctVec)
    return dctVec

def inverseDctDescriptor(dctVec: np.ndarray, originalWidth: int, originalHeight: int, dctWeight=8, dctHeight=8) -> cv.Mat:
    dctLowFreq = dctVec.reshape((dctHeight, dctWeight))
    dct = np.zeros((originalHeight, originalWidth), dtype=np.float32)
    dct[:dctHeight, :dctWeight] = dctLowFreq
    reconstructedImage = cv.idct(dct)
    return cv.normalize(reconstructedImage, None, 0, 255, cv.NORM_MINMAX, cv.CV_8U)

def rgbSobel(image: cv.Mat, ksize: int) -> cv.Mat:
    imageChannels = cv.split(image)
    imageSobelRX = cv.Sobel(imageChannels[2], cv.CV_16S, 1, 0, ksize=ksize)
    imageSobelRY = cv.Sobel(imageChannels[2], cv.CV_16S, 0, 1, ksize=ksize)
    imageSobelGX = cv.Sobel(imageChannels[1], cv.CV_16S, 1, 0, ksize=ksize)
    imageSobelGY = cv.Sobel(imageChannels[1], cv.CV_16S, 0, 1, ksize=ksize)
    imageSobelBX = cv.Sobel(imageChannels[0], cv.CV_16S, 1, 0, ksize=ksize)
    imageSobelBY = cv.Sobel(imageChannels[0], cv.CV_16S, 0, 1, ksize=ksize)
    imageSobelR = cv.convertScaleAbs(cv.addWeighted(cv.convertScaleAbs(imageSobelRX), 1, cv.convertScaleAbs(imageSobelRY), 1, 0))
    imageSobelG = cv.convertScaleAbs(cv.addWeighted(cv.convertScaleAbs(imageSobelGX), 1, cv.convertScaleAbs(imageSobelGY), 1, 0))
    imageSobelB = cv.convertScaleAbs(cv.addWeighted(cv.convertScaleAbs(imageSobelBX), 1, cv.convertScaleAbs(imageSobelBY), 1, 0))
    imageSobel = cv.convertScaleAbs(cv.addWeighted(cv.addWeighted(imageSobelR, 1/3, imageSobelG, 1/3, 0), 1, imageSobelB, 1/3, 0))
    return typing.cast(cv.Mat, imageSobel)

def rgbDiffMask(lhs: cv.Mat, rhs: cv.Mat, threshold: int) -> cv.Mat:
    diff = cv.absdiff(lhs, rhs)
    return cv.bitwise_not(cv.inRange(diff, (0, 0, 0), (threshold, threshold, threshold)))

def ensureMat(frame):
    if isinstance(frame, cv.UMat):
        return frame.get()
    return frame

def maxResScaleDown(
    src: cv.Mat,
    maxResolution: int = 1800
) -> typing.Tuple[cv.Mat, int]:
    scale: int = 1
    while src.shape[0] // scale > maxResolution or src.shape[1] // scale > maxResolution:
        scale *= 2
    dst = src
    if scale > 1:
        dst = cv.resize(src, (src.shape[1] // scale, src.shape[0] // scale), interpolation=cv.INTER_AREA)
    return dst, scale

def phaseCorrelateMaxRes(
    src1: cv.Mat,
    src2: cv.Mat,
    maxResolution: int = 1800,
) -> typing.Tuple[cv.typing.Point2d, float]:
    src1, scale1 = maxResScaleDown(src1, maxResolution)
    src2, scale2 = maxResScaleDown(src2, maxResolution)
    assert scale1 == scale2
    hann = cv.createHanningWindow(src1.shape[::-1], cv.CV_32F)
    (shiftX, shiftY), response = cv.phaseCorrelate(src1, src2, window=hann)
    shiftX *= scale1
    shiftY *= scale2
    return (shiftX, shiftY), response

def morphologyWeightUpperBound(image: cv.Mat, erodeWeight: int, dilateWeight: int) -> cv.Mat:
    imageErode = cv.morphologyEx(image, cv.MORPH_ERODE, kernel=cv.getStructuringElement(cv.MORPH_ELLIPSE, (erodeWeight, erodeWeight)))
    imageErodeDilate = cv.morphologyEx(imageErode, cv.MORPH_DILATE, kernel=cv.getStructuringElement(cv.MORPH_ELLIPSE, (dilateWeight, dilateWeight)))
    imageWeightUpperBound = cv.bitwise_and(image, cv.bitwise_not(imageErodeDilate))
    return imageWeightUpperBound

def morphologyWeightLowerBound(image: cv.Mat, erodeWeight: int, dilateWeight: int) -> cv.Mat:
    imageErode = cv.morphologyEx(image, cv.MORPH_ERODE, kernel=cv.getStructuringElement(cv.MORPH_ELLIPSE, (erodeWeight, erodeWeight)))
    imageErodeDilate = cv.morphologyEx(imageErode, cv.MORPH_DILATE, kernel=cv.getStructuringElement(cv.MORPH_ELLIPSE, (dilateWeight, dilateWeight)))
    imageWeightLowerBound = cv.bitwise_and(image, imageErodeDilate)
    return imageWeightLowerBound

def morphologyNear(base: cv.Mat, ref: cv.Mat, weight: int) -> cv.Mat:
    refDilate = cv.morphologyEx(ref, cv.MORPH_DILATE, kernel=cv.getStructuringElement(cv.MORPH_ELLIPSE, (weight, weight)))
    return cv.bitwise_and(base, refDilate)

def avFrame2CvMat(frame: av.frame.Frame, scaleDown: int) -> cv.Mat:
    image = frame.to_ndarray(format='bgr24')
    if scaleDown > 1:
        image = cv.resize(image, (frame.width // scaleDown, frame.height // scaleDown), interpolation=cv.INTER_AREA)
    return image

def ms2Timestamp(ms: int, timeBase: fractions.Fraction) -> int:
    return int(ms / timeBase / 1000)

class AttachmentKey:
    """Base class for external attachment keys used with FramePoint/Interval.attachments."""
    pass

# Generate a unique filename based on the source path.
# If the file already exists, try the next letter. '
# Example: "./video.mp4" -> "./video#20250603a" (no extension).
# If any file starting with "./video#20250603a" already exists (any suffix),
# e.g. "./video#20250603a.ass", "./video#20250603a-ref.txt" etc.,
# then try "./video#20250603b", "./video#20250603c", etc.
def autoNumberedNaming(srcPath: str) -> str:
    base, ext = os.path.splitext(srcPath)
    dateStr = datetime.datetime.now().strftime("%Y%m%d")
    suffix = 'a'
    
    dirPath = os.path.dirname(srcPath) or '.'
    
    while True:
        targetPrefix = f"{base}#{dateStr}{suffix}"
        
        conflictExists = False
        for filename in os.listdir(dirPath):
            if filename.startswith(os.path.basename(targetPrefix)):
                conflictExists = True
                break
        
        if not conflictExists:
            return targetPrefix
        
        suffix = nextSuffix(suffix)
        if len(suffix) > 4:
            raise Exception("Too many files with the same base name, please clean up the directory.")

def nextSuffix(s: str) -> str:
    """Increment alphabetic suffix: a->b->...->z->aa->ab->...->az->ba->...->zz->aaa"""
    if not s:
        return 'a'
    if s[-1] < 'z':
        return s[:-1] + chr(ord(s[-1]) + 1)
    else:
        return nextSuffix(s[:-1]) + 'a'

def checkerboardBackground(width: int, height: int, squareSize: int = 8) -> cv.Mat:
    """Generate a BGR checkerboard image resembling Photoshop's transparency background.
    Light squares are (192, 192, 192) and dark squares are (128, 128, 128)."""
    rows = np.arange(height) // squareSize
    cols = np.arange(width) // squareSize
    checker = (rows[:, None] + cols[None, :]) % 2  # 0 = light, 1 = dark
    channel = np.where(checker, 128, 192).astype(np.uint8)
    img = np.stack([channel, channel, channel], axis=2)
    return typing.cast(cv.Mat, img)

@dataclasses.dataclass
class TextDetectionResult:
    """Result of a text detection pass including the raw probability map."""
    boxes: list  # list of dt_polys (numpy arrays of polygon vertices)
    scores: list  # list of float confidence scores
    probabilityMap: np.ndarray  # float32, same size as input frame/crop

class PaddleTextDetectionAdapter:
    """Wraps a paddleocr.TextDetection instance to expose the raw prob map.
    This implementation is based on paddlepaddle==3.1.0, paddlex[ocr]==3.1.3, paddleocr==3.1.0

    Usage:
        ocr = paddleocr.TextDetection(...)
        adapter = PaddleTextDetectionAdapter(ocr)
        result = adapter.detect(frame)
        # result.boxes, result.scores, result.probabilityMap
    """

    def __init__(self, text_detection: paddleocr.TextDetection) -> None:
        # Validate that internal attributes exist
        assert hasattr(text_detection, "paddlex_predictor"), (
            "paddleocr.TextDetection missing 'paddlex_predictor' attribute. "
            "Check paddleocr version compatibility."
        )
        self.predictor = text_detection.paddlex_predictor
        p = self.predictor
        assert hasattr(p, "pre_tfs"), "PaddleX predictor missing 'pre_tfs'"
        assert hasattr(p, "infer"), "PaddleX predictor missing 'infer'"
        assert hasattr(p, "post_op"), "PaddleX predictor missing 'post_op'"

    def detect(self, frame: np.ndarray) -> TextDetectionResult:
        """Run text detection on a single frame and return boxes + probability map.

        Args:
            frame: BGR uint8 image (numpy array).

        Returns:
            TextDetectionResult with boxes, scores, and probabilityMap
            (float32, same spatial size as input frame).
        """
        p = self.predictor

        # Replicate PaddleX TextDetPredictor.process() for a single image
        batch_raw_imgs = p.pre_tfs["Read"](imgs=[frame])
        batch_imgs, batch_shapes = p.pre_tfs["Resize"](
            imgs=batch_raw_imgs,
            limit_side_len=p.limit_side_len,
            limit_type=p.limit_type,
            max_side_limit=p.max_side_limit,
        )
        batch_imgs = p.pre_tfs["Normalize"](imgs=batch_imgs)
        batch_imgs = p.pre_tfs["ToCHW"](imgs=batch_imgs)
        x = p.pre_tfs["ToBatch"](imgs=batch_imgs)

        # Forward pass: get raw predictions (probability map)
        preds = p.infer(x=x)

        # Post-process to get boxes and scores
        polys, scores = p.post_op(
            preds,
            batch_shapes,
            thresh=p.thresh,
            box_thresh=p.box_thresh,
            unclip_ratio=p.unclip_ratio,
        )

        # Extract probability map: preds[0] shape is (batch, 1, H, W)
        prob_map = preds[0][0, 0]  # float32, 2D, detector-resolution

        # Resize probability map back to original frame size
        src_h, src_w, _, _ = batch_shapes[0]
        prob_map_resized = cv.resize(
            prob_map,
            (int(src_w), int(src_h)),
            interpolation=cv.INTER_LINEAR,
        )

        return TextDetectionResult(
            boxes=polys[0],
            scores=scores[0],
            probabilityMap=prob_map_resized,
        )

def suppressPaddleWarnings():
    warnings.filterwarnings(
        "ignore",
        message="No ccache found",
        category=UserWarning,
        module="paddle.utils.cpp_extension"
    )
