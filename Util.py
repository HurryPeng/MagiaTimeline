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

class CompressedDisk(diskcache.Disk):
    """Cache key and value using zlib compression."""

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
    global _tempLock, _tempDir, _tempDirPath, _diskCache
    with _tempLock:
        if tempDirPath is not None:
            _tempDirPath = tempDirPath
        else:
            _tempDir = tempfile.TemporaryDirectory(prefix="MagiaTimeline_")
            _tempDirPath = _tempDir.name
        _diskCache = diskcache.Cache(_tempDirPath, eviction_policy='none', disk=CompressedDisk)
        print(f"Disk cache initialized at {_tempDirPath}")

    @atexit.register
    def _cleanupCache():
        global _tempDir, _diskCache
        if _diskCache is not None:
            _diskCache.close()
        if _tempDir is not None:
            _tempDir.cleanup()

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
        self.future = _threadPool.submit(writeTask)

    def get(self) -> typing.Any:
        global _diskCache
        assert _diskCache is not None
        if not self.future.done():
            self.future.result()
        value = _diskCache[self.key]
        assert value is not None
        return value

def imwriteAsync(frame: cv.Mat, filename: str, params: typing.Optional[list[int]] = None) -> concurrent.futures.Future:
    global _threadPool
    future = _threadPool.submit(cv.imwrite, filename, frame, params)
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
    timeStr = dTimestamp.strftime("%H:%M:%S.%f")[:-3]
    return timeStr[:-1]

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
    return imageSobel

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
    imageErodeDialate = cv.morphologyEx(imageErode, cv.MORPH_DILATE, kernel=cv.getStructuringElement(cv.MORPH_ELLIPSE, (dilateWeight, dilateWeight)))
    imageWeightUpperBound = cv.bitwise_and(image, cv.bitwise_not(imageErodeDialate))
    return imageWeightUpperBound

def morphologyWeightLowerBound(image: cv.Mat, erodeWeight: int, dilateWeight: int) -> cv.Mat:
    imageErode = cv.morphologyEx(image, cv.MORPH_ERODE, kernel=cv.getStructuringElement(cv.MORPH_ELLIPSE, (erodeWeight, erodeWeight)))
    imageErodeDialate = cv.morphologyEx(imageErode, cv.MORPH_DILATE, kernel=cv.getStructuringElement(cv.MORPH_ELLIPSE, (dilateWeight, dilateWeight)))
    imageWeightLowerBound = cv.bitwise_and(image, imageErodeDialate)
    return imageWeightLowerBound

def morphologyNear(base: cv.Mat, ref: cv.Mat, Weight: int) -> cv.Mat:
    refDialate = cv.morphologyEx(ref, cv.MORPH_DILATE, kernel=cv.getStructuringElement(cv.MORPH_ELLIPSE, (Weight, Weight)))
    return cv.bitwise_and(base, refDialate)

def avFrame2CvMat(frame: av.frame.Frame) -> cv.Mat:
    return frame.to_ndarray(format='bgr24')

def ms2Timestamp(ms: int, timeBase: fractions.Fraction) -> int:
    return int(ms / timeBase / 1000)

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
        
        suffix = chr(ord(suffix) + 1)
        if suffix > 'z':
            raise Exception("Too many files with the same base name, please clean up the directory.")
