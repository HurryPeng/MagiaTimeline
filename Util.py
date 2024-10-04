import datetime
import typing
import cv2 as cv
import numpy as np
import fractions
import av.frame

def formatTimestamp(timeBase: fractions.Fraction, timestamp: int) -> str:
    dTimestamp = datetime.datetime.fromtimestamp(float(timestamp * timeBase), datetime.timezone(datetime.timedelta()))
    return dTimestamp.strftime("%H:%M:%S.%f")[:-4]

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

def ensureMat(frame):
    if isinstance(frame, cv.UMat):
        return frame.get()
    return frame

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
