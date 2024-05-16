import datetime
import typing
import cv2 as cv
import numpy as np

def formatTimestamp(timestamp: float) -> str:
    dTimestamp = datetime.datetime.fromtimestamp(timestamp / 1000, datetime.timezone(datetime.timedelta()))
    return dTimestamp.strftime("%H:%M:%S.%f")[:-4]

def inRange(frame, lower: typing.List[int], upper: typing.List[int]):
    # just a syntactic sugar
    return cv.inRange(frame, np.array(lower), np.array(upper))

def cosineSimilarity(lhs: np.ndarray, rhs: np.ndarray):
    return np.dot(lhs, rhs) / (np.linalg.norm(lhs) * np.linalg.norm(rhs))

def dctDescriptor(image: cv.Mat, dctWidth=8, dctHeight=8) -> np.ndarray:
    dct: cv.Mat = cv.dct(np.float32(image))
    dctLowFreq = dct[:dctHeight, :dctWidth]
    # Double the weight of the top-left quarter of the DCT
    # dctLowFreq[:dctHeight//2, :dctWidth//2] *= 2
    dctVec = dctLowFreq.flatten()
    # Handle when the image is all black
    if np.linalg.norm(dctVec) == 0:
        return dctVec
    dctVec /= np.linalg.norm(dctVec)
    return dctVec
