import typing
import cv2 as cv
import argparse

from Rectangle import *
from Util import *

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--src", type=str, default="src.mp4", help="source video file")
    parser.add_argument("--dst", type=str, default="pyrdown.mp4", help="destination video file")
    parser.add_argument("--pyrdown", type=int, default=1, help="the number of rounds to pyramid down, each rounds halves width and height")
    args = parser.parse_args()
    if True: # data validity test
        srcMp4Test = open(args.src, "rb")
        srcMp4Test.close()
        dstMp4Test = open(args.dst, "wb")
        dstMp4Test.close()
        if not args.pyrdown >= 1:
            raise Exception("Invalid pyramid limit! ")
    
    srcMp4 = cv.VideoCapture(args.src)
    srcRect = SrcRectangle(srcMp4)
    fps: float = srcMp4.get(cv.CAP_PROP_FPS)
    size: typing.Tuple[int, int] = srcRect.getSizeInt()
    for _ in range (args.pyrdown):
        size = ((size[0] + 1) // 2, (size[1] + 1) // 2)
    dstMp4: typing.Any = cv.VideoWriter(args.dst, cv.VideoWriter_fourcc('m','p','4','v'), fps, size)

    print("==== Pyramid Down ====")
    while True:

        # Frame reading

        frameIndex: int = int(srcMp4.get(cv.CAP_PROP_POS_FRAMES))
        timestamp: int = int(srcMp4.get(cv.CAP_PROP_POS_MSEC))
        isValidFrame, frameOrig = srcMp4.read()
        if not isValidFrame:
            break

        if frameIndex % 1000 == 0:
            print("frame {} {}".format(frameIndex, formatTimestamp(timestamp)))

        frameOut = frameOrig
        for _ in range(args.pyrdown):
            frameOut = cv.pyrDown(frameOrig)
        dstMp4.write(frameOut)

    srcMp4.release()
    dstMp4.release()

if __name__ == "__main__":
    main()
