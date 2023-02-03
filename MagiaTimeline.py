from __future__ import annotations
import typing
import cv2 as cv
import argparse

from Rectangle import *
from IR import *
from Util import *
from Strategies.MagirecoStrategy import *

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--src", type=str, default="src.mp4", help="source video file")
    parser.add_argument("--ass", type=str, default="template.ass", help="source ass template file")
    parser.add_argument("--leftblackbar", type=float, default=0.0, help="width ratio of black bar on the left of canvas, right is assumed symmetric if it is not set")
    parser.add_argument("--rightblackbar", type=float, default=None, help="width ratio of black bar on the right of canvas, assumed symmetric with left if not set")
    parser.add_argument("--topblackbar", type=float, default=0.0, help="height ratio of black bar on the top of canvas, bottom is assumed symmetric if it is not set")
    parser.add_argument("--bottomblackbar", type=float, default=None, help="height ratio of black bar on the bottom of canvas, assumed symmetric with top if not set")
    parser.add_argument("--dst", type=str, default="MagiaTimelineOutput.ass", help="destination ass subtitle file")
    parser.add_argument("--debug", default=False, action="store_true", help="for debugging only, show frames with debug info and save to debug.mp4")
    parser.add_argument("--shortcircuit", default=False, action="store_true", help="accelerates the program by skipping detecting other types of subtitles once one type has been confirmed, not compatible with debug mode")
    args = parser.parse_args()
    if True: # data validity test
        srcMp4Test = open(args.src, "rb")
        srcMp4Test.close()
        if not (args.leftblackbar >= 0.0 and args.leftblackbar <= 1.0):
            raise Exception("Invalid left black bar ratio! ")
        if not (args.topblackbar >= 0.0 and args.topblackbar <= 1.0):
            raise Exception("Invalid top black bar ratio! ")
        if args.rightblackbar is None:
            args.rightblackbar = args.leftblackbar
        if args.bottomblackbar is None:
            args.bottomblackbar = args.topblackbar
        if not (args.rightblackbar >= 0.0 and args.rightblackbar <= 1.0):
            raise Exception("Invalid right black bar ratio! ")
        if not (args.bottomblackbar >= 0.0 and args.bottomblackbar <= 1.0):
            raise Exception("Invalid bottom black bar ratio! ")
        if args.debug and args.shortcircuit:
            raise Exception("Debug mode is not compatible with short circuit mode! ")
    
    srcMp4 = cv.VideoCapture(args.src)
    srcRect = SrcRectangle(srcMp4)
    fps: float = srcMp4.get(cv.CAP_PROP_FPS)
    size: typing.Tuple[int, int] = srcRect.getSizeInt()

    debugMp4: typing.Any = None
    if args.debug:
        debugMp4 = cv.VideoWriter('debug.mp4', cv.VideoWriter_fourcc('m','p','4','v'), fps, size)
    templateAss = open(args.ass, "r")
    dstAss = open(args.dst, "w")
    dstAss.writelines(templateAss.readlines())
    templateAss.close()

    contentRect = RatioRectangle(srcRect, args.leftblackbar, 1 - args.rightblackbar, args.topblackbar, 1 - args.bottomblackbar)

    strategy = MagirecoStrategy(None, contentRect)
    flagIndexType = strategy.getFlagIndexType()

    print("==== FPIR Building ====")
    fpir = FPIR(flagIndexType)
    while True: # Process each frame to build FPIR

        # Frame reading

        frameIndex: int = int(srcMp4.get(cv.CAP_PROP_POS_FRAMES))
        timestamp: int = int(srcMp4.get(cv.CAP_PROP_POS_MSEC))
        validFrame, frame = srcMp4.read()
        if not validFrame:
            break

        # CV and frame point building

        framePoint = FramePoint(flagIndexType, frameIndex, timestamp)
        for cvPass in strategy.getCvPasses():
            mayShortcircuit = cvPass(frame, framePoint)
            if mayShortcircuit and args.shortcircuit:
                break
        fpir.framePoints.append(framePoint)

        # Outputs

        if frameIndex % 1000 == 0:
            print(framePoint.toString())

        if args.debug:
            frameOut = frame
            for name, rect in strategy.getRectangles().items():
                frameOut = rect.draw(frameOut)
            height = 50
            for name, index in flagIndexType.__members__.items():
                value = framePoint.flags[index]
                frameOut = cv.putText(frameOut, name + ": " + str(value), (50, height), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3)
                frameOut = cv.putText(frameOut, name + ": " + str(value), (50, height), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                height += 20
            cv.imshow("show", frameOut)
            if cv.waitKey(1) == ord('q'):
                break
            print("debug frame", frameIndex, formatTimestamp(timestamp), framePoint.getDebugFlag())
            debugMp4.write(frameOut)
    srcMp4.release()
    if args.debug:
        debugMp4.release()

    print("==== FPIR Passes ====")
    for name, fpirPass in strategy.getFpirPasses().items():
        print(name)
        fpirPass.apply(fpir)

    print("==== FPIR to IIR ====")
    iir = IIR(flagIndexType)
    for name, fpirToIirPass in strategy.getFpirToIirPasses().items():
        print(name)
        iir.appendFromFpir(fpir, fpirToIirPass)
    iir.sort()

    print("==== IIR Passes ====")
    for name, iirPass in strategy.getIirPasses().items():
        print(name)
        iirPass.apply(iir)

    print("==== IIR to ASS ====")
    dstAss.write(iir.toAss())
    dstAss.close()

if __name__ == "__main__":
    main()
