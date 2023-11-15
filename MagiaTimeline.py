from __future__ import annotations
import typing
import cv2 as cv
import argparse
import json
import jsonschema
import yaml
import time

from Rectangle import *
from IR import *
from Util import *
from Strategies.MagirecoStrategy import *
from Strategies.MagirecoScene0Strategy import *
from Strategies.LimbusCompanyStrategy import *

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--config", type=str, default="config.yml", help="config file stating parameters to run with")
    parser.add_argument("--schema", type=str, default="ConfigSchema.json", help="schema file specifying the format of config file")
    args = parser.parse_args()
    
    schema = json.load(open(args.schema, "r"))
    config = yaml.load(open(args.config, "r").read(), Loader=yaml.FullLoader)
    if True: # config validation
        jsonschema.validate(config, schema=schema) # raises exception on failure
        if len(config["source"]) != len(config["destination"]):
            raise Exception("Source and destination have different length")
        for src in config["source"]:
            srcMp4Test = open(src, "rb") # raises exception on failure
            srcMp4Test.close()
        if not config["strategy"] in config:
            raise Exception("No config found for strategy \"" + config["strategy"] + "\"")
        if not config["preset"] in config[config["strategy"]]:
            raise Exception("No preset \"" + config["preset"] + "\" found for strategy \"" + config["strategy"] + "\"")
    strategyConfig = config[config["strategy"]][config["preset"]]

    cv.ocl.setUseOpenCL(config["enableOpenCL"])

    for nTask, src in enumerate(config["source"]):
        timeStart = time.time()

        dst = config["destination"][nTask]

        print("")
        print("Task {}: {} -> {}".format(nTask, src, dst))

        srcMp4 = cv.VideoCapture(src)
        srcRect = SrcRectangle(srcMp4)
        fps: float = srcMp4.get(cv.CAP_PROP_FPS)
        frameCount: float = srcMp4.get(cv.CAP_PROP_FRAME_COUNT)
        size: typing.Tuple[int, int] = srcRect.getSizeInt()

        debugMp4: typing.Any = None
        if config["mode"] == "debug":
            debugMp4 = cv.VideoWriter("debug.mp4", cv.VideoWriter_fourcc('m','p','4','v'), fps, size)
        templateAsst = open(config["assTemplate"], "r")
        assStr: str = templateAsst.read()
        templateAsst.close()
        dstAss = open(dst, "w")

        contentRect = RatioRectangle(srcRect, *config["contentRect"])

        strategy: AbstractStrategy | None = None
        print(config["strategy"])
        if config["strategy"] == "mr":
            strategy = MagirecoStrategy(strategyConfig, contentRect)
        elif config["strategy"] == "mr-s0":
            strategy = MagirecoScene0Strategy(strategyConfig, contentRect)
        elif config["strategy"] == "lcb":
            strategy = LimbusCompanyStrategy(strategyConfig, contentRect)
        elif config["strategy"] == "lcb-mech":
            strategy = LimbusCompanyMechanicsStrategy(strategyConfig, contentRect)
        else:
            raise Exception("Unknown strategy! ")
        flagIndexType = strategy.getFlagIndexType()

        print("==== FPIR Building ====")
        fpir = FPIR(flagIndexType)
        while True: # Process each frame to build FPIR

            # Frame reading

            frameIndex: int = int(srcMp4.get(cv.CAP_PROP_POS_FRAMES))
            timestamp: int = int(srcMp4.get(cv.CAP_PROP_POS_MSEC))
            validFrame, frame = srcMp4.read()
            frame = cv.UMat(frame)
            if not validFrame:
                break

            # CV and frame point building

            framePoint = FramePoint(flagIndexType, frameIndex, timestamp)
            for cvPass in strategy.getCvPasses():
                mayShortcircuit = cvPass(frame, framePoint)
                if mayShortcircuit and config["mode"] == "shortcircuit":
                    break
            fpir.framePoints.append(framePoint)

            # Outputs

            if frameIndex % 1000 == 0:
                print(framePoint.toString())

            if config["mode"] == "debug":
                frameOut = frame
                frameOut = contentRect.draw(frameOut)
                for name, rect in strategy.getRectangles().items():
                    frameOut = rect.draw(frameOut)
                height = 50
                for name, index in flagIndexType.__members__.items():
                    value = framePoint.getFlag(index)
                    frameOut = cv.putText(frameOut, name + ": " + str(value), (50, height), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3)
                    frameOut = cv.putText(frameOut, name + ": " + str(value), (50, height), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    height += 20
                print("debug frame", frameIndex, formatTimestamp(timestamp), framePoint.getDebugFlag())
                debugMp4.write(frameOut)
                if framePoint.getDebugFrame() is not None:
                    frameOut = framePoint.getDebugFrame()
                    framePoint.clearDebugFrame()
                if cv.waitKey(1) == ord('q'):
                    break
                cv.imshow("show", frameOut)
        srcMp4.release()
        if config["mode"] == "debug":
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
        assStr = assStr.format(styles = "".join(strategy.getStyles()), events = iir.toAss())

        dstAss.write(assStr)
        dstAss.close()

        timeEnd = time.time()
        timeElapsed = timeEnd - timeStart
        
        print("Elapsed", timeElapsed, "s")
        print("Speed", (frameCount / fps) / timeElapsed, "x")

if __name__ == "__main__":
    main()
