import cv2 as cv
import av
import av.container
import av.video
import av.video.stream
import argparse
import json
import jsonschema
import yaml
import pytesseract
import paddleocr
import fractions
import time

from Rectangle import *
from IR import *
from Util import *
from Strategies.MagirecoStrategy import *
from Strategies.MagirecoScene0Strategy import *
from Strategies.LimbusCompanyStrategy import *
from Strategies.LimbusCompanyMechanicsStrategy import *
from Strategies.PokemonEmeraldStrategy import *
from Strategies.ParakoStrategy import *
from Strategies.BanGDreamStrategy import *
from Strategies.OutlineStrategy import *

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

    sampleRate: int = 2

    cv.ocl.setUseOpenCL(config["enableOpenCL"])

    for nTask, src in enumerate(config["source"]):
        timeStart = time.time()

        dst = config["destination"][nTask]

        print("")
        print("Task {}: {} -> {}".format(nTask, src, dst))

        srcContainer: av.container.InputContainer = av.open(src)
        srcStream: av.video.stream.VideoStream = srcContainer.streams.video[0]
        timeBase: fractions.Fraction = srcStream.time_base
        srcStream.thread_type = 'FRAME'
        fps: float = float(srcStream.average_rate)
        frameCount: float = srcStream.frames
        size: typing.Tuple[int, int] = (srcStream.codec_context.width, srcStream.codec_context.height)

        debugMp4: typing.Any = None
        templateAsst = open(config["assTemplate"], "r")
        assStr: str = templateAsst.read()
        templateAsst.close()
        dstAss = open(dst, "w")

        contentRect = RatioRectangle(SrcRectangle(*size), *config["contentRect"])
        offset: int = config["offset"]

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
        elif config["strategy"] == "pkm":
            strategy = PokemonEmeraldStrategy(strategyConfig, contentRect)
        elif config["strategy"] == "prk":
            strategy = ParakoStrategy(strategyConfig, contentRect)
        elif config["strategy"] == "bdr":
            strategy = BanGDreamStrategy(strategyConfig, contentRect)
        elif config["strategy"] == "otl":
            strategy = OutlineStrategy(strategyConfig, contentRect)
        else:
            raise Exception("Unknown strategy! ")
        flagIndexType = strategy.getFlagIndexType()

        print("==== FPIR Building ====")
        fpir = FPIR(flagIndexType, sampleRate)
        frameIndex: int = 0

        for frame in srcContainer.decode(srcStream):
            if frameIndex % sampleRate != 0:
                frameIndex += 1
                continue

            timestamp: int = frame.pts
            img: cv.Mat = frame.to_ndarray(format='bgr24')

            # CV and frame point building
            framePoint = FramePoint(flagIndexType, int(frameIndex // sampleRate), timestamp)
            for cvPass in strategy.getCvPasses():
                mayShortcircuit = cvPass(img, framePoint)
                if mayShortcircuit and config["mode"] == "shortcircuit":
                    doubleBreak = True
                    break
            fpir.framePoints.append(framePoint)

            # Outputs
            if frameIndex % 720 == 0:
                print(framePoint.toString(timeBase, sampleRate))

            if config["mode"] == "debug":
                frameOut = img
                frameOut = contentRect.draw(frameOut)
                for name, rect in strategy.getRectangles().items():
                    frameOut = rect.draw(frameOut)
                height = 50
                for name, index in flagIndexType.__members__.items():
                    value = framePoint.getFlag(index)
                    frameOut = cv.putText(frameOut, name + ": " + str(value), (50, height), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3)
                    frameOut = cv.putText(frameOut, name + ": " + str(value), (50, height), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    height += 20
                print("debug frame", frameIndex, formatTimestamp(timeBase, timestamp), framePoint.getDebugFlag())
                if framePoint.getDebugFrame() is not None:
                    frameOut = ensureMat(framePoint.getDebugFrame())
                    if len(frameOut.shape) == 2:
                        frameOut = cv.cvtColor(frameOut, cv.COLOR_GRAY2BGR)
                if debugMp4 is None:
                    debugMp4 = cv.VideoWriter("debug.mp4", cv.VideoWriter_fourcc('m','p','4','v'), fps, (frameOut.shape[1], frameOut.shape[0]))
                debugMp4.write(frameOut)
                if cv.waitKey(1) == ord('q'):
                    break
                cv.imshow("show", frameOut)
            framePoint.clearDebugFrame()
            frameIndex += 1

        if config["mode"] == "debug" and debugMp4 is not None:
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
        for name, iirPass in (strategy.getIirPasses()).items():
            print(name)
            iirPass.apply(iir)
        print("iirPassOffset")
        IIRPassOffset(offset / fps * 1000).apply(iir)

        print("==== IIR to ASS ====")
        assStr = assStr.format(styles = "".join(strategy.getStyles()), events = iir.toAss(timeBase))

        dstAss.write(assStr)
        dstAss.close()

        timeEnd = time.time()
        timeElapsed = timeEnd - timeStart
        
        print("Elapsed", timeElapsed, "s")
        print("Speed", (frameIndex / fps) / timeElapsed, "x")

        doOcr = True

        if doOcr and isinstance(strategy, OcrStrategy):
            decodedFrameCount = 0
            sampleTimestamps: typing.List[typing.Tuple[str, int]] = iir.getMidpoints()
            paddle = paddleocr.PaddleOCR(use_angle_cls=True, lang='japan', show_log=False) 

            for name, timestamp in sampleTimestamps:
                srcContainer.seek(timestamp, backward=True, any_frame=False, stream=srcStream)
                for frame in srcContainer.decode(srcStream):
                    decodedFrameCount += 1
                    thisTimestamp: int = frame.pts
                    if thisTimestamp < timestamp:
                        continue

                    img: cv.Mat = frame.to_ndarray(format='bgr24')

                    # Tesseract
                    tesseractFrame = strategy.cutCleanOcrFrame(img)
                    tesseractFrame = ensureMat(tesseractFrame)
                    tesseractText: str = pytesseract.image_to_string(tesseractFrame, config="-l jpn --psm 6")
                    tesseractText = tesseractText[:-1].replace("\n", "")

                    # PaddleOCR
                    paddleFrame = strategy.cutOcrFrame(img)
                    paddleResult = paddle.ocr(paddleFrame, cls=False, bin=False)
                    paddleText: str = ""
                    for line in paddleResult:
                        if line is None:
                            continue
                        lineText = "".join([wordInfo[1][0] for wordInfo in line])
                        paddleText += lineText + '\n'
                    paddleText = paddleText.strip()

                    print(f"{name},{tesseractText}@{paddleText}")

                    break

            print("All frame count", frameCount)
            print("Decoded frame count", decodedFrameCount)

        srcContainer.close()


if __name__ == "__main__":
    main()
