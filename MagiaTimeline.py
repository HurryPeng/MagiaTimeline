import cv2 as cv
import av
import argparse
import json
import jsonschema
import yaml
import fractions
import time
import typing
import traceback
import pathlib

import TouchPaddle # Before anything that imports paddleocr

from Rectangle import *
from IR import *
from Util import *
from Strategies import *
from Engines import *
from ExtraJobs import *

from Version import VERSION

def cli():
    parser = argparse.ArgumentParser(
        description=f"MagiaTimeline {VERSION} - https://github.com/HurryPeng/MagiaTimeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--config", type=str, default="config.yml", help="config file specifying the source and destination files and other parameters")
    parser.add_argument("--schema", type=str, default="ConfigSchema.json", help="schema file for config validation")
    parser.add_argument("--no-pause", action="store_true", help="do not wait for Enter at the end of the run")
    parser.add_argument("--version", action="version", version=VERSION)
    args = parser.parse_args()
    
    with open(args.schema, "r", encoding="utf-8") as f:
        schema = json.load(f)
    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.load(f.read(), Loader=yaml.FullLoader)

    try:
        main(config, schema)
    finally:
        if not args.no_pause:
            input("Press Enter to continue...")

def main(config: dict, schema: dict, tempDirPath: typing.Optional[str] = None):

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
    if not config["engine"] in config:
        raise Exception("No config found for engine \"" + config["engine"] + "\"")
    strategyConfig = config[config["strategy"]][config["preset"]]
    engineConfig = config[config["engine"]]

    cv.ocl.setUseOpenCL(config["enableOpenCL"])

    initDiskCache(tempDirPath)

    for nTask, src in enumerate(config["source"]):
        timeStart = time.time()

        clearDiskCache()

        dst = config["destination"][nTask]
        if dst == "...":
            dst = autoNumberedNaming(src)

        print("")
        print("Task {}: {} -> {}".format(nTask, src, dst))

        srcContainer: av.container.InputContainer = av.open(src, mode='r')
        srcStream: av.video.stream.VideoStream = srcContainer.streams.video[0]
        srcStream.thread_type = 'FRAME'
        originalSize: typing.Tuple[int, int] = (srcStream.codec_context.width, srcStream.codec_context.height)
        size: typing.Tuple[int, int] = originalSize
        maxResWidth: int = config["maxResWidth"]
        maxResHeight: int = config["maxResHeight"]
        scaleDown: int = 1
        while (maxResWidth > 0 and size[0] > maxResWidth) or (maxResHeight > 0 and size[1] > maxResHeight):
            scaleDown *= 2
            size = (size[0] // 2, size[1] // 2)
        timeBase: fractions.Fraction = srcStream.time_base
        fps: fractions.Fraction = srcStream.average_rate

        with open(config["assTemplate"], "r", encoding="utf-8") as templateAsst:
            asstStr: str = templateAsst.read()

        contentRect = RatioRectangle(SrcRectangle(*size), *config["contentRect"])
        print(f"Resolution: {size[0]}x{size[1]}" + (f" (scaled down by {scaleDown})" if scaleDown > 1 else ""))
        print(f"FPS: {float(fps):.2f} ({fps})")

        strategy: AbstractStrategy | None = None
        print("Strategy:", config["strategy"])
        print("Preset:", config["preset"])
        strategy = createStrategy(config["strategy"], strategyConfig, contentRect)
        
        engine: AbstractEngine | None = None
        print("Engine:", config["engine"])
        engine = createEngine(config["engine"], scaleDown, engineConfig)

        print("==== Running Engine ====")
        iir: IIR = engine.checkAndRun(strategy, srcContainer, srcStream)

        timeTimelineEnd = time.time()
        timeTimelineElapsed = timeTimelineEnd - timeStart
        
        print("Timeline Elapsed {:.2f} s".format(timeTimelineElapsed))
        print("Timeline Speed {:.2f}x".format(float(srcStream.frames / fps) / timeTimelineElapsed))

        if config["extraJobs"]:
            if config["engine"] != "speculative":
                print("Error: extra jobs are currently only supported with the speculative engine. Skipping all extra jobs.")
            elif not isinstance(strategy, AbstractExtraJobStrategy):
                print("Error: Strategy does not support extra jobs. Skipping all extra jobs.")
            else:
                for jobName in config["extraJobs"]:
                    if jobName not in config:
                        raise KeyError(f"extraJobs includes '{jobName}' but no '{jobName}' section found in config.")
                    print(f"==== Extra Job: {jobName} ====")
                    job = createExtraJob(jobName, config[jobName], strategy.getExtraJobFrameKey(), dst)
                    job.apply(iir)

        print("==== IIR to ASS ====")
        assStr: str = asstStr.format(
            playResX = originalSize[0],
            playResY = originalSize[1],
            styles = "".join(iir.stylesStr()),
            events = iir.assEventsStr()
        )
        with open(dst + ".ass", "w", encoding="utf-8") as dstAss:
            dstAss.write(assStr)
        print("Result written to", dst + ".ass")

        if config["outputSrt"]:
            print("==== IIR to SRT ====")
            with open(dst + ".srt", "w", encoding="utf-8") as dstSrt:
                dstSrt.write(iir.srtEventStr())
            print("Result written to", dst + ".srt")

        timeOverallEnd = time.time()
        timeOverallElapsed = timeOverallEnd - timeStart
            
        print("Overall Elapsed {:.2f} s".format(timeOverallElapsed))
        print("Overall Speed {:.2f}x".format(float(srcStream.frames / fps) / timeOverallElapsed))

        srcContainer.close()

        # Print disk cache directory size
        getDiskCacheDir = getDiskCache().directory
        cacheDir = pathlib.Path(getDiskCacheDir)
        cacheSize = sum(f.stat().st_size for f in cacheDir.glob('**/*') if f.is_file())
        print(f"Disk cache size: {cacheSize // (1024 * 1024)} MB")

if __name__ == "__main__":
    try:
        cli()
    except Exception as e:
        print("Exception caught: ", e)
        traceback.print_exc()
