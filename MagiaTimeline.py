import cv2 as cv
import av.container
import av.container.input
import av.video
import av.video.stream
import argparse
import json
import jsonschema
import yaml
import fractions
import time
import typing
import traceback

from Rectangle import *
from IR import *
from Util import *
from Strategies.MagirecoStrategy import *
from Strategies.MagirecoScene0Strategy import *
from Strategies.MadodoraStrategy import *
from Strategies.LimbusCompanyStrategy import *
from Strategies.LimbusCompanyMechanicsStrategy import *
from Strategies.PokemonEmeraldStrategy import *
from Strategies.ParakoStrategy import *
from Strategies.BanGDreamStrategy import *
from Strategies.OutlineStrategy import *
from Strategies.BoxColourStatStrategy import *
from Engines.SpeculativeEngine import *
from Engines.FramewiseEngine import *
from ExtraJobs import *

VERSION = "1.0.0"

def main():
    parser = argparse.ArgumentParser(
        description=f"MagiaTimeline {VERSION} - https://github.com/HurryPeng/MagiaTimeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--config", type=str, default="config.yml", help="config file specifying the source and destination files and other parameters")
    parser.add_argument("--schema", type=str, default="ConfigSchema.json", help="schema file for config validation")
    parser.add_argument("--version", action="version", version=VERSION)
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
        if not config["engine"] in config:
            raise Exception("No config found for engine \"" + config["engine"] + "\"")
    strategyConfig = config[config["strategy"]][config["preset"]]
    engineConfig = config[config["engine"]]

    cv.ocl.setUseOpenCL(config["enableOpenCL"])

    for nTask, src in enumerate(config["source"]):
        timeStart = time.time()

        dst = config["destination"][nTask]

        print("")
        print("Task {}: {} -> {}".format(nTask, src, dst))

        srcContainer: "av.container.InputContainer" = av.open(src, mode='r')
        srcStream: "av.video.stream.VideoStream" = srcContainer.streams.video[0]
        srcStream.thread_type = 'FRAME'
        size: typing.Tuple[int, int] = (srcStream.codec_context.width, srcStream.codec_context.height)
        timeBase: fractions.Fraction = srcStream.time_base
        fps: fractions.Fraction = srcStream.average_rate

        templateAsst = open(config["assTemplate"], "r")
        asstStr: str = templateAsst.read()
        templateAsst.close()
        dstAss = open(dst + ".ass", "w")

        contentRect = RatioRectangle(SrcRectangle(*size), *config["contentRect"])

        strategy: AbstractStrategy | None = None
        print("Strategy:", config["strategy"])
        print("Preset:", config["preset"])
        if config["strategy"] == "mr":
            strategy = MagirecoStrategy(strategyConfig, contentRect)
        elif config["strategy"] == "mr-s0":
            strategy = MagirecoScene0Strategy(strategyConfig, contentRect)
        elif config["strategy"] == "md":
            strategy = MadodoraStrategy(strategyConfig, contentRect)
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
        elif config["strategy"] == "bcs":
            strategy = BoxColourStatStrategy(strategyConfig, contentRect)
        else:
            raise Exception("Unknown strategy! ")
        assert strategy is not None
        
        engine: AbstractEngine | None = None
        print("Engine:", config["engine"])
        if config["engine"] == "speculative":
            engine = SpeculativeEngine(engineConfig)
        elif config["engine"] == "framewise":
            engine = FramewiseEngine(engineConfig)
        else:
            raise Exception("Unknown engine! ")
        assert engine is not None

        print("==== Running Engine ====")
        iir: IIR = engine.checkAndRun(strategy, srcContainer, srcStream)

        print("==== IIR to ASS ====")
        asstStr = asstStr.format(styles = "".join(strategy.getStyles()), events = iir.toAss(timeBase))

        dstAss.write(asstStr)
        dstAss.close()

        timeTimelineEnd = time.time()
        timeTimelineElapsed = timeTimelineEnd - timeStart
        
        print("Timeline Elapsed", timeTimelineElapsed, "s")
        print("Timeline Speed", float(srcStream.frames / fps) / timeTimelineElapsed, "x")

        if "ocr" in config["extraJobs"]:
            print("Extra job: ocr")
            if not isinstance(strategy, AbstractOcrStrategy):
                print("Error: Strategy does not support OCR. Skipping. ")
                continue
            iirOcrPass = IIROcrPass(config["ocr"], dst, strategy)
            iirOcrPass.apply(iir)

        timeOverallEnd = time.time()
        timeOverallElapsed = timeOverallEnd - timeStart
            
        print("Overall Elapsed", timeOverallElapsed, "s")
        print("Overall Speed", float(srcStream.frames / fps) / timeOverallElapsed, "x")

        srcContainer.close()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("Exception caught: ", e)
        traceback.print_exc()
    input("Press Enter to continue...")
