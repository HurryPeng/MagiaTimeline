from IR import *
from Strategies.AbstractStrategy import *
from Util import *
from Engines.AbstractEngine import *

import av.container
import av.container.input
import av.video
import av.video.stream
import av.frame
import typing
import fractions

class FramewiseEngine(AbstractEngine):
    def __init__(self, config: dict) -> None:
        self.config: dict = config
        self.sampleInterval: int = config["sampleInterval"]
        self.debug: bool = config["debug"]
        self.debugPyrDown: int = config["debugPyrDown"]
    def getRequiredAbstractStrategyType(self) -> type[AbstractStrategy]:
        return AbstractFramewiseStrategy

    def run(self, strategy: AbstractFramewiseStrategy, container: "av.container.InputContainer", stream: "av.video.stream.VideoStream") -> IIR:
        timeBase: fractions.Fraction = stream.time_base
        fps: fractions.Fraction = stream.average_rate
        frameCount: float = stream.frames

        flagIndexType = strategy.getFlagIndexType()

        debugMp4: typing.Any = None

        print("==== FPIR Building ====")
        fpir = FPIR(flagIndexType, self.sampleInterval)
        frameIndex: int = 0

        for frame in container.decode(stream):
            if frameIndex % self.sampleInterval != 0:
                frameIndex += 1
                continue

            timestamp: int = frame.pts
            img: cv.Mat = frame.to_ndarray(format='bgr24')

            # CV and frame point building
            framePoint = FramePoint(flagIndexType, timestamp)
            for cvPass in strategy.getCvPasses():
                mayShortcircuit = cvPass(img, framePoint)
                # if mayShortcircuit and config["mode"] == "shortcircuit":
                #     break
            fpir.framePoints.append(framePoint)

            # Outputs
            if frameIndex % 720 == 0:
                print(framePoint.toString(timeBase))

            if self.debug:
                frameOut = img
                frameOut = strategy.getContentRect().draw(frameOut)
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
                    debugMp4 = cv.VideoWriter("debug.mp4", cv.VideoWriter_fourcc('m','p','4','v'), float(fps), (frameOut.shape[1], frameOut.shape[0]))
                debugMp4.write(frameOut)
                if self.debugPyrDown > 0:
                    for _ in range(self.debugPyrDown):
                        frameOut = cv.pyrDown(frameOut)
                if cv.waitKey(1) == ord('q'):
                    break
                cv.imshow("show", frameOut)
            framePoint.clearDebugFrame()
            frameIndex += 1

        if self.debug and debugMp4 is not None:
            debugMp4.release()

        print("==== FPIR Passes ====")
        for name, fpirPass in strategy.getFpirPasses().items():
            print(name)
            fpirPass.apply(fpir)

        print("==== FPIR to IIR ====")
        iir = IIR(flagIndexType, fps, timeBase)
        for name, fpirToIirPass in strategy.getFpirToIirPasses().items():
            print(name)
            iir.appendFromFpir(fpir, fpirToIirPass)
        iir.sort()

        print("==== IIR Passes ====")
        for name, iirPass in (strategy.getIirPasses()).items():
            print(name)
            iirPass.apply(iir)

        return iir