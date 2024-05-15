import typing
import enum
import collections

from Util import *
from Strategies.AbstractStrategy import *
from AbstractFlagIndex import *
from Rectangle import *
from IR import *

class PokemonEmeraldStrategy(AbstractStrategy):
    class FlagIndex(AbstractFlagIndex):
        Dialog = enum.auto()
        DialogBg = enum.auto()
        DialogText = enum.auto()

        @classmethod
        def getDefaultFlagsImpl(cls) -> typing.List[typing.Any]:
            return [False, 0.0, 0.0]

    def __init__(self, config: dict, contentRect: AbstractRectangle) -> None:
        self.rectangles: collections.OrderedDict[str, AbstractRectangle] = collections.OrderedDict()
        for k, v in config.items():
            self.rectangles[k] = RatioRectangle(contentRect, *v)

        self.dialogBgRect = self.rectangles["dialogBgRect"]

        self.cvPasses = [self.cvPassDialog]

        self.fpirPasses = collections.OrderedDict()
        self.fpirPasses["fpirPassRemoveNoiseDialogFalse"] = FPIRPassBooleanRemoveNoise(PokemonEmeraldStrategy.FlagIndex.Dialog, False, 1)
        self.fpirPasses["fpirPassRemoveNoiseDialogTrue"] = FPIRPassBooleanRemoveNoise(PokemonEmeraldStrategy.FlagIndex.Dialog, True, 10)

        self.fpirToIirPasses = collections.OrderedDict()
        self.fpirToIirPasses["fpirPassBuildIntervals"] = FPIRPassBooleanBuildIntervals(
            PokemonEmeraldStrategy.FlagIndex.Dialog
        )

        self.iirPasses = collections.OrderedDict()
        self.iirPasses["iirPassFillGapDialog"] = IIRPassFillGap(PokemonEmeraldStrategy.FlagIndex.Dialog, 300)

    @classmethod
    def getFlagIndexType(cls) -> typing.Type[AbstractFlagIndex]:
        return cls.FlagIndex

    def getRectangles(self) -> collections.OrderedDict[str, AbstractRectangle]:
        return self.rectangles

    def getCvPasses(self) -> typing.List[typing.Callable[[cv.Mat, FramePoint], bool]]:
        return self.cvPasses

    def getFpirPasses(self) -> collections.OrderedDict[str, FPIRPass]:
        return self.fpirPasses

    def getFpirToIirPasses(self) -> collections.OrderedDict[str, FPIRPassBuildIntervals]:
        return self.fpirToIirPasses

    def getIirPasses(self) -> collections.OrderedDict[str, IIRPass]:
        return self.iirPasses

    def cvPassDialog(self, frame: cv.Mat, framePoint: FramePoint) -> bool:
        roiDialogBg = self.dialogBgRect.cutRoiToUmat(frame)
        roiDialogBgGray = cv.cvtColor(roiDialogBg, cv.COLOR_BGR2GRAY)
        _, roiDialogBgBin = cv.threshold(roiDialogBgGray, 230, 255, cv.THRESH_BINARY)
        _, roiDialogBgTextBin = cv.threshold(roiDialogBgGray, 128, 255, cv.THRESH_BINARY)
        meanDialogTextBin: float = cv.mean(roiDialogBgTextBin)[0]
        meanDialogBgBin: float = cv.mean(roiDialogBgBin)[0]
        hasDialogBg: bool = meanDialogBgBin > 170
        hasDialogText: bool = meanDialogTextBin < 253 and meanDialogTextBin > 220

        isValidDialog = hasDialogBg and hasDialogText

        framePoint.setFlag(PokemonEmeraldStrategy.FlagIndex.Dialog, isValidDialog)
        framePoint.setFlag(PokemonEmeraldStrategy.FlagIndex.DialogBg, meanDialogBgBin)
        framePoint.setFlag(PokemonEmeraldStrategy.FlagIndex.DialogText, meanDialogTextBin)
        return isValidDialog
