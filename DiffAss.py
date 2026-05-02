from __future__ import annotations
import argparse
import dataclasses
import enum
import pathlib
import re
import typing


@dataclasses.dataclass
class AssEvent:
    index: int
    layer: str
    startCs: int
    endCs: int
    style: str
    text: str
    rawLine: str
    label: str

    def durationCs(self) -> int:
        return self.endCs - self.startCs

    def timeString(self) -> str:
        return f"{formatTimeCs(self.startCs)} -> {formatTimeCs(self.endCs)}"


class ProblemType(enum.Enum):
    Merge = "merge"
    Split = "split"
    Missing = "missing"
    Extra = "extra"


@dataclasses.dataclass
class DiffProblem:
    type: ProblemType
    standardIndexes: typing.List[int]
    trialIndexes: typing.List[int]
    tags: typing.List[str] = dataclasses.field(default_factory=list)
    note: str = ""


@dataclasses.dataclass
class DiffReport:
    standardEvents: typing.List[AssEvent]
    trialEvents: typing.List[AssEvent]
    standardByIndex: typing.Dict[int, AssEvent]
    trialByIndex: typing.Dict[int, AssEvent]
    merges: typing.List[DiffProblem]
    splits: typing.List[DiffProblem]
    missing: typing.List[DiffProblem]
    extra: typing.List[DiffProblem]


def parseAssTime(timeString: str) -> int:
    match = re.match(r"^(\d+):(\d{2}):(\d{2})\.(\d{2})$", timeString.strip())
    if match is None:
        raise ValueError(f"Invalid ASS timestamp: {timeString}")
    hour = int(match.group(1))
    minute = int(match.group(2))
    second = int(match.group(3))
    centisecond = int(match.group(4))
    return hour * 360000 + minute * 6000 + second * 100 + centisecond


def formatTimeCs(timeCs: int) -> str:
    hour = timeCs // 360000
    rem = timeCs % 360000
    minute = rem // 6000
    rem = rem % 6000
    second = rem // 100
    centisecond = rem % 100
    return f"{hour:02d}:{minute:02d}:{second:02d}.{centisecond:02d}"


def parseEventLabel(text: str) -> str:
    if text.startswith("Subtitle_"):
        payload = text[len("Subtitle_"):]
        if "_" in payload:
            return payload.split("_", 1)[0]
        return payload
    return "Unknown"


def parseDialogueLine(line: str, index: int) -> AssEvent | None:
    if not line.startswith("Dialogue:"):
        return None
    parts = line[len("Dialogue:"):].split(",", 9)
    if len(parts) < 10:
        return None
    layer, start, end, style, Name, MarginL, MarginR, MarginV, Effect, text = parts
    text = text.strip()
    return AssEvent(
        index=index,
        layer=layer.strip(),
        startCs=parseAssTime(start),
        endCs=parseAssTime(end),
        style=style.strip(),
        text=text,
        rawLine=line.rstrip("\n"),
        label=parseEventLabel(text),
    )


def loadAssEvents(path: pathlib.Path) -> typing.List[AssEvent]:
    events: typing.List[AssEvent] = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            event = parseDialogueLine(line, len(events))
            if event is not None:
                events.append(event)
    return events


def overlapCs(lhs: AssEvent, rhs: AssEvent) -> int:
    return max(0, min(lhs.endCs, rhs.endCs) - max(lhs.startCs, rhs.startCs))


def gapCs(lhs: AssEvent, rhs: AssEvent) -> int:
    if lhs.endCs < rhs.startCs:
        return rhs.startCs - lhs.endCs
    if rhs.endCs < lhs.startCs:
        return lhs.startCs - rhs.endCs
    return 0


def isMatched(lhs: AssEvent, rhs: AssEvent, toleranceCs: int) -> bool:
    if overlapCs(lhs, rhs) > 0:
        return True
    return abs(lhs.startCs - rhs.startCs) <= toleranceCs and abs(lhs.endCs - rhs.endCs) <= toleranceCs


def collectMatches(
    sourceEvents: typing.List[AssEvent],
    targetEvents: typing.List[AssEvent],
    toleranceCs: int,
) -> typing.Dict[int, typing.List[int]]:
    matches: typing.Dict[int, typing.List[int]] = {}
    for sourceEvent in sourceEvents:
        sourceMatches: typing.List[int] = []
        for targetEvent in targetEvents:
            if isMatched(sourceEvent, targetEvent, toleranceCs):
                sourceMatches.append(targetEvent.index)
        matches[sourceEvent.index] = sourceMatches
    return matches


def buildExtraTags(event: AssEvent, noiseMaxCs: int) -> typing.List[str]:
    tags: typing.List[str] = []
    if event.durationCs() <= noiseMaxCs:
        tags.append("Noise")
    return tags


def analyzeAssDiff(
    standardPath: pathlib.Path,
    trialPath: pathlib.Path,
    toleranceCs: int,
    noiseMaxCs: int,
) -> DiffReport:
    standardAllEvents = loadAssEvents(standardPath)
    trialEvents = loadAssEvents(trialPath)

    standardEvents = standardAllEvents

    standardToTrial = collectMatches(standardEvents, trialEvents, toleranceCs)
    trialToStandard = collectMatches(trialEvents, standardEvents, toleranceCs)

    merges: typing.List[DiffProblem] = []
    splits: typing.List[DiffProblem] = []
    missing: typing.List[DiffProblem] = []
    extra: typing.List[DiffProblem] = []

    for trialEvent in trialEvents:
        standardIndexes = trialToStandard[trialEvent.index]
        if len(standardIndexes) >= 2:
            merges.append(DiffProblem(
                type=ProblemType.Merge,
                standardIndexes=standardIndexes,
                trialIndexes=[trialEvent.index],
            ))
        elif len(standardIndexes) == 0:
            extra.append(DiffProblem(
                type=ProblemType.Extra,
                standardIndexes=[],
                trialIndexes=[trialEvent.index],
                tags=buildExtraTags(trialEvent, noiseMaxCs),
            ))

    for standardEvent in standardEvents:
        trialIndexes = standardToTrial[standardEvent.index]
        if len(trialIndexes) >= 2:
            splits.append(DiffProblem(
                type=ProblemType.Split,
                standardIndexes=[standardEvent.index],
                trialIndexes=trialIndexes,
            ))
        elif len(trialIndexes) == 0:
            missing.append(DiffProblem(
                type=ProblemType.Missing,
                standardIndexes=[standardEvent.index],
                trialIndexes=[],
            ))

    return DiffReport(
        standardEvents=standardEvents,
        trialEvents=trialEvents,
        standardByIndex={event.index: event for event in standardEvents},
        trialByIndex={event.index: event for event in trialEvents},
        merges=merges,
        splits=splits,
        missing=missing,
        extra=extra,
    )


def formatEventRef(event: AssEvent) -> str:
    return f"#{event.index} {event.timeString()}"


def formatIndentedLine(label: str, content: str, indent: str = "  ", labelWidth: int = 10) -> str:
    return f"{indent}{label:<{labelWidth}} {content}"


def formatProblem(
    problem: DiffProblem,
    report: DiffReport,
) -> typing.List[str]:
    lines: typing.List[str] = []
    tagPrefix = ""
    if len(problem.tags) > 0:
        tagPrefix = " (" + ", ".join(problem.tags) + ")"

    if problem.type == ProblemType.Merge:
        trialEvent = report.trialByIndex[problem.trialIndexes[0]]
        lines.append(f"- Merge{tagPrefix}")
        lines.append(formatIndentedLine("Trial", formatEventRef(trialEvent)))
        for standardIndex in problem.standardIndexes:
            standardEvent = report.standardByIndex[standardIndex]
            lines.append(formatIndentedLine("Standard", formatEventRef(standardEvent)))
    elif problem.type == ProblemType.Split:
        standardEvent = report.standardByIndex[problem.standardIndexes[0]]
        lines.append(f"- Split{tagPrefix}")
        lines.append(formatIndentedLine("Standard", formatEventRef(standardEvent)))
        for trialIndex in problem.trialIndexes:
            trialEvent = report.trialByIndex[trialIndex]
            lines.append(formatIndentedLine("Trial", formatEventRef(trialEvent)))
    elif problem.type == ProblemType.Missing:
        standardEvent = report.standardByIndex[problem.standardIndexes[0]]
        lines.append(f"- Missing{tagPrefix}")
        lines.append(formatIndentedLine("Standard", formatEventRef(standardEvent)))
    elif problem.type == ProblemType.Extra:
        trialEvent = report.trialByIndex[problem.trialIndexes[0]]
        lines.append(f"- Extra{tagPrefix}")
        lines.append(formatIndentedLine("Trial", formatEventRef(trialEvent)))

    if problem.note != "":
        lines.append(formatIndentedLine("Note", problem.note))

    return lines


def printProblemSection(
    title: str,
    problems: typing.List[DiffProblem],
    report: DiffReport,
) -> None:
    print("")
    print(f"{title}: {len(problems)}")
    if len(problems) == 0:
        print("  None")
        return
    for problem in problems:
        for line in formatProblem(problem, report):
            print(line)


def printHumanReadableReport(
    report: DiffReport,
    standardPath: pathlib.Path,
    trialPath: pathlib.Path,
    toleranceCs: int,
    noiseMaxCs: int,
) -> None:
    print("ASS Timeline Diff Report")
    print(f"Standard: {standardPath}")
    print(f"Trial: {trialPath}")
    print(f"Match tolerance: {toleranceCs} cs")
    print(f"Noise threshold: {noiseMaxCs} cs")
    print(f"Standard event count: {len(report.standardEvents)}")
    print(f"Trial event count: {len(report.trialEvents)}")

    print("")
    print("Summary")
    print(f"- Merge: {len(report.merges)}")
    print(f"- Split: {len(report.splits)}")
    print(f"- Missing: {len(report.missing)}")
    print(f"- Extra: {len(report.extra)}")

    printProblemSection("Merge Details", report.merges, report)
    printProblemSection("Split Details", report.splits, report)
    printProblemSection("Missing Details", report.missing, report)
    printProblemSection("Extra Details", report.extra, report)


def buildArgumentParser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare two ASS timeline files and classify timeline differences."
    )
    parser.add_argument("standard", type=str, help="Standard ASS file path")
    parser.add_argument("trial", type=str, help="Trial ASS file path")
    parser.add_argument(
        "--toleranceCs",
        type=int,
        default=35,
        help="Matching tolerance in centiseconds for tiny timing drift",
    )
    parser.add_argument(
        "--noiseMaxCs",
        type=int,
        default=20,
        help="Maximum duration in centiseconds for tagging an extra event as Noise",
    )
    return parser


def main() -> None:
    parser = buildArgumentParser()
    args = parser.parse_args()

    standardPath = pathlib.Path(args.standard)
    trialPath = pathlib.Path(args.trial)

    report = analyzeAssDiff(standardPath, trialPath, args.toleranceCs, args.noiseMaxCs)
    printHumanReadableReport(report, standardPath, trialPath, args.toleranceCs, args.noiseMaxCs)


if __name__ == "__main__":
    main()
