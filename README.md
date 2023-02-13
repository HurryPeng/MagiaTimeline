# MagiaTimeline

A CV-based automatic subtitle timeline marking tool for RPGs (Role-Play Games). 

## Introduction

### Purpose

Fans of RPGs would translate game videos when their native language is not supported by the game company. Besides translating, this also requires marking the original subtitles' timeline so that translated text can sync with video contents, which is a time-consuming job. This project provides a framework to automate this process with CV (computer vision) algorithms. You can extend the framework to support more games by implementing a Strategy class that works with the main pipeline. 

### Supported Games

- [Magia Record 「マギアレコード」 《魔法纪录》](https://magireco.com/), for which this project was initially written
- [Limbus Company 「림버스컴퍼니」 《边狱公司》](https://limbuscompany.com/)

### Limitations

Not supporting videos with no subtitle at all. Hard to support videos where subtitles appear at random places (possible if you add some kind of AI to your strategy, but that will be way slower). 

## Getting Started

### Prerequisites

- Python 3.10.9
    - opencv-contrib-python 4.7.0.68

Should also work on other versions, but not tested. 

### Usage

**Working Directory.** Before running any of the following commands, you should change the working directory of you command line to the root folder of this project. 

**Help.** To see full help info, run:

```
python MagiaTimeline.py --help
```

**Strategy.** Specify which Strategy to use (which game). `mr` for Magia Record and `lcb` for Limbus Company. This is by default `mr`. 

```
python MagiaTimeline.py --strategy mr
```

**Source and Destination.** Specify source video file after `--src` and destination subtitle file after `--dst`. If you do not specify them, they will be `src.mp4` and `MagiaTimelineOutput.ass` by default. 

```
python MagiaTimeline.py --src src.mp4 --dst MagiaTimelineOutput.ass
```

**Black Bar Cutting.** If your video has black bars around the canvas, you should specify its ratio to the whole frame. 

```
python MagiaTimeline.py --leftblackbar 0.01 --rightblackbar 0.01 --topblackbar 0.09 --bottomblackbar --0.06
```

**Debug Mode.** It is recommended to run debug mode and adjust parameters before processing the whole video. In debug mode, a real-time window will be displayed where you can preview the cutting of black bars, the alignment of critical sections and intermediate identification results. However, debug mode slows down the program significantly, so you should turn it off after adjustment completes. 

```
python MagiaTimeline.py --debug
```

**Short Circuit Mode.** Short circuit mode accelerates the program by skipping detecting other types of subtitles once one type has been confirmed. This should have no side-effect if different kinds of subtitle guarantee not to appear at the same time. This mode is not compatible with debug mode, which needs to collect all intermediate information. 

```
python MagiaTimeline.py --shortcircuit
```

**Combination of Arguments.** All the arguments above can be combined, unless explicitly forbidden. 

**Without Command Line**. If you have totally no idea how to use a command line to run anything above, you can try running this program by double-clicking it from GUI. Before running, you should check that:

- The input video is renamed to `src.mp4` and put under the same folder as `MagiaTimeline.py`. 
- There is no black bar around the canvas in the input video. 
- The output file will be named `MagiaTimelineOutput.ass` and overwrite any existing file of this name. 

Even though, it is not guaranteed to behave correctly. Please learn how to use a command line. 

**Compressing.** It is recommended to control the resolution of the video within 1280\*720 for performance. You can compress your video with `PyrDown.py`, which halves the width and height of a video by pyramiding down each frame. However, it is more recommended to compress with professional video editors for less compressing time and smaller video size. 

```
python PyrDown.py --src src.mp4 --dst pyrdown.mp4
```

## Architecture

MagiaTimeline's main pipeline adopts a compiler-like architecture. The source video is analyzed and transformed into intermediate representations (IR). Optimization passes are then applied on IRs for better timeline generation. 

For example, with the Strategy of Magia Record, it works in the following steps: 

- Frame-wise computer vision (CV) analysis
    - Detect subtitles in each frame and tag them
        - Dialog
        - Blackscreen
        - Whitescreen
        - CG Sub
- Frame Point Intermediate Representation (FPIR)
    - Each frame is represented by a Frame Point (FP) with attributes
        - Frame index
        - Timestamp
        - Flags
    - Passes
        - Noise filtering
        - Interval building
- Interval Intermediate Representation (IIR)
    - Each subtitle is represented by an Interval with attributes
        - Time range
        - Flags
    - Passes
        - Gap filling
        - ASS formatting

You can also implement another Strategy class to support other games. It needs to follow the CV -> FPIR -> IIR framework, but how the transformations are done depends on your implementation. 

## Acknowledgements

- [水银h2oag](https://space.bilibili.com/246606859), who has been working hard to translate Magia Record videos into Chinese. 
- 冰柠初夏_lemon (QQ: 2624002941), who has been marking timelines for Magia Record videos, testing and advocating this tool. 
- [啊哈哈QAQ](https://space.bilibili.com/2141525), who has also been working on Magia Record videos. 
- 行光 (QQ: 2263221094), who has been leading [都市零协会汉化组](https://space.bilibili.com/1247764479) to translate videos of games from Project Moon. 
