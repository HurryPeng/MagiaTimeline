# MagiaTimeline

A CV-based automatic subtitle timeline marking tool for RPGs (Role-Play Games). 

<img src="./logo/MagiaTimeline-Logo-Transparent.png" width="300">

## Introduction

### Purpose

Fans of RPGs would translate game videos when their native language is not supported by the game company. Besides translating, marking the original subtitles' timeline to sync with video contents is also a time-consuming job. This project provides a framework to automate this process with CV (computer vision) algorithms. You can extend the framework by implementing a Strategy class that works with the main pipeline to support more games. 

### Supported Games

- [Magia Record 「マギアレコード」 《魔法纪录》](https://magireco.com/), for which this project was initially written
- [Limbus Company 「림버스컴퍼니」 《边狱公司》](https://limbuscompany.com/)

### Limitations

Not supporting videos with no subtitle at all. Hard to support videos where subtitles appear at random places (possible if you add some kind of AI to your strategy, but that will be way slower). 

## Getting Started

### Prerequisites

- Python 3.10.9
    - opencv-contrib-python 4.7.0.68
    - PyYAML 6.0
    - jsonschema 4.17.3

Should also work on other versions, but not tested. 

### Workflow

**Working Directory.** Before running any of the following commands, you should change the working directory of you command line to the root folder of this project. 

**Help.** To see help info, run:

```
python MagiaTimeline.py --help
python PyrDown.py --help
```

**Compressing.** It is recommended to control the resolution of the video within 1280\*720 for performance. You can compress your video with `PyrDown.py`, which halves the width and height of a video by pyramiding down each frame. However, it is more recommended to compress with professional video editors for less compressing time and smaller video size. 

```
python PyrDown.py --src src.mp4 --dst pyrdown.mp4
```

**Configuration.** Open `config.yml` to check out configurations for the main program. You can open this file with any text editor, but it is more recommended to open it with an IDE that provides syntax checking according to `ConfigSchema.json`. There are comments in `config.yml` to guide you through. 

**Debug Running.** Run the main program in debug mode to check whether black bars are properly cut and critical areas are aligned to rectangles. 

To run in debug mode, first change `mode` in `conig.yml` to `debug`: 

```yaml
mode: debug
```

Then run `MagiaTimeline.py` in command line: 

```
python MagiaTimeline.py
```

**Productive Running.** Debug mode slows down the program significantly, so after checking the alignments with it, you should process the full video in default mode or short circuit mode. 

**Using Another Configuration File.** You can copy `config.yml` and modify it as you like. To use another configuration file, tell the main program which file you are using. 

```
python MagiaTimeline.py --config myconfig.yml
```

**Without Command Line.** If you have totally no idea how to use a command line, you can try running this program by double-clicking it from GUI. Even though, it is not guaranteed to behave correctly. Please learn how to use a command line. 

## Architecture

MagiaTimeline's main pipeline adopts a compiler-like architecture. The source video is analyzed and transformed into intermediate representations (IR). Optimization passes are then applied on IRs for better timeline generation. 

For example, with the Strategy for Magia Record, it works in the following steps: 

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
