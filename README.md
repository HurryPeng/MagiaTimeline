# MagiaTimeline

A CV-based precise subtitle time-coding and alignment tool for RPGs (Role-Playing Game) and Anime Conte (アニメコント、小剧场) videos. 

<img src="./logo/MagiaTimeline-Logo-Transparent.png" width="300">

## Introduction

### Purpose

Fans of RPG and Anime Conte videos often translate content when their native language is not supported by the original creators. Besides translating, syncing the subtitles to match the video's original timeline is also a time-consuming job. This project provides a framework to automate this process using CV (computer vision) algorithms. For supported RPGs and Anime Contes, MagiaTimeline takes video files as inputs and generates `.ass` subtitle files. Each RPG or Anime Conte has its own specialized algorithm set (Strategy), and it is not likely that they will work with unseen RPGs and Anime Contes. However, you can extend the framework by implementing a new Strategy class. 

### Supported RPGs and Anime Contes

- [Magia Record 「マギアレコード」 《魔法纪录》](https://magireco.com/), for which this project was initiated
- [Limbus Company 「림버스컴퍼니」 《边狱公司》](https://limbuscompany.com/)
- [Parako 「私立パラの丸高校」 《超能力高校》](https://www.youtube.com/@parako)
- [BanG Dream! Girls Band Party! 「バンドリ！ ガールズバンドパーティ！」 《BanG Dream! 少女乐团派对!》](https://bang-dream.bushimo.jp/)

### Limitations

Does not support videos with only voice but no subtitles. 

## Getting Started

### Prerequisites

- Python 3.10.9
    - opencv-contrib-python 4.7.0.68
    - PyYAML 6.0
    - jsonschema 4.17.3

Should also work on other versions, but not tested. 

### Workflow

**Working Directory.** Before running any of the following commands, you should change the working directory of your command line to the root folder of this project. 

**Help.** To see help info, run:

```
python MagiaTimeline.py --help
```

**Configuration.** Open `config.yml` to check out configurations for the main program. You can open this file with any text editor, but it is more recommended to open it with an IDE that provides syntax checking according to `ConfigSchema.json`. There are comments in `config.yml` to guide you through. 

**Debug Running.** Run the main program in debug mode to check whether black bars are properly cut and critical areas are aligned to rectangles. 

To run in debug mode, first change `mode` in `config.yml` to `debug`: 

```yaml
mode: debug
```

Then run `MagiaTimeline.py` in the command line: 

```
python MagiaTimeline.py
```

**Productive Running.** Debug mode slows down the program significantly, so after checking the alignments with it, you should process the full video in default mode or short-circuit mode. 

**Using Another Configuration File.** You can copy `config.yml` and modify it as you like. To use another configuration file, tell the main program which file you are using. 

```
python MagiaTimeline.py --config myconfig.yml
```

**Without Command Line.** If you have no idea how to use a command line, you can try running this program by double-clicking it from the GUI. In that case, there is no guarantee that the program would behave correctly. Please learn how to use a command line. 

## Architecture

MagiaTimeline adopts a compiler-like architecture. The source video is frame-by-frame analyzed and transformed into intermediate representations (IR). Optimization passes are then applied to IRs for higher timeline quality. 

For example, it works in the following steps in MagiaRecordStrategy: 

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

You can also implement another Strategy class to support your use case. It needs to follow the CV -> FPIR -> IIR framework, but you can freely specify how the transformations in the middle are done and reuse existing algorithms as much as possible. 

## Acknowledgements

- [水银h2oag](https://space.bilibili.com/246606859), working consistently on translating Magia Record videos into Chinese. 
- 冰柠初夏_lemon (QQ: 2624002941), marking timelines for Magia Record videos, testing and advocating this tool. 
- [啊哈哈QAQ](https://space.bilibili.com/2141525), also working on Magia Record videos. 
- 行光 (QQ: 2263221094), who had been leading [都市零协会汉化组](https://space.bilibili.com/1247764479) to translate games by Project Moon. 
- [灰色渔歌](https://space.bilibili.com/7653809), also working on games by Project Moon. 
- [甜隐君子](https://space.bilibili.com/929197), working consistently on translating Parako into Chinese. 
