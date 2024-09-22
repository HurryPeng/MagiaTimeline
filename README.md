# MagiaTimeline

A general-purpose CV-based framework for extracting precise subtitle timelines from videos with embedded subtitles, from video to .ass file. 

<img src="./logo/MagiaTimeline-Logo-Transparent.png" width="300">

## Introduction

### Motivation

Many videos have embedded unstructured subtitle texts that are part of the frames. Typical examples include RPC/GalGame recordings, Yukkuri Introduction (ゆっくり解説) videos, Voiceroid videos and Anime Conte (アニメコント) videos. Translating these videos, essentially translating their subtitles, requires the added target language subtitles to sync with the original subtitles. This needs extracting the timepoints when the original subtitles appear, change, and diasppear, into a structured subtitle file (.ass, .srt). Traditionally, this extraction job is done manually, which is a noticable workload in the translation workflow. Voice recognition does not work here because some of the videos do not have dubbings, and even when they do, the dubbings don't necessarily sync with the embedded subtitles well. MagiaTimeline aims to automate this process to empower individual translators, and ultimately advocate cultural communication. 

### Goals

- User-friendliness: A general-purpose algorithm provided by MagiaTimeline supports a variety of videos without the user tuning a lot. 
- Minimum hardware requirement: MagiaTimeline does not require a GPU. 
- Accuracy: The extracted timeline syncs with the original subtitle at a same-frame level accuracy in most cases. 
- Performance: The program takes less time to run than playing the video once. 
- Extendability: MagiaTimeline allows specialized algorithms sets (namely Strategies) that speed up and generate more accurate results for certain types of videos to be integrated into the framework, while reusing most parts of it. 

## Toolbox

### Strategies

A Strategy is a set of CV algorithms that tells the framework how to identify the subtitles from a frame. You can choose which strategy to use for your video. 

MagiaTimeline provides two general-puropse Strategies that are recommended: 

- Box colour stat (`bcs`): A CV pipeline that consists of ML-based text detection, colour statistics and feature extraction. This Strategy works for most videos without tuning. Top recommended.
- Outline (`otl`): A CV pipeline that relies on predefined parameters of colour and weight of text and outline to detect subtitles. It is much faster than `bcs` in applicable cases, but needs manually setting many parameters. 

Traditionally, there are Strategies that are specialized for certain niches of videos. They are very fast and accurate, but not generalizable. 

- [Magia Record 「マギアレコード」 《魔法纪录》](https://magireco.com/), for which this project was initiated
- [Limbus Company 「림버스컴퍼니」 《边狱公司》](https://limbuscompany.com/)
- [Parako 「私立パラの丸高校」 《超能力高校》](https://www.youtube.com/@parako)
- [BanG Dream! Girls Band Party! 「バンドリ！ ガールズバンドパーティ！」 《BanG Dream! 少女乐团派对!》](https://bang-dream.bushimo.jp/)

### Engines

An Engine decides how frames are sampled before they are processed by Strategies. MagiaTimeline provides two Engines:

- Framewise: Processes every frame (or every n-th frame, n configurable) to form a linear string of features before connecting them as subtitle intervals. This Engine is slow but supports all Strategies. In its debug mode, it also provides a visual window that shows the frame currently being processed, which is very useful for debugging and tuning parameters. 
- Speculative: Seeks to process less frames by skipping frames that are likely not to contain subtitle changes. This Engine is about 20x faster than the Framewise Engine. It only supports the `bcl` and `otl` general-purpose strategies. 

## Getting Started

### Installing

1. Install [Python](https://www.python.org) 3.12.0 (or above). 
2. Run `install.bat` (for Windows) or `install.sh` (for GNU/Linux or macOS), which automatically installs dependencies listed in    `requirements.txt` into a Python venv. 
3. (For Extra Job `ocr` only, not required for timeline generation) Tesseract OCR v5.3.3.20231005 (or above). 

### Workflow

**Configuration.** Open `config.yml` to check out configurations. You can open this file with any text editor, but it is more recommended to open it with an IDE that provides syntax checking and pretty formatting. There are detailed comments in `config.yml` to guide you through. 

For beginners, it is recommended to do the following settings: 

- Specify the input video file in `source` section.
- Set `strategy` to `bcs` and `preset` to `default`. This should have been done by default. 
- Set `engine` to `speculative`. This should have been done by default. 

**Debug Running.** Run in debug mode to check whether the subtitle area are included in the detection window. 

Temporarily make these changes to `config.yml`:

- Set `engine` to `framewise`. 
- Under `framewise` section, set `debug` to `true`. This should have been done by default. 

Then, run `MagiaTimeline.bat` (for Windows) or `MagiaTimeline.sh` (for GNU/Linux or macOS). You should see a window popping out that shows the frames from your video and a red box at the bottom of the frame. The red box should cover the area where the subtitles are displayed. If not, you should adjust `dialogRect` under the `bcs`, `default` section and rerun. You don't have to run the program to the end in debug mode. You can quit by typing q in the display window. 

Sometimes the video has too high a resolution that the display window can't show it all. In that case, consider adjusting `debugPyrDown` under the `framewise` section. 

**Productive Running.** Debug mode slows down the program significantly. So, after checking the alignments with it, you should process the full video with debug off. Reset the parameters as instructed in the **Configuration** subsection, and run `MagiaTimeline.bat` (for Windows) or `MagiaTimeline.sh` (for GNU/Linux or macOS). After the program finishes, you should be able to see `MagiaTimelineOutput.ass` generated. As a video maker, you should have been very familiar with this file format. Enjoy.

## Architecture

MagiaTimeline adopts a compiler-like architecture. The source video is analyzed and transformed into intermediate representations (IR). Optimization passes are then applied to IRs for higher timeline quality. 

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

## Acknowledgements

### Early-stage Users

- [水银h2oag](https://space.bilibili.com/246606859), who had been working consistently on translating Magia Record videos into Chinese. 
- [冰柠初夏_lemon](https://space.bilibili.com/1927412001), who had been marking timelines for Magia Record videos, testing and advocating this tool. 
- [啊哈哈QAQ](https://space.bilibili.com/2141525), who had also been working on Magia Record videos. 
- 行光 (QQ: 2263221094), who had been leading [都市零协会汉化组](https://space.bilibili.com/1247764479) to translate games by Project Moon. 
- [灰色渔歌](https://space.bilibili.com/7653809), who is also working on games by Project Moon. 
- [甜隐君子](https://space.bilibili.com/929197), who is working consistently on translating Parako into Chinese. 

### Temporary Collaborators

- [Andrew Jeon](https://www.linkedin.com/in/andrew-jeon-58b294107)
- [Wei-Yu (William) Chen](https://www.linkedin.com/in/wei-yu-william-chen)
