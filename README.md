# MagiaTimeline

A general-purpose CV-based framework for extracting precise subtitle timelines from videos with embedded subtitles, from video to .ass file.

<img src="./logo/MagiaTimeline-Logo-Transparent.png" width="300">

## Introduction

### Motivation

Many videos have embedded, unstructured subtitle text that is part of the video frames. Typical examples include RPG/GalGame recordings, Yukkuri Introduction (ゆっくり解説) videos, Voiceroid videos, and Anime Conte (アニメコント) videos. Translating these videos, essentially translating their subtitles, requires the added target language subtitles to sync with the original subtitles. This necessitates extracting the timepoints when the original subtitles appear, change, and disappear, into a structured subtitle file (.ass, .srt). Traditionally, this extraction is done manually, which is a significant workload in the translation workflow. Voice recognition does not work here because some of the videos do not have voiceovers, and even when they do, the dubbing does not necessarily sync with the embedded subtitles. MagiaTimeline aims to automate this process to empower individual translators, and ultimately advocate for cultural communication.

### Goals

- **User-friendliness**: A general-purpose algorithm provided by MagiaTimeline supports a variety of videos without requiring much user tuning.
- **Minimum hardware requirement**: MagiaTimeline does not require a GPU.
- **Accuracy**: The extracted timeline syncs with the original subtitles with same-frame accuracy in most cases.
- **Performance**: The program takes less time to run than it takes to watch the video once.
- **Extendability**: MagiaTimeline allows specialized algorithm sets (called Strategies) that speed up processing and generate more accurate results for certain types of videos, while reusing most of the core components of the framework.

## Toolbox

### Strategies

A Strategy is a set of CV algorithms that tells the framework how to identify the subtitles from a frame. You can choose which strategy to use for your video.

MagiaTimeline provides two general-purpose Strategies that are recommended:

- **Box colour stat (`bcs`)**: A CV pipeline consisting of ML-based text detection, color statistics, and feature extraction. This Strategy works for most videos without tuning and is the top recommendation.
- **Outline (`otl`)**: A CV pipeline that relies on predefined parameters for text and outline color and weight to detect subtitles. It is much faster than `bcs` in applicable cases, but it requires manual parameter adjustments.

Traditionally, there are Strategies specialized for certain niches of videos. They are fast and accurate, but not generalizable:

- [Magia Record 「マギアレコード」 《魔法纪录》](https://magireco.com/), for which this project was initiated.
- [Limbus Company 「림버스컴퍼니」 《边狱公司》](https://limbuscompany.com/).
- [Parako 「私立パラの丸高校」 《超能力高校》](https://www.youtube.com/@parako).
- [BanG Dream! Girls Band Party! 「バンドリ！ ガールズバンドパーティ！」 《BanG Dream! 少女乐团派对!》](https://bang-dream.bushimo.jp/).

### Engines

An Engine decides how frames are sampled before they are processed by Strategies. MagiaTimeline provides two Engines:

- **Framewise**: Processes every frame (or every n-th frame, n configurable) to form a linear string of features before connecting them as subtitle intervals. This Engine is slow but supports all Strategies. In its debug mode, it also provides a visual window showing the currently processed frame, useful for debugging and tuning parameters.
- **Speculative**: Seeks to process fewer frames by skipping frames unlikely to contain subtitle changes. This Engine is about 20x faster than the Framewise Engine but only supports the `bcs` and `otl` general-purpose strategies.

## Getting Started

### Installing

1. Install [Python](https://www.python.org) 3.12.0 (or above).
2. Run `install.bat` (for Windows) or `install.sh` (for GNU/Linux or macOS), which automatically installs dependencies listed in `requirements.txt` into a Python virtual environment.
3. (For Extra Job `ocr` only, not required for timeline generation) Install Tesseract OCR v5.3.3.20231005 (or above).

### Workflow

**Configuration**: Open `config.yml` to review the configurations. You can open this file with any text editor, but using an IDE that provides syntax checking and pretty formatting is recommended. Detailed comments in `config.yml` will guide you through the setup.

For beginners, the following settings are recommended:

- Specify the input video file in the `source` section.
- Set `strategy` to `bcs` and `preset` to `default`. This should be the default setting.
- Set `engine` to `speculative`. This should be the default setting.

**Debug Running**: Run in debug mode to check whether the subtitle area is included in the detection window.

Temporarily make these changes to `config.yml`:

- Set `engine` to `framewise`.
- Under the `framewise` section, set `debug` to `true`. This should be the default setting.

Then, run `MagiaTimeline.bat` (for Windows) or `MagiaTimeline.sh` (for GNU/Linux or macOS). A window will pop up showing the frames from your video with a red box at the bottom. The red box should cover the area where the subtitles are displayed. If not, adjust `dialogRect` under the `bcs`, `default` section and rerun. You don’t have to run the program to the end in debug mode; you can quit by typing 'q' in the display window.

If the video has a resolution too high for the display window, consider adjusting `debugPyrDown` under the `framewise` section.

**Productive Running**: Debug mode significantly slows down the program. After verifying alignments in debug mode, process the full video with debug off. Reset the parameters as instructed in the **Configuration** section and run `MagiaTimeline.bat` (for Windows) or `MagiaTimeline.sh` (for GNU/Linux or macOS). Once the program finishes, you should see a `MagiaTimelineOutput.ass` file generated. As a video maker, you should already be familiar with this file format. Enjoy!

## Architecture

MagiaTimeline adopts a compiler-like architecture. The source video is analyzed and transformed into intermediate representations (IR). Optimization passes are applied to IRs for higher timeline quality.

For example, in the **MagiaRecordStrategy**, it works in the following steps:

1. **Frame-wise computer vision (CV) analysis**:
    - Detect subtitles in each frame and tag them as:
        - Dialog
        - Blackscreen
        - Whitescreen
        - CG Sub
2. **Frame Point Intermediate Representation (FPIR)**:
    - Each frame is represented by a Frame Point (FP) with attributes:
        - Frame index
        - Timestamp
        - Flags
    - Passes:
        - Noise filtering
        - Interval building
3. **Interval Intermediate Representation (IIR)**:
    - Each subtitle is represented by an Interval with attributes:
        - Time range
        - Flags
    - Passes:
        - Gap filling
        - ASS formatting

## Acknowledgements

### Early-stage Users

- [水银h2oag](https://space.bilibili.com/246606859), who consistently worked on translating Magia Record videos into Chinese.
- [冰柠初夏_lemon](https://space.bilibili.com/1927412001), who marked timelines for Magia Record videos, tested, and advocated for this tool.
- [啊哈哈QAQ](https://space.bilibili.com/2141525), who also worked on Magia Record videos.
- 行光 (QQ: 2263221094), who led [都市零协会汉化组](https://space.bilibili.com/1247764479) in translating games by Project Moon.
- [灰色渔歌](https://space.bilibili.com/7653809), who is also working on games by Project Moon.
- [甜隐君子](https://space.bilibili.com/929197), who is consistently translating Parako into Chinese.

### Temporary Collaborators

- [Andrew Jeon](https://www.linkedin.com/in/andrew-jeon-58b294107)
- [Wei-Yu (William) Chen](https://www.linkedin.com/in/wei-yu-william-chen)
