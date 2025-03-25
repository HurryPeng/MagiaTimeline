# MagiaTimeline

English | [简体中文](./README-zh_CN.md)

A general-purpose CV-based framework for extracting precise subtitle timelines from videos with embedded subtitles, from video to .ass file.

<img src="./logo/MagiaTimeline-Logo-Transparent.png" width="300">

## Introduction

### Motivation

Many videos have embedded, unstructured subtitle text that is part of the video frames. Typical examples include RPG/GalGame recordings, Yukkuri Introduction (ゆっくり解説) videos, Voiceroid videos, and Anime Conte (アニメコント) videos. Translating these videos—essentially translating their subtitles—requires the added target-language subtitles to sync with the original subtitles. This necessitates extracting the time points when the original subtitles appear, change, and disappear, into a structured subtitle file (.ass, .srt). Traditionally, this extraction is done manually, which is a significant workload in the translation workflow. Voice recognition does not help because some videos have no voiceovers, and even when they do, the dubbing does not necessarily align with the embedded subtitles. MagiaTimeline aims to automate this process to empower individual translators and ultimately foster cultural communication.

### Goals

- **User-Friendliness**: A general-purpose algorithm provided by MagiaTimeline supports various videos without requiring much user tuning.
- **Minimum Hardware Requirement**: MagiaTimeline does not require a GPU.
- **Accuracy**: The extracted timeline matches the original subtitles with same-frame accuracy in most cases.
- **Performance**: The program takes less time to run than the total duration of the video.
- **Extendability**: MagiaTimeline allows specialized algorithm sets (called Strategies) that speed up processing and generate more accurate results for certain video types, while reusing most of the framework’s core components.

## Getting Started

### Installing

#### Prebuilt Binary

Currently, only win_amd64 is supported. No Python installation is needed. You can find the prebuilt package on the release page.

#### Install from Source

1. Install [Python](https://www.python.org) 3.12.6 (or above, but < 3.13), 64-bit version. 
2. Run `install.bat` (for Windows) or `install.sh` (for GNU/Linux or macOS). This automatically installs dependencies listed in `requirements.txt` into a Python virtual environment.
3. *(For Extra Job `ocr` only, not required for basic timeline generation)* Install Tesseract OCR v5.3.3.20231005 (or above).

### Workflow

**Configuration**: Open `config.yml` to review the configurations. You can open this file with any text editor, but using an IDE that provides syntax checks and formatting is recommended. The detailed comments in `config.yml` will guide you through the setup.

For beginners, the following settings are recommended:

- Specify the input video file in the `source` section.
- Set `strategy` to `bcs` and `preset` to `default`. This should already be the default.
- Set `engine` to `speculative`. This should also be the default.

**Debug Running**: Run in debug mode to check whether the subtitle area is included in the detection window.

Temporarily make these changes to `config.yml`:

- Set `engine` to `framewise`.
- Under the `framewise` section, set `debug` to `true` (this is the default).

Then, run `MagiaTimeline.bat`/`MagiaTimeline.exe` (Windows) or `MagiaTimeline.sh` (GNU/Linux or macOS). A window will pop up showing the frames from your video with a red box at the bottom. The red box should cover the area where the subtitles appear. If it does not, adjust `dialogRect` under the `bcs`, `default` section and rerun. You don’t need to run the entire program in debug mode—quit by typing `q` in the display window whenever you’re done checking.

If the video’s resolution is too high for the display window, consider adjusting `debugPyrDown` under the `framewise` section.

**Productive Running**: Debug mode significantly slows down the program. After verifying alignment in debug mode, run the full video with debug off. Reset the parameters as described in the **Configuration** section, then run `MagiaTimeline.bat`/`MagiaTimeline.exe` (Windows) or `MagiaTimeline.sh` (GNU/Linux or macOS). Once the program finishes, you should see a `MagiaTimelineOutput.ass` file generated. As a video creator, you should already be familiar with this file format. Enjoy!

## Toolbox

### Strategies

A *Strategy* is a set of CV algorithms that tells the framework how to identify subtitles in a frame. You can choose which Strategy to use for your video.

MagiaTimeline provides two recommended general-purpose Strategies:

- **Box colour stat (`bcs`)**: A CV pipeline consisting of ML-based text detection, color statistics, and feature extraction. This strategy works for most videos without tuning and is the top recommendation.
- **Outline (`otl`)**: A CV pipeline that relies on predefined parameters for text and outline color and weight to detect subtitles. It is much faster than `bcs` in applicable cases, but it requires manual parameter adjustments.

Traditionally, there are also Strategies specialized for certain types of videos. These are fast and accurate but not generalizable:

- [Magia Record 「マギアレコード」 《魔法纪录》](https://magireco.com/), for which this project was initially created.
- [Magia Exedra](https://madoka-exedra.com/)
- [Limbus Company 「림버스컴퍼니」 《边狱公司》](https://limbuscompany.com/).
- [Parako 「私立パラの丸高校」 《超能力高校》](https://www.youtube.com/@parako).
- [BanG Dream! Girls Band Party! 「バンドリ！ ガールズバンドパーティ！」 《BanG Dream! 少女乐团派对!》](https://bang-dream.bushimo.jp/).

### Engines

An *Engine* determines how frames are sampled before they are processed by Strategies. MagiaTimeline provides two Engines:

- **Framewise**: Processes every frame (or every n-th frame, where n is configurable) to form a linear string of features before connecting them as subtitle intervals. This engine is slow but supports all Strategies. In debug mode, it provides a visual window showing the currently processed frame, useful for debugging and tuning parameters.
- **Speculative**: Processes fewer frames by skipping those unlikely to contain subtitle changes. This engine is about 20× faster than the Framewise engine but only supports the `bcs` and `otl` general-purpose strategies.

## Troubleshooting

### Enabling Entire-Frame Detection

By default, the `bcs` strategy only detects subtitles in the bottom quarter of the screen (`dialogRect` starts at `0.75`). This is because the upper part of the screen often contains text or other content that may interfere with subtitle detection.

If your video has subtitles above this region and you need full-screen detection, open `config.yml`. Under the `bcs:` → `default:` section, locate the `dialogRect:` parameter and change the `0.75` to `0.00`. This allows subtitle detection across the entire frame.

### Black-and-White Subtitles

If your video features black-and-white (or gray) subtitles, you may find the `bcs` strategy difficult to detect them. MagiaTimeline assigns a lower priority to pure grayscale colors (black/white/gray), which helps reduce false positives but can interfere with accurately picking up purely black-and-white subtitles.
 
To improve black-and-white subtitle detection, open `config.yml` and go to `bcs:` → `default:`. Find `maxGreyscalePenalty: 0.70` and change it to `0.00`. This removes the penalty for grayscale pixels and may significantly improve the recognition of black-and-white subtitles.

## Architecture

MagiaTimeline adopts a compiler-like architecture. The source video is analyzed and transformed into intermediate representations (IR), and optimization passes are applied to these IRs to improve timeline quality.

For example, in the **MagiaRecordStrategy**, the process involves:

1. **Frame-Wise Computer Vision (CV) Analysis**  
   - Detect subtitles in each frame and tag them as one of the following:
     - Dialog
     - Blackscreen
     - Whitescreen
     - CG Sub

2. **Frame Point Intermediate Representation (FPIR)**
   - Each frame is represented by a Frame Point (FP) with attributes:
     - Frame index
     - Timestamp
     - Flags
   - Passes:
     - Noise filtering
     - Interval building

3. **Interval Intermediate Representation (IIR)**
   - Each subtitle is represented by an Interval with attributes:
     - Time range
     - Flags
   - Passes:
     - Gap filling
     - ASS formatting

## Acknowledgements

### Early-Stage Users

- [水银h2oag](https://space.bilibili.com/246606859), who consistently worked on translating Magia Record videos into Chinese.
- [冰柠初夏_lemon](https://space.bilibili.com/1927412001), who marked timelines for Magia Record videos, tested, and promoted this tool.
- [啊哈哈QAQ](https://space.bilibili.com/2141525), who also worked on Magia Record videos.
- 行光 (QQ: 2263221094), who led [都市零协会汉化组](https://space.bilibili.com/1247764479) in translating games by Project Moon.
- [灰色渔歌](https://space.bilibili.com/7653809), who also works on games by Project Moon.
- [甜隐君子](https://space.bilibili.com/929197), who consistently translates Parako into Chinese.

### Temporary Collaborators

- [Andrew Jeon](https://www.linkedin.com/in/andrew-jeon-58b294107)
- [Wei-Yu (William) Chen](https://www.linkedin.com/in/wei-yu-william-chen)

## License

This project is licensed under the GNU General Public License v3.0 (GPLv3). You are free to use, modify, and distribute this software for both commercial and non-commercial purposes, provided you comply with the GPLv3 terms and release any derivative works under the same license. When using this framework in your video production workflow, please mention “Subtitles timed by MagiaTimeline” in the credits or description. This software is provided without any warranty.
