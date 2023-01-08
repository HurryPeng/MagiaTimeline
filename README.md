# MagiaTimeline

Automatic subtitle timeline marking tool for [Magia Record 「マギアレコード」 《魔法纪录》](https://magireco.com/)

## Purpose

Fans from outside Japan are recording videos of Magia Record's quests and translating the subtitles into local languages, in which marking and aligning the subtitles' timeline is a time-consuming job. This tool automatically generates a timeline file by scanning and analyzing such a video with basic CV methods. This is possible because most Magia Record's subtitles are shown at the same place with the same format. Though not all subtitles can be successfully identified, this tool can still save video makers' time. 

## Getting Started

### Prerequisites

- Python 3.10.9
    - opencv-contrib-python 4.7.0.68

Should also work on other versions, but not tested. 

### Usage

**Help.** To see full help info, run:

```
python MagiaTimeline.py --help
```

**Basics.** In most common cases:

```
python MagiaTimeline.py --src ./src.mp4 --dst MagiaTimelineOutput.ass
```

**Black Bar Cutting.** If your video has black bars around the canvas, you should specify its ratio to the whole frame. It is enough to specify top and bottom because the black bars are assumed symmetric.  

```
python MagiaTimeline.py --src ./src.mp4 --dst MagiaTimelineOutput.ass --topblackbar 0.09 --leftblackbar 0.0
```

**Debug Mode.** You can run in debug mode to preview the cutting of black bars and intermediate identification results. It is recommended to do such a quick check before processing the whole video. However, debug mode slows down the program significantly, so you should turn it off after the quick check. 

```
python MagiaTimeline.py --src ./src.mp4 --dst MagiaTimelineOutput.ass --topblackbar 0.09 --leftblackbar 0.0 --debug
```

**Without Command Line**. If you have totally no idea how to use a command line to run anything above, you can still run this program from GUI. Before running, you should check that:

- The input video is renamed to `src.mp4` and put under the same folder as `MagiaTimeline.py`. 
- There is no black bar around the canvas in the input video. 
- The output file will be named `MagiaTimelineOutput.ass` and overwrite any existing file of this name. 

## Acknowledgements

- [水银h2oag](https://space.bilibili.com/246606859), who has been working hard to translate event quests into Chinese. 
