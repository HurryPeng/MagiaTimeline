# MagiaTimeline 魔法轴机

[English](./README.md) | 简体中文

基于计算机视觉（CV）的视频打轴软件，自动识别内嵌字幕的出入时间，输出.ass时轴文件。

<img src="./logo/MagiaTimeline-Logo-Transparent.png" width="300">

## 简介

### 开发动机

很多视频里的字幕是内嵌在画面里的，典型场景包括RPG/Galgame录屏、油库里解说（ゆっくり解説）、Voiceroid视频以及动画短剧（アニメコント），翻译这类视频本质上是翻译字幕。专业的字幕组一般会要求做到原字幕和翻译字幕一帧不差地同进同出，为此，就有一个专门的职位叫“时轴”（轴man），专门负责“打轴”，也就是人工盯帧标注每段原字幕的出入时间，导出结构化的时轴文件（.ass/.srt），方便翻译者填写、后期压制。打轴很枯燥很耗时，容易让灵魂积攒污秽，而MagiaTimeline则致力于全世界轴man的解放事业！虽然现在已经有了基于语音的打轴软件，但是因为语音和已有字幕不一定是完全对齐的，所以后期还是需要花很多时间查轴。如果能用MagiaTimeline把打轴的全流程都自动化了，那么很多“一人汉化组”就不用在打轴上消耗自己的翻译热情了。

### 设计目标

- **方便使用**：MagiaTimeline提供通用算法，支持多种视频类型，无需复杂调参
- **无需显卡**：纯CPU即可运行
- **高精准度**：在多数场景下，输出的时间轴可达到与原字幕一帧不差的精度
- **快速打轴**：程序运行时间短于视频时长
- **支持扩展**：允许针对特定类型的视频定制算法，加速打轴、提升精度，不用重写大量框架代码

## 快速入门

### 安装指南

#### 预编译版本（即下即用版）

当前仅支持win_amd64架构（包括大部分Windows电脑），无需安装Python。可以在发布页下载预编译的软件包。

#### 源码安装

1. 安装[Python](https://www.python.org) 3.12.8或更高版本（需<3.13），64位版本
2. 运行`install.bat`（Windows）或`install.sh`（GNU/Linux或macOS），该脚本将自动在Python虚拟环境中安装`requirements.txt`列出的依赖项
3. *（仅需执行`ocr`额外任务时，基础时间轴生成无需此步骤）* 安装Tesseract OCR v5.3.3.20231005或更高版本

### 工作流程

**配置阶段**：用文本编辑器打开`config.yml`查看配置项，配置文件里每一个配置项都有详细注释，建议使用支持语法检查和自动格式化的IDE

新手推荐配置：
- 在`source`部分指定输入视频文件路径
- 设置`strategy`为`dtd`，`preset`为`default`（默认值）
- 设置`engine`为`speculative`（默认值）

**调试运行**：通过调试模式验证字幕区域是否在检测窗口内

临时修改`config.yml`：
- 将`engine`改为`framewise`
- 在`framewise`部分设置`debug`为`true`（默认值）

下一步运行`MagiaTimeline.bat`/`MagiaTimeline.exe`（Windows）或`MagiaTimeline.sh`（GNU/Linux或macOS）。会有一个弹出窗口显示正在处理的帧，底部一个有红色框，请确认字幕出现在框内。如果有超出框的字幕，请调整`dtd`→`default`下的`dialogRect`参数并重新运行。调试模式不用等整个视频打完轴，按`q`键可随时退出检测窗口。

如果视频分辨率过高，调试窗口在屏幕上显示不下，可以调整`framewise`下的`debugPyrDown`参数。

**正式打轴**：调试模式会明显比较慢，检查对齐之后，请关闭调试模式正式打轴。恢复**配置阶段**的参数设置，运行`MagiaTimeline.bat`/`MagiaTimeline.exe`（Windows）或`MagiaTimeline.sh`（GNU/Linux或macOS）。程序运行结束后，文件夹里会出现`MagiaTimelineOutput.ass`。拿去吧你！

## 概念讲解

### 策略组（Strategies）

策略组（Strategy）指的是针对某种类型的视频（也可以是通用策略）的一套计算机视觉算法。你可根据视频类型选择合适的策略组。

MagiaTimeline提供两款推荐的通用策略组：

- **差分文字检测（`dtd`）**: 通过对画面变化部分（差分）执行文本检测从而判断字幕变化的算法。该策略组适用于多数视频且无需调参，首选推荐。
- **颜色统计（`bcs`）**：通过对文本检测框定的区域执行颜色统计从而检测文本变化的算法。该策略组可以作为`dtd`失效时的后备选项。
- **轮廓检测（`otl`）**：依赖特定文本颜色、轮廓颜色和粗细参数的算法。速度显著快于`bcs`，但需要手动调整参数。

除此之外，MagiaTimeline针对几种特定视频类型开发了专用策略组，这几个策略组处理速度快、精度高，但无法泛用：

- [《魔法纪录》「マギアレコード」](https://magireco.com/)（开山鼻祖）
- [Magia Exedra](https://madoka-exedra.com/)
- [《边狱公司》「림버스컴퍼니」](https://limbuscompany.com/)
- [《超能力高校》「私立パラの丸高校」](https://www.youtube.com/@parako)
- [《BanG Dream! 少女乐团派对!》「バンドリ！ガールズバンドパーティ！」](https://bang-dream.bushimo.jp/)

### 处理引擎（Engines）

处理引擎（Engine）决定策略组处理视频时的帧采样方式。MagiaTimeline提供两种引擎：

- **逐帧扫描（Framewise）**：检测每一帧（或者每隔几帧抽一帧），提取信息后连接为字幕区间。该引擎速度较慢，但支持全部策略组。调试模式下可显示实时处理画面，便于调参。
- **二分跳帧（Speculative）**：通过二分法查找字幕变化前后的帧，跳过中间字幕不变的帧。该引擎速度约为逐帧扫描的20倍，但仅支持`dtd`、`bcs`和`otl`策略组。

## 常见问题

### 启用全屏字幕检测

`dtd`和`bcs`策略组默认仅检测屏幕底部四分之一区域，因为屏幕上方可能会有文字干扰字幕检测。如果你的视频字幕会高于下四分之一屏，请打开`config.yml`，在`dtd:`（或`bcs:`）-> `default:`部分找到`dialogRect:`参数，将`0.75`改为`0.00`。

## 架构设计

MagiaTimeline模仿编译器架构，源视频经分析后转换为中间表示（Intermediate Representations, IR），并通过优化阶段（Passes）提升时轴质量。

以**MagiaRecordStrategy**为例，其处理流程包含：

1. **逐帧计算机视觉分析**  
   - 检测每帧字幕并标记为以下类型：
     - 对话字幕（Dialog）
     - 黑屏（Blackscreen）
     - 白屏（Whitescreen）
     - CG字幕（CG Sub）

2. **帧点中间表示（Frame Point IR, FPIR）**  
   - 每帧表示为一个帧点（Frame Point, FP），属性包括：
     - 序号
     - 时间戳
     - 内容类型
   - 优化阶段：
     - 降噪
     - 连成区间

3. **区间中间表示（Interval IR, IIR）**  
   - 每条字幕表示为一个区间（Interval），属性包括：
     - 时间范围
     - 内容类型
   - 优化阶段：
     - 间隙填充（避免闪轴）
     - ASS格式转换

## 致谢

### 早期用户

- [水银h2oag](https://space.bilibili.com/246606859)：《魔法纪录》汉化
- [冰柠初夏_lemon](https://space.bilibili.com/1927412001)：《魔法纪录》打轴、测试并推广本工具
- [啊哈哈QAQ](https://space.bilibili.com/2141525)：《魔法纪录》汉化
- 行光（QQ: 2263221094）：曾带领[都市零协会汉化组](https://space.bilibili.com/1247764479)汉化月亮计划的游戏
- [灰色渔歌](https://space.bilibili.com/7653809)：月亮计划游戏汉化
- [甜隐君子](https://space.bilibili.com/929197)：《超能力高校》汉化

### 临时贡献者

- [Andrew Jeon](https://www.linkedin.com/in/andrew-jeon-58b294107)
- [Wei-Yu (William) Chen](https://www.linkedin.com/in/wei-yu-william-chen)

## 开源协议

本项目基于 GNU 通用公共许可证第三版 (GPLv3) 授权。你可以将本软件用于商业或非商业目的，并可自由复制、修改和分发，但必须遵守 GPLv3 的相关条款，并使用相同的许可证发布衍生代码。若在视频制作中使用本软件，请在简介中注明“自动打轴：MagiaTimeline”。本软件不提供任何形式的担保，包括可用性、正确性等。