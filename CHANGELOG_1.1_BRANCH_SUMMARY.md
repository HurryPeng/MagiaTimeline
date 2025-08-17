# MagiaTimeline 1.1分支代码改动总结

## 概述

此文档总结了MagiaTimeline 1.1分支自1.1.0-beta.3版本发布（commit e63289c）以来到最新commit（0392afb）的所有代码改动。总共涵盖了约50个commit，包含多个重要功能改进和修复。

## 主要改动分类

### 1. 🤖 DTD (Deep Text Detection) 算法优化

DTD（深度文本检测）是项目的核心算法，本期间收到了大量改进：

**准确性提升：**
- 添加了sobelLineAngleDiffMask以提供更精确的文本边缘对比
- 改进了纹理背景上分段字幕行的处理
- 增加了预处理和后处理步骤以提高准确性
- 限制了phaseCorrelate使用的分辨率以提高性能
- 性能调优和调试输出更新

**算法改进：**
- 在Sobel算子上实现ECC（Enhanced Correlation Coefficient）
- 实验性SSIM（结构相似性指数）实现
- 调整ECC的最小偏移阈值
- 为了性能考虑，恢复了sobelLineAngleDiffMask

### 2. 🖥️ GUI界面改进

用户界面收到了多项重要更新：

**新功能：**
- 添加了进度条显示处理进度
- 添加了OCR复选框选项
- 显示dialogRect调试信息
- 改进了成功退出消息
- 增加了崩溃时的错误消息打印

### 3. 📝 OCR功能升级

OCR（光学字符识别）功能得到重大升级：

**版本升级：**
- PaddleOCR从2.x升级到3.0版本
- 支持PaddleX 3.0
- 显式启用MKLDNN加速

**功能增强：**
- 添加了角度过滤功能
- 增加了非主要抑制功能
- 限制了发送到OCR的分辨率以提高性能
- 改进了autoNumberedNaming功能

### 4. ⚡ SpeculativeEngine修复

推测性引擎收到重要修复：

**关键修复：**
- 修复了获取最后一帧失败的问题
- 修复了proposal不前进的问题
- 改进了准确性
- 以人类可读时间打印详细信息

### 5. 🛠️ 工具和实用功能改进

**线程池优化：**
- 从工作队列切换到线程池
- 主线程退出时等待线程池完成
- 异步写入磁盘缓存以避免I/O阻塞主线程

**缓存优化：**
- LZ4压缩级别改为3
- 打印磁盘缓存大小
- 改进了磁盘缓存管理

### 6. 🏗️ 构建和项目配置

**版本管理：**
- 版本号提升到1.1.0-beta.4
- 独立版本号文件管理
- 添加了TouchPaddle.py确保模型下载

**依赖更新：**
- Paddle升级到3.1版本
- 更新了PaddleX 3.0的隐藏依赖
- 设置PADDLE_PDX_CACHE_HOME环境变量

**文档更新：**
- 更新了README文件
- 改进了项目文档

### 7. 🎯 架构和API改进

**核心架构：**
- FramePoint和FPIR现在带有timeBase
- 从必需策略API中移除了aggregateAndMoveFeatureToIntervalOnHook()
- 泛化cutOcrFrame为cutExtraJobFrame

**输出改进：**
- 输出文件自动命名
- 将分辨率写入目标ASS文件
- 限制默认分辨率为2K
- 调整ASS模板

### 8. 🔧 配置和参数优化

**参数调整：**
- 更新了默认参数
- 修复了debugLevel拼写错误
- 修复了未导入shutil的问题

## 详细Commit列表

以下是按时间顺序排列的所有commit：

1. `67da3c5` - util: switched from work queue to thread pool
2. `44daa43` - util: wait for thread pool to finish on main thread exit  
3. `d86567b` - IR: FramePoint and FPIR now brings timeBase
4. `2bf3743` - dtd: fix producing segmented subtitle lines on textured backgrounds
5. `25a67e1` - dtd: fix not importing shutil
6. `d6041f8` - dtd: fix debugLevel spelling
7. `d0f9351` - main: limit resolution to 2k by default
8. `28d48a7` - speculative: print verbose info in human readable time
9. `ea812ef` - dtd: add debug info
10. `88cce53` - speculative: accuracy improvements
11. `376a788` - ocr: add non-major suppression
12. `0d7aa32` - config: update default params
13. `14f4714` - ocr: add angle filter
14. `85625d0` - build: upgrade paddle to 3.1
15. `678d643` - dtd: accuracy enhancements with pre- and post-inpaint processings
16. `e559d71` - dtd: perform ECC on Sobel
17. `e020eba` - dtd: experimental ssim
18. `d31cd0b` - dtd: add sobelLineAngleDiffMask for more accurate text edge comparison
19. `2b49fad` - dtd: reverted sobelLineAngleDiffMask for performance
20. `734ef9a` - gui: show dialogRect
21. `ccbe767` - dtd: performance tuning
22. `a11fc9e` - dtd: update debug output
23. `8d6a4f1` - proj: bump version to 1.1.0-beta.4
24. `d25215c` - util: remove "-test" in autoNumberedNaming
25. `1d469de` - speculative: fix failing to retrieve the last frame and proposal not making progress
26. `138d3a4` - gui: print error message on crash
27. `0392afb` - gui: change successful exit message

## 总结

这个版本周期主要专注于以下几个方面：

1. **算法准确性**：大量工作投入到DTD算法的改进，特别是文本检测的准确性
2. **性能优化**：通过线程池、缓存优化和分辨率限制提高处理速度
3. **用户体验**：GUI改进包括进度条和更好的错误处理
4. **现代化升级**：OCR库升级到最新版本，依赖项更新
5. **稳定性修复**：解决了多个关键bug，特别是SpeculativeEngine相关问题

这些改动显著提升了MagiaTimeline的功能性、准确性和用户体验，为1.1版本的正式发布奠定了坚实基础。