# Video Subtitle Generator | 视频字幕生成器

[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![DeepSeek](https://img.shields.io/badge/API-DeepSeek-orange)](https://deepseek.com)
[![Distil-Whisper](https://img.shields.io/badge/Model-Distil--Whisper-yellow)](https://huggingface.co/distil-whisper/distil-large-v2)

基于 Distil-Whisper 和 DeepSeek API 的智能视频字幕生成器，支持中英双语字幕，具备上下文感知的智能纠错和专业术语映射功能。

[English](README_en.md) | 简体中文

## 🌟 特色功能

- 支持多种视频格式（MP4, MOV, AVI, MKV）
- 使用 Distil-Whisper 生成准确的英文字幕，支持句子级别的时间戳
- 智能分段处理：
  - 基于语义和标点符号的智能分段
  - 自动合并过短的片段
  - 动态分割过长的段落（超过 20 个单词或 8 秒）
  - 保持句子的完整性和上下文连贯性
- 硬件加速支持：
  - NVIDIA GPU：使用 CUDA 加速
  - Apple Silicon：使用 MPS (Metal Performance Shaders) 加速
  - 其他设备：使用 CPU 处理
- 使用 DeepSeek API 进行上下文感知的智能纠错：
  - 特别处理发音相似词的混淆问题
  - 准确识别技术术语和专有名词
  - 基于上下文的语义理解和修正
  - 自动移除重复内容
- 专业术语映射：
  - 维护常用专业术语的中英文对照表
  - 确保翻译的一致性和专业性
  - 支持自定义术语映射
- 生成标准 SRT 格式的双语字幕文件
- 支持批量处理多个视频文件
- 支持两种输入方式：
  - 文件上传：通过界面上传视频文件
  - 本地路径：直接输入本地视频文件路径
- 字幕文件自动保存到原视频所在目录，保持相同文件名
- 智能缓存系统：
  - 缓存语音识别结果，避免重复处理
  - 使用 MD5 哈希确保缓存的唯一性
  - 支持断点续传

## 工作原理

1. **音频提取**：
   - 从视频中提取音频轨道
   - 自动处理多种视频格式
   - 使用临时文件避免磁盘占用

2. **语音识别**：
   - 使用 Distil-Whisper 模型进行语音识别
   - 生成句子级别的时间戳
   - 支持硬件加速（CUDA/MPS）
   - 结果自动缓存，提高效率

3. **智能分段**：
   - 基于语义和标点的多级分段策略
   - 动态调整段落长度
   - 确保字幕显示时长合理
   - 保持句子的完整性

4. **智能纠错**：
   - 收集完整的字幕文本
   - 分析每个句子的上下文
   - 使用 DeepSeek API 进行深度纠错
   - 特别关注技术术语和专有名词
   - 自动移除重复内容

5. **中文翻译**：
   - 使用 DeepSeek API 进行翻译
   - 应用专业术语映射
   - 确保翻译的一致性
   - 保持专业术语的准确性

6. **字幕生成**：
   - 生成标准 SRT 格式
   - 中英文对照显示
   - 合理的时间戳分配
   - 自动保存到视频目录

## 安装要求

1. Python 3.10 或更高版本
2. CUDA 支持（推荐，但不是必需）
3. DeepSeek API 密钥

## 安装步骤

1. 克隆仓库：
```bash
git clone https://github.com/wayhome/video-subtitle-generator.git
cd video-subtitle-generator
```

2. 安装 uv：
```bash
pip install uv
```

3. 使用 uv 安装依赖：
```bash
uv sync
```

## 环境变量配置

1. 复制环境变量示例文件：
```bash
cp .env.example .env
```

2. 编辑 `.env` 文件，设置必要的环境变量：
```env
# 必需设置
DEEPSEEK_API_KEY=your_api_key_here

# 可选设置
WHISPER_MODEL_ID=distil-whisper/distil-large-v2
DEEPSEEK_MODEL_ID=deepseek-chat
DEEPSEEK_BASE_URL=https://api.deepseek.com/v1
```

注意：如果没有在 `.env` 文件中设置 `DEEPSEEK_API_KEY`，您也可以在应用运行时通过界面输入。

## 使用方法

1. 启动应用：
```bash
uv run run.py
```

2. 在浏览器中打开显示的地址（通常是 http://localhost:8501）

3. 如果还没有设置 API 密钥，在侧边栏中输入您的 DeepSeek API 密钥

4. 选择输入方式：
   
   a. 上传文件：
   - 点击"选择视频文件"按钮
   - 可以选择一个或多个视频文件（支持 MP4, MOV, AVI, MKV 格式）
   - 点击"开始处理"按钮
   - 处理完成后可以下载生成的字幕文件
   
   b. 本地文件路径：
   - 在文本框中输入本地视频文件的完整路径
   - 支持多个路径，每行一个
   - 点击"开始处理"按钮
   - 字幕文件会自动保存到视频所在目录

## 字幕分段策略

系统采用多级分段策略，确保字幕的可读性和观看体验：

1. **主要分段**：
   - 基于句号、问号、感叹号等主要标点
   - 保持句子的完整性和语义连贯性

2. **次要分段**：
   - 当段落过长时，使用逗号、分号等次要标点分割
   - 确保每段字幕不超过 20 个单词

3. **强制分段**：
   - 当没有合适的分割点时，按单词数强制分割
   - 保证每段字幕的显示时长不超过 8 秒

4. **智能合并**：
   - 自动合并过短的片段（少于 3 个单词）
   - 确保合并后不会影响可读性

## 注意事项

- 首次运行时会下载必要的模型文件，可能需要一些时间
- 处理时间取决于视频长度和系统性能
- 硬件加速会自动选择最优方案：
  - NVIDIA GPU：使用 CUDA 加速
  - Apple Silicon (M1/M2)：使用 MPS 加速
  - 其他设备：使用 CPU 处理
- 生成的 SRT 文件可以直接用于大多数视频播放器
- 请确保您有足够的 DeepSeek API 配额
- 使用本地文件路径时，请确保对视频所在目录有写入权限
- 缓存文件保存在 `cache` 目录下，可以手动清理

## 技术栈

- Streamlit：Web 界面框架
- Distil-Whisper：语音识别
- DeepSeek API：智能纠错和机器翻译
- moviepy：视频处理
- srt：字幕文件处理
- python-dotenv：环境变量管理
- uv：Python 包管理器

## 字幕纠错示例

系统能够智能处理各种场景：

1. 技术术语：
```
原始识别：github commet hash and pull reques
上下文纠错：GitHub commit hash and pull request
```

2. 专业术语：
```
原始识别：beta nutral and dollar nutral strategy
上下文纠错：beta neutral and dollar neutral strategy
翻译结果：贝塔中性和美元中性策略
```

3. 重复内容处理：
```
原始文本：Let me explain the concept. The concept is about...
处理后：Let me explain the concept. It is about...
```

## 许可证

MIT License
