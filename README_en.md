# Video Subtitle Generator

[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![DeepSeek](https://img.shields.io/badge/API-DeepSeek-orange)](https://deepseek.com)
[![Distil-Whisper](https://img.shields.io/badge/Model-Distil--Whisper-yellow)](https://huggingface.co/distil-whisper/distil-large-v2)

An intelligent video subtitle generator based on Distil-Whisper and DeepSeek API, supporting bilingual subtitles (English & Chinese) with context-aware error correction and professional terminology mapping.

English | [简体中文](README.md)

## ✨ Features

- Support for multiple video formats (MP4, MOV, AVI, MKV)
- Generate accurate English subtitles using Distil-Whisper with sentence-level timestamps
- Intelligent segmentation:
  - Smart segmentation based on semantics and punctuation
  - Automatic merging of short segments
  - Dynamic splitting of long paragraphs (over 20 words or 8 seconds)
  - Maintain sentence integrity and context coherence
- Hardware acceleration support:
  - NVIDIA GPU: CUDA acceleration
  - Apple Silicon: MPS (Metal Performance Shaders) acceleration
  - Other devices: CPU processing
- Context-aware error correction using DeepSeek API:
  - Special handling of phonetically similar word confusion
  - Accurate recognition of technical terms and proper nouns
  - Context-based semantic understanding and correction
  - Automatic removal of duplicate content
- Professional terminology mapping:
  - Maintain English-Chinese mapping for common professional terms
  - Ensure translation consistency and professionalism
  - Support for custom terminology mapping
- Generate standard SRT format bilingual subtitle files
- Support batch processing of multiple video files
- Two input methods:
  - File upload: Upload video files through the interface
  - Local path: Direct input of local video file paths
- Subtitle files automatically saved to the original video directory
- Intelligent caching system:
  - Cache speech recognition results to avoid reprocessing
  - Use MD5 hash to ensure cache uniqueness
  - Support for breakpoint resumption

## How It Works

1. **Audio Extraction**:
   - Extract audio track from video
   - Automatic handling of multiple video formats
   - Use temporary files to avoid disk usage

2. **Speech Recognition**:
   - Use Distil-Whisper model for speech recognition
   - Generate sentence-level timestamps
   - Support hardware acceleration (CUDA/MPS)
   - Automatic result caching for efficiency

3. **Intelligent Segmentation**:
   - Multi-level segmentation strategy based on semantics and punctuation
   - Dynamic paragraph length adjustment
   - Ensure reasonable subtitle display duration
   - Maintain sentence integrity

4. **Error Correction**:
   - Collect complete subtitle text
   - Analyze context for each sentence
   - Deep correction using DeepSeek API
   - Special focus on technical terms and proper nouns
   - Automatic duplicate content removal

5. **Chinese Translation**:
   - Translation using DeepSeek API
   - Apply professional terminology mapping
   - Ensure translation consistency
   - Maintain accuracy of professional terms

6. **Subtitle Generation**:
   - Generate standard SRT format
   - Bilingual display
   - Reasonable timestamp allocation
   - Automatic save to video directory

## Requirements

1. Python 3.10 or higher
2. CUDA support (recommended but not required)
3. DeepSeek API key

## Installation

1. Clone the repository:
```bash
git clone https://github.com/wayhome/video-subtitle-generator.git
cd video-subtitle-generator
```

2. Install uv:
```bash
pip install uv
```

3. Install dependencies using uv:
```bash
uv sync
```

## Environment Variables

1. Copy the environment variable example file:
```bash
cp .env.example .env
```

2. Edit the `.env` file and set the necessary environment variables:
```env
# Required
DEEPSEEK_API_KEY=your_api_key_here

# Optional
WHISPER_MODEL_ID=distil-whisper/distil-large-v2
DEEPSEEK_MODEL_ID=deepseek-chat
DEEPSEEK_BASE_URL=https://api.deepseek.com/v1
```

Note: If you haven't set `DEEPSEEK_API_KEY` in the `.env` file, you can input it through the interface when running the application.

## Usage

1. Start the application:
```bash
uv run run.py
```

2. Open the displayed address in your browser (usually http://localhost:8501)

3. If you haven't set the API key, enter your DeepSeek API key in the sidebar

4. Choose input method:
   
   a. File upload:
   - Click the "Choose video files" button
   - Select one or more video files (supports MP4, MOV, AVI, MKV formats)
   - Click "Start processing"
   - Download the generated subtitle file after processing
   
   b. Local file path:
   - Enter the full path of local video files in the text box
   - Support multiple paths, one per line
   - Click "Start processing"
   - Subtitle files will be automatically saved to the video directory

## Subtitle Segmentation Strategy

The system employs a multi-level segmentation strategy to ensure subtitle readability and viewing experience:

1. **Primary Segmentation**:
   - Based on major punctuation (periods, question marks, exclamation marks)
   - Maintain sentence integrity and semantic coherence

2. **Secondary Segmentation**:
   - Use commas, semicolons for long paragraphs
   - Ensure each subtitle doesn't exceed 20 words

3. **Forced Segmentation**:
   - Split by word count when no suitable breakpoints found
   - Ensure subtitle display duration doesn't exceed 8 seconds

4. **Smart Merging**:
   - Automatically merge short segments (less than 3 words)
   - Ensure readability after merging

## Notes

- First run will download necessary model files, which may take some time
- Processing time depends on video length and system performance
- Hardware acceleration automatically selects the best option:
  - NVIDIA GPU: CUDA acceleration
  - Apple Silicon (M1/M2): MPS acceleration
  - Other devices: CPU processing
- Generated SRT files are compatible with most video players
- Ensure you have sufficient DeepSeek API quota
- When using local file paths, ensure write permission to the video directory
- Cache files are stored in the `cache` directory and can be manually cleared

## Tech Stack

- Streamlit: Web interface framework
- Distil-Whisper: Speech recognition
- DeepSeek API: Intelligent error correction and machine translation
- moviepy: Video processing
- srt: Subtitle file processing
- python-dotenv: Environment variable management
- uv: Python package manager

## Error Correction Examples

The system can intelligently handle various scenarios:

1. Technical Terms:
```
Original: github commet hash and pull reques
Corrected: GitHub commit hash and pull request
```

2. Professional Terms:
```
Original: beta nutral and dollar nutral strategy
Corrected: beta neutral and dollar neutral strategy
Translation: 贝塔中性和美元中性策略
```

3. Duplicate Content Handling:
```
Original: Let me explain the concept. The concept is about...
Processed: Let me explain the concept. It is about...
```

## License

MIT License 