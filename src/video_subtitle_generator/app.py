import streamlit as st
import torch
from transformers import pipeline, AutoModelForSpeechSeq2Seq, AutoProcessor
import os
import tempfile
from moviepy.editor import VideoFileClip
import srt
from datetime import timedelta
import re
from translator import Translator
from dotenv import load_dotenv
from pathlib import Path
import io
import logging
import sys
from typing import Union, List, Dict, Optional
import json
import hashlib

# 加载环境变量
load_dotenv()

# 设置页面配置
st.set_page_config(
    page_title="视频字幕生成器",
    page_icon="🎬",
    layout="wide"
)

# 初始化会话状态
if 'video_paths' not in st.session_state:
    st.session_state.video_paths = set()

def on_file_change():
    """当文件被选择时的回调函数"""
    if st.session_state.uploaded_files:
        for file in st.session_state.uploaded_files:
            # 获取文件的本地路径
            try:
                file_path = file.name
                if os.path.exists(file_path):
                    st.session_state.video_paths.add(file_path)
                else:
                    st.warning(f"无法访问文件: {file_path}")
            except Exception as e:
                st.error(f"处理文件时出错: {str(e)}")

# 初始化翻译器
@st.cache_resource
def load_translator():
    if not os.getenv("DEEPSEEK_API_KEY"):
        st.error("请设置 DEEPSEEK_API_KEY 环境变量")
        st.stop()
    return Translator()

# 初始化 Whisper 模型
@st.cache_resource
def load_whisper_model():
    """加载 Whisper 模型"""
    model_id = os.getenv("WHISPER_MODEL_ID", "distil-whisper/distil-large-v2")
    
    # 设备选择逻辑
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    
    print(f"使用设备: {device}")
    
    # 创建语音识别管道
    pipe = pipeline(
        "automatic-speech-recognition",
        model=model_id,
        chunk_length_s=30,
        return_timestamps=True,  # 使用句子级别的时间戳
        device=device
    )
    
    return pipe

def extract_audio(video_source, is_path=False):
    """从视频中提取音频
    
    Args:
        video_source: 视频源，可以是文件路径或上传的文件对象
        is_path: 是否是文件路径
    
    Returns:
        临时音频文件的路径
    """
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_audio:
        if is_path:
            # 直接从本地文件提取音频
            video = VideoFileClip(video_source)
        else:
            # 从上传的文件提取音频
            with tempfile.NamedTemporaryFile(suffix=f'.{video_source.name.split(".")[-1]}', delete=False) as temp_video:
                temp_video.write(video_source.getvalue())
                video = VideoFileClip(temp_video.name)
                # 删除临时视频文件
                os.unlink(temp_video.name)
        
        # 提取并保存音频
        video.audio.write_audiofile(temp_audio.name)
        video.close()
        
        return temp_audio.name

def split_long_segment(segment: Dict, max_duration: float = 8.0, max_words: int = 20) -> List[Dict]:
    """将过长的段落分割成更小的片段
    
    Args:
        segment: 字幕段落
        max_duration: 最大时长（秒）
        max_words: 每段最大单词数（英文）
    
    Returns:
        List[Dict]: 分割后的段落列表
    """
    text = segment["text"].strip()
    duration = segment["end"] - segment["start"]
    
    # 计算当前文本的单词数
    word_count = len(re.findall(r'\b\w+\b', text))
    
    # 如果段落不需要分割，直接返回
    if duration <= max_duration and word_count <= max_words:
        return [segment]
    
    # 按句子分割文本
    sentences = []
    
    # 使用更细致的分割规则
    # 首先按主要标点分割
    major_parts = re.split(r'([.!?。！？]+)', text)
    
    # 合并标点符号与其前面的文本
    combined_parts = []
    i = 0
    while i < len(major_parts):
        if i + 1 < len(major_parts):
            # 合并文本和标点
            combined_parts.append(major_parts[i] + major_parts[i+1])
            i += 2
        else:
            # 处理最后一个部分
            if major_parts[i].strip():
                combined_parts.append(major_parts[i])
            i += 1
    
    # 如果没有主要标点，或者分割后还是太长，进一步处理
    for part in combined_parts:
        part = part.strip()
        if not part:
            continue
            
        # 计算当前部分的单词数
        part_word_count = len(re.findall(r'\b\w+\b', part))
        
        # 如果当前部分超过最大单词数，尝试按次要标点分割
        if part_word_count > max_words:
            # 按逗号、分号等次要分隔符分割
            minor_parts = re.split(r'([,;，；]+)', part)
            
            # 合并次要标点与其前面的文本
            i = 0
            while i < len(minor_parts):
                if i + 1 < len(minor_parts):
                    # 合并文本和标点
                    current = minor_parts[i] + minor_parts[i+1]
                    i += 2
                else:
                    # 处理最后一个部分
                    current = minor_parts[i]
                    i += 1
                
                current = current.strip()
                if not current:
                    continue
                
                # 计算当前文本的单词数
                current_word_count = len(re.findall(r'\b\w+\b', current))
                
                # 如果合并后的文本仍然太长，按单词数分割
                if current_word_count > max_words:
                    words = current.split()
                    # 每 max_words/2 个单词为一组
                    chunk_size = max(1, max_words // 2)
                    for j in range(0, len(words), chunk_size):
                        chunk = ' '.join(words[j:j+chunk_size])
                        if chunk:
                            sentences.append(chunk)
                else:
                    sentences.append(current)
        else:
            sentences.append(part)
    
    # 如果没有找到任何分割点，强制按单词数分割
    if not sentences:
        words = text.split()
        # 每 max_words/2 个单词为一组
        chunk_size = max(1, max_words // 2)
        for i in range(0, len(words), chunk_size):
            chunk = ' '.join(words[i:i+chunk_size])
            if chunk:
                sentences.append(chunk)
    
    # 计算时间戳
    result = []
    num_sentences = len(sentences)
    
    # 计算每个句子的权重（基于单词数）
    weights = []
    total_words = 0
    for sentence in sentences:
        # 计算有效单词数（排除标点符号）
        word_count = len(re.findall(r'\b\w+\b', sentence))
        weights.append(word_count)
        total_words += word_count
    
    # 确保权重之和为1
    if total_words > 0:
        weights = [w/total_words for w in weights]
    else:
        # 如果没有有效单词，平均分配
        weights = [1.0/num_sentences] * num_sentences
    
    # 分配时间戳
    current_time = segment["start"]
    min_duration = 1.0  # 最小持续时间（秒）
    
    for i, (sentence, weight) in enumerate(zip(sentences, weights)):
        # 计算当前句子的理想持续时间
        ideal_duration = duration * weight
        
        # 确保最小持续时间
        current_duration = max(ideal_duration, min_duration)
        
        # 如果是最后一个句子，使用剩余的所有时间
        if i == num_sentences - 1:
            end_time = segment["end"]
        else:
            # 确保不会超过总时长
            end_time = min(current_time + current_duration, segment["end"])
            # 为后续句子预留足够时间
            remaining_duration = segment["end"] - end_time
            remaining_sentences = num_sentences - i - 1
            if remaining_duration < remaining_sentences * min_duration:
                # 如果剩余时间不足，适当缩短当前句子
                end_time = segment["end"] - (remaining_sentences * min_duration)
        
        result.append({
            "text": sentence,
            "start": current_time,
            "end": end_time
        })
        
        current_time = end_time
    
    return result

def remove_duplicates(text1: str, text2: str) -> str:
    """移除两个文本之间的重复内容
    
    Args:
        text1: 前一段文本
        text2: 当前文本
    
    Returns:
        str: 移除重复内容后的文本
    """
    if not text1 or not text2:
        return text2
    
    words1 = text1.split()
    words2 = text2.split()
    
    # 查找最长的重复序列
    for i in range(len(words1), 0, -1):
        if ' '.join(words1[-i:]) == ' '.join(words2[:i]):
            return ' '.join(words2[i:])
    
    return text2

def create_srt_content(segments: List[Dict], translator: Translator) -> str:
    """创建SRT格式的字幕内容"""
    srt_segments = []
    merged_segments = []
    
    # 第一步：合并过短的片段
    i = 0
    while i < len(segments):
        current = segments[i]
        text = current.get("text", "").strip()
        
        # 如果当前文本太短且不是最后一个，尝试与下一个合并
        if len(text.split()) < 3 and i + 1 < len(segments):  # 减少合并阈值
            next_seg = segments[i + 1]
            next_text = next_seg.get("text", "").strip()
            
            # 只有当合并后不会太长时才合并
            combined_text = f"{text} {next_text}"
            if len(combined_text) <= 50:  # 控制合并后的长度
                merged_segments.append({
                    "text": combined_text,
                    "start": current["start"],
                    "end": next_seg["end"]
                })
                i += 2
            else:
                merged_segments.append(current)
                i += 1
        else:
            merged_segments.append(current)
            i += 1
    
    # 第二步：分割过长的片段
    split_segments = []
    for segment in merged_segments:
        split_segments.extend(split_long_segment(segment))
    
    # 创建术语映射表
    term_mapping = {
        "funding arbitrage": "资金费率套利",
        "perpetual": "永续合约",
        "spot": "现货",
        "beta neutral": "贝塔中性",
        "dollar neutral": "美元中性",
        "leverage": "杠杆",
        "beta": "贝塔",
        "capital asset pricing model": "资本资产定价模型",
        "regression model": "回归模型",
        "market index": "市场指数",
        "sensitivity": "敏感度",
    }
    
    # 第三步：处理字幕段落
    prev_text = ""
    for i, segment in enumerate(split_segments):
        try:
            # 获取上下文
            next_text = split_segments[i+1]["text"] if i < len(split_segments)-1 else None
            
            # 处理当前段落
            start_time = timedelta(seconds=segment["start"])
            end_time = timedelta(seconds=segment["end"])
            text = segment["text"]
            
            # 移除重复内容
            if prev_text:
                text = remove_duplicates(prev_text, text)
            
            # 使用上下文进行深度纠错
            corrected_text = translator.review_english_with_context(text, prev_text, next_text)
            
            # 应用术语映射
            for eng, chn in term_mapping.items():
                corrected_text = corrected_text.replace(eng.lower(), eng)
            
            # 翻译文本
            chinese_text = translator.translate(corrected_text)
            
            # 确保术语一致性
            for eng, chn in term_mapping.items():
                chinese_text = chinese_text.replace(eng, chn)
            
            # 创建字幕
            srt_segments.append(
                srt.Subtitle(index=len(srt_segments)+1,
                           start=start_time,
                           end=end_time,
                           content=f"{corrected_text}\n{chinese_text}")
            )
            
            # 更新上下文
            prev_text = text
            
        except Exception as e:
            logging.error(f"处理字幕段落 {i+1} 失败: {str(e)}")
            srt_segments.append(
                srt.Subtitle(index=len(srt_segments)+1,
                           start=start_time,
                           end=end_time,
                           content=f"{text}\n[翻译失败]")
            )
    
    return srt.compose(srt_segments)

def get_cache_path(video_path: str) -> str:
    """获取缓存文件路径
    
    Args:
        video_path: 视频文件路径
        
    Returns:
        str: 缓存文件路径
    """
    # 使用视频文件的路径生成唯一的哈希值
    video_hash = hashlib.md5(str(video_path).encode()).hexdigest()
    cache_dir = Path("cache")
    cache_dir.mkdir(exist_ok=True)
    return str(cache_dir / f"{video_hash}.json")

def save_to_cache(cache_path: str, result: Dict) -> None:
    """保存识别结果到缓存
    
    Args:
        cache_path: 缓存文件路径
        result: 识别结果
    """
    with open(cache_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

def load_from_cache(cache_path: str) -> Optional[Dict]:
    """从缓存加载识别结果
    
    Args:
        cache_path: 缓存文件路径
        
    Returns:
        Dict: 缓存的识别结果，如果不存在则返回 None
    """
    try:
        if os.path.exists(cache_path):
            with open(cache_path, 'r', encoding='utf-8') as f:
                return json.load(f)
    except Exception as e:
        logging.warning(f"读取缓存失败: {str(e)}")
    return None

def process_video(video_source, pipe, translator, is_path=False):
    """处理视频文件，支持文件路径或上传的文件对象"""
    try:
        # 显示总体进度条
        progress_placeholder = st.empty()
        progress_bar = progress_placeholder.progress(0)
        status_placeholder = st.empty()
        
        # 获取缓存路径
        cache_path = get_cache_path(str(video_source) if is_path else video_source.name)
        
        # 尝试从缓存加载
        cached_result = load_from_cache(cache_path)
        if cached_result:
            status_placeholder.info("✨ 使用缓存的识别结果")
            progress_bar.progress(20)
            raw_segments = []
            if isinstance(cached_result, dict):
                if "chunks" in cached_result:
                    # 处理 chunks 格式
                    total_chunks = len(cached_result["chunks"])
                    for i, chunk in enumerate(cached_result["chunks"]):
                        if isinstance(chunk, dict):
                            text = chunk.get("text", "").strip()
                            timestamp = chunk.get("timestamp", (0, 0))
                            raw_segments.append({
                                "text": text,
                                "start": timestamp[0],
                                "end": timestamp[1]
                            })
                        # 更新进度
                        progress = 20 + (i + 1) / total_chunks * 10
                        progress_bar.progress(int(progress))
                        status_placeholder.info(f"📝 处理缓存数据: {i+1}/{total_chunks}")
                else:
                    # 如果没有 chunks，使用完整文本
                    raw_segments = [{
                        "text": cached_result.get("text", "").strip(),
                        "start": 0,
                        "end": 0
                    }]
                    progress_bar.progress(30)
        else:
            # 提取音频
            status_placeholder.info("🎵 提取音频中...")
            progress_bar.progress(10)
            audio_path = extract_audio(video_source, is_path)
            progress_bar.progress(20)
            
            # 生成字幕
            status_placeholder.info("🎯 识别语音中...")
            result = pipe(audio_path)
            progress_bar.progress(30)
            
            # 保存到缓存
            status_placeholder.info("💾 保存缓存...")
            save_to_cache(cache_path, result)
            
            # 处理结果为标准格式
            raw_segments = []
            if isinstance(result, dict):
                if "chunks" in result:
                    # 处理 chunks 格式
                    total_chunks = len(result["chunks"])
                    for i, chunk in enumerate(result["chunks"]):
                        if isinstance(chunk, dict):
                            text = chunk.get("text", "").strip()
                            timestamp = chunk.get("timestamp", (0, 0))
                            raw_segments.append({
                                "text": text,
                                "start": timestamp[0],
                                "end": timestamp[1]
                            })
                        # 更新进度
                        progress = 30 + (i + 1) / total_chunks * 10
                        progress_bar.progress(int(progress))
                        status_placeholder.info(f"📝 处理音频段落: {i+1}/{total_chunks}")
                else:
                    # 如果没有 chunks，使用完整文本
                    raw_segments = [{
                        "text": result.get("text", "").strip(),
                        "start": 0,
                        "end": 0
                    }]
            
            # 清理临时文件
            os.unlink(audio_path)
        
        # 分割长段落
        status_placeholder.info("✂️ 分割长段落...")
        segments = []
        total_raw = len(raw_segments)
        for i, segment in enumerate(raw_segments):
            split_segs = split_long_segment(segment)
            segments.extend(split_segs)
            # 更新进度
            progress = 40 + (i + 1) / total_raw * 20
            progress_bar.progress(int(progress))
            status_placeholder.info(f"✂️ 分割长段落: {i+1}/{total_raw}")
        
        # 创建SRT内容
        status_placeholder.info("🌏 生成双语字幕中...")
        total_segments = len(segments)
        srt_segments = []
        
        # 分批处理字幕段落，每批显示进度
        for i, segment in enumerate(segments):
            try:
                # 获取上下文
                prev_text = segments[i-1]["text"] if i > 0 else None
                next_text = segments[i+1]["text"] if i < total_segments-1 else None
                
                # 处理当前段落
                start_time = timedelta(seconds=segment["start"])
                end_time = timedelta(seconds=segment["end"])
                text = segment["text"]
                
                # 移除重复内容
                if prev_text:
                    text = remove_duplicates(prev_text, text)
                
                # 使用上下文进行深度纠错
                status_placeholder.info(f"🔍 正在校对第 {i+1}/{total_segments} 段字幕")
                corrected_text = translator.review_english_with_context(text, prev_text, next_text)
                
                # 翻译文本
                status_placeholder.info(f"🌏 正在翻译第 {i+1}/{total_segments} 段字幕")
                chinese_text = translator.translate(corrected_text)
                
                # 创建字幕
                srt_segments.append(
                    srt.Subtitle(index=len(srt_segments)+1,
                               start=start_time,
                               end=end_time,
                               content=f"{corrected_text}\n{chinese_text}")
                )
                
                # 更新进度
                progress = 60 + (i + 1) / total_segments * 30
                progress_bar.progress(int(progress))
                
            except Exception as e:
                logging.error(f"处理字幕段落 {i+1} 失败: {str(e)}")
                srt_segments.append(
                    srt.Subtitle(index=len(srt_segments)+1,
                               start=start_time,
                               end=end_time,
                               content=f"{text}\n[翻译失败]")
                )
        
        # 合成最终的SRT内容
        srt_content = srt.compose(srt_segments)
        
        # 保存或下载字幕文件
        status_placeholder.info("💾 保存字幕文件...")
        progress_bar.progress(95)
        
        if is_path:
            video_dir = os.path.dirname(video_source)
            video_name = Path(video_source).stem
            srt_path = os.path.join(video_dir, f"{video_name}.srt")
            with open(srt_path, 'w', encoding='utf-8') as f:
                f.write(srt_content)
            status_placeholder.success(f"✅ 字幕文件已保存到: {srt_path}")
        else:
            # 如果是上传的文件，提供下载按钮
            srt_filename = f"{Path(video_source.name).stem}.srt"
            st.download_button(
                label=f"下载 {srt_filename}",
                data=srt_content.encode('utf-8'),
                file_name=srt_filename,
                mime='text/srt'
            )
            status_placeholder.success("✅ 字幕生成完成，请点击下载按钮获取字幕文件")
        
        # 完成
        progress_bar.progress(100)
        
    except Exception as e:
        name = video_source if isinstance(video_source, str) else video_source.name
        st.error(f"❌ 处理文件 {name} 时发生错误: {str(e)}")
        # 打印更详细的错误信息以便调试
        import traceback
        st.error(traceback.format_exc())

def main():
    st.title("🎬 视频字幕生成器")
    st.write("选择视频文件或输入本地文件路径，自动生成中英文双语字幕")
    
    # API密钥设置（优先使用环境变量，如果没有则使用界面输入）
    if not os.getenv("DEEPSEEK_API_KEY"):
        api_key = st.sidebar.text_input("DeepSeek API Key", type="password")
        if api_key:
            os.environ["DEEPSEEK_API_KEY"] = api_key
    
    # 加载模型
    try:
        pipe = load_whisper_model()
        translator = load_translator()
    except Exception as e:
        st.error(f"模型加载失败: {str(e)}")
        return

    # 选择输入方式
    input_method = st.radio(
        "选择输入方式",
        ["上传文件", "本地文件路径"]
    )
    
    if input_method == "上传文件":
        # 文件上传
        uploaded_files = st.file_uploader(
            "选择视频文件（支持 MP4, MOV, AVI, MKV）",
            type=["mp4", "mov", "avi", "mkv"],
            accept_multiple_files=True
        )
        
        if uploaded_files:
            st.write(f"已上传 {len(uploaded_files)} 个文件")
            if st.button("开始处理"):
                for uploaded_file in uploaded_files:
                    st.write(f"处理文件: {uploaded_file.name}")
                    process_video(uploaded_file, pipe, translator, is_path=False)
    
    else:
        # 本地文件路径输入
        local_paths = st.text_area(
            "输入本地视频文件路径（每行一个路径）",
            help="例如：\n/path/to/video1.mp4\n/path/to/video2.mp4",
            height=100
        )
        
        if local_paths:
            paths = [path.strip() for path in local_paths.split('\n') if path.strip()]
            valid_paths = []
            
            # 验证文件
            for path in paths:
                if not os.path.exists(path):
                    st.error(f"文件不存在: {path}")
                    continue
                if not path.lower().endswith(('.mp4', '.mov', '.avi', '.mkv')):
                    st.error(f"不支持的文件格式: {path}")
                    continue
                valid_paths.append(path)
            
            if valid_paths:
                st.write(f"找到 {len(valid_paths)} 个有效文件")
                if st.button("开始处理"):
                    for path in valid_paths:
                        st.write(f"处理文件: {path}")
                        process_video(path, pipe, translator, is_path=True)

if __name__ == "__main__":
    main() 