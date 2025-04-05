import logging
from typing import Optional, Dict, Any
import os
from dotenv import load_dotenv
import requests
import re
from tenacity import retry, stop_after_attempt, wait_exponential

# 加载环境变量
load_dotenv()

class TranslationError(Exception):
    """翻译相关错误的基类"""
    pass

class APIError(TranslationError):
    """API调用错误"""
    pass

class ContentError(TranslationError):
    """内容处理错误"""
    pass

class Translator:
    def __init__(self):
        self.api_key = os.getenv("DEEPSEEK_API_KEY")
        if not self.api_key:
            raise ValueError("请设置 DEEPSEEK_API_KEY 环境变量")
        
        # 编译正则表达式
        self.result_pattern = re.compile(r'<result>(.*?)</result>', re.DOTALL)
        self.translation_pattern = re.compile(r'<translation>(.*?)</translation>', re.DOTALL)
    
    def review_english_with_context(self, text: str, prev_text: Optional[str] = None, next_text: Optional[str] = None) -> str:
        """使用上下文进行英文文本纠错
        
        Args:
            text: 需要纠错的文本
            prev_text: 上文内容
            next_text: 下文内容
            
        Returns:
            str: 纠错后的文本
            
        Raises:
            ContentError: 当文本处理失败时
        """
        try:
            prompt = self._build_review_prompt(text, prev_text, next_text)
            response = self._call_api(prompt)
            return self._extract_result(response, self.result_pattern, text)
        except Exception as e:
            logging.error(f"文本纠错失败: {str(e)}")
            raise ContentError(f"文本纠错失败: {str(e)}")
    
    def translate(self, text: str) -> str:
        """将英文文本翻译为中文
        
        Args:
            text: 需要翻译的英文文本
            
        Returns:
            str: 中文翻译结果
            
        Raises:
            ContentError: 当翻译失败时
        """
        try:
            prompt = self._build_translation_prompt(text)
            response = self._call_api(prompt)
            return self._extract_result(response, self.translation_pattern, f"[翻译失败: {text}]")
        except Exception as e:
            logging.error(f"翻译失败: {str(e)}")
            raise ContentError(f"翻译失败: {str(e)}")
    
    def _build_review_prompt(self, text: str, prev_text: Optional[str], next_text: Optional[str]) -> str:
        """构建文本纠错的提示词"""
        return f"""作为一个专业的字幕校对专家，请根据上下文对文本进行纠错和优化。

===上文===
{prev_text if prev_text else '（无）'}

===当前文本===
{text}

===下文===
{next_text if next_text else '（无）'}

请严格遵循以下要求：
1. 保持专业术语的准确性
2. 确保句子的连贯性和完整性
3. 修正语法和拼写错误
4. 如果文本已经正确，直接返回原文本
5. 必须使用<result>标签包裹你的输出

示例输出格式：
<result>修正后的英文文本</result>

请仅返回被<result>标签包裹的修正文本，不要包含任何其他内容。"""
    
    def _build_translation_prompt(self, text: str) -> str:
        """构建翻译的提示词"""
        return f"""作为一个专业的字幕翻译专家，请将以下英文文本翻译成中文。

===英文原文===
{text}

翻译要求：
1. 保持专业术语的准确性
2. 确保译文通顺自然
3. 保持原文的语气和风格
4. 必须使用<translation>标签包裹你的中文翻译

示例输出格式：
<translation>中文翻译结果</translation>

请仅返回被<translation>标签包裹的中文翻译，不要包含任何其他内容。"""
    
    def _extract_result(self, response: str, pattern: re.Pattern, default: str) -> str:
        """从响应中提取结果
        
        Args:
            response: API响应内容
            pattern: 用于提取内容的正则表达式
            default: 提取失败时的默认值
            
        Returns:
            str: 提取的内容或默认值
        """
        match = pattern.search(response)
        if not match:
            logging.warning(f"无法从响应中提取内容: {response[:100]}...")
            return default
        return match.group(1).strip()
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry_error_cls=APIError
    )
    def _call_api(self, prompt: str) -> str:
        """调用 DeepSeek API
        
        Args:
            prompt: 提示词
            
        Returns:
            str: API响应内容
            
        Raises:
            APIError: 当API调用失败时
        """
        try:
            response = requests.post(
                "https://api.deepseek.com/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "deepseek-chat",
                    "messages": [
                        {
                            "role": "system", 
                            "content": """你是一个专业的字幕翻译和校对专家。你的回复必须严格遵循用户指定的输出格式，
                            使用对应的XML标签包裹输出内容。不要添加任何额外的解释或标记。"""
                        },
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": 0.3,
                    "max_tokens": 1000
                },
                timeout=30
            )
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]
        except requests.exceptions.RequestException as e:
            raise APIError(f"API请求失败: {str(e)}")
        except (KeyError, IndexError) as e:
            raise APIError(f"API响应格式错误: {str(e)}")
        except Exception as e:
            raise APIError(f"API调用异常: {str(e)}") 