"""启动视频字幕生成器"""
from streamlit.web import cli as stcli
import sys
from pathlib import Path

if __name__ == "__main__":
    sys.argv = [
        "streamlit",
        "run",
        str(Path(__file__).parent / "src" / "video_subtitle_generator" / "app.py"),
    ]
    sys.exit(stcli.main()) 