import whisper
from pathlib import Path
import re
from pydub import AudioSegment
import time

class WhisperTranscriber:
    def __init__(self):
        self.model = None
        
    def clean_text(self, text):
        """对文本进行清理和优化"""
        corrections = {
            '的的': '的',
            '了了': '了',
            '吗吗': '吗',
            '呢呢': '呢',
            '嘛嘛': '嘛',
            '啊啊': '啊',
            '哦哦': '哦',
            '额额': '额',
        }
        
        for wrong, right in corrections.items():
            text = text.replace(wrong, right)
        
        text = re.sub(r'([，。！？；：])\s*([，。！？；：])', r'\1', text)
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    def prepare_audio(self, audio_path):
        """准备音频文件"""
        try:
            audio = AudioSegment.from_file(audio_path)
            if Path(audio_path).suffix.lower() != '.mp3':
                temp_path = f"temp_{Path(audio_path).stem}.mp3"
                audio.export(temp_path, format="mp3")
                return temp_path
            return audio_path
        except Exception as e:
            print(f"处理音频时发生错误: {str(e)}")
            return audio_path
    
    def transcribe(self, audio_path):
        """转录音频为文本"""
        print("正在处理音频...")
        audio_file = self.prepare_audio(audio_path)
        
        print("正在加载模型...")
        self.model = whisper.load_model("small")
        #whisper各种模型说明
        #tiny: 最小模型(39M)，速度最快但准确率最低，适合快速测试
        #base: 基础模型(74M)，速度和准确率平衡，适合一般用途
        #small: 中等模型(244M)，准确率优于base，但速度较慢
        #medium: 较大模型(769M)，准确率高，需要较好的GPU
        #large: 最大模型(1.5G)，准确率最高但速度最慢，需要高性能GPU


        print(f"正在转录文件: {audio_path}")
        result = self.model.transcribe(
            audio_file,
            language="zh",
            task="transcribe",
            fp16=False,
            verbose=True,
            beam_size=5,
            best_of=5,
            temperature=0.0,
            condition_on_previous_text=True,
            initial_prompt="这是一段多人对话的中文音频。"
        )
        
        # 清理临时文件
        if audio_file.startswith("temp_"):
            Path(audio_file).unlink(missing_ok=True)
            
        return result["segments"] 