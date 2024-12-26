import whisper
import sys
from pathlib import Path
import json
import re
from pydub import AudioSegment

def clean_text(text):
    """对文本进行清理和优化"""
    # 修正常见的错别字（可以根据需要添加更多）
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
    
    # 应用修正
    for wrong, right in corrections.items():
        text = text.replace(wrong, right)
    
    # 确保标点符号前后格式正确
    text = re.sub(r'([，。！？；：])\s*([，。！？；：])', r'\1', text)
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()

def trim_audio(audio_path, duration_ms=90000):  # 90秒 = 90000毫秒
    """截取音频的前90秒"""
    try:
        # 加载音频
        audio = AudioSegment.from_file(audio_path)
        # 截取前90秒
        trimmed_audio = audio[:duration_ms]
        # 保存临时文件
        temp_path = f"temp_{Path(audio_path).name}"
        trimmed_audio.export(temp_path, format="mp3")
        return temp_path
    except Exception as e:
        print(f"处理音频时发生错误: {str(e)}")
        return audio_path

def get_speaker_label(start_time, speakers_segments):
    """根据时间戳确定说话人"""
    for speaker_id, segments in speakers_segments.items():
        for segment in segments:
            if start_time >= segment[0] and start_time <= segment[1]:
                return f"说话人{speaker_id}"
    return "说话人1"  # 默认说话人

def transcribe_audio(audio_path):
    try:
        print("正在处理音频...")
        temp_audio = trim_audio(audio_path)
        
        print("正在加载模型...")
        model = whisper.load_model("medium")
        
        print(f"正在转录文件: {audio_path}")
        result = model.transcribe(
            temp_audio,
            language="zh",
            task="transcribe",
            word_timestamps=True,
            fp16=False,
            verbose=True,
            beam_size=5,
            best_of=5,
            temperature=0.0,
            condition_on_previous_text=True,
            initial_prompt="这是一段中文音频。"
        )
        
        # 删除临时文件
        if temp_audio.startswith("temp_"):
            Path(temp_audio).unlink(missing_ok=True)
        
        output_file = Path(audio_path).stem + "_转录结果.txt"
        
        # 使用简单的时间间隔来区分说话人
        current_speaker = None
        formatted_text = ""
        last_end_time = 0
        speaker_change_threshold = 1.0  # 1秒以上的停顿认为是说话人改变
        
        for segment in result["segments"]:
            start_time = segment["start"]
            
            # 检测说话人改变
            if last_end_time > 0 and (start_time - last_end_time) > speaker_change_threshold:
                current_speaker = None
            
            # 如果没有当前说话人或检测到说话人改变
            if current_speaker is None:
                current_speaker = f"说话人{len(set(formatted_text.split('说话人'))) + 1}"
            
            text = clean_text(segment["text"])
            if text:
                formatted_text += f"{current_speaker}：{text}\n\n"
            
            last_end_time = segment["end"]
        
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(formatted_text)
            
        print(f"转录完成！结果已保存到: {output_file}")
        
    except Exception as e:
        print(f"发生错误: {str(e)}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        audio_file = sys.argv[1]
        transcribe_audio(audio_file)
    else:
        print("请将音频文件拖放到此脚本上运行") 