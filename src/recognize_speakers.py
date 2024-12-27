import sys
from pathlib import Path
import json
from speaker_recognizer import SpeakerRecognizer

def load_segments(json_path):
    """从JSON文件加载语音片段"""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data["segments"]

def main(audio_path, segments_json):
    try:
        # 1. 加载语音片段
        print(f"正在加载语音片段: {segments_json}")
        segments = load_segments(segments_json)
        
        # 2. 识别说话人
        recognizer = SpeakerRecognizer()
        labeled_segments = recognizer.recognize_speakers(audio_path, segments)
        
        # 3. 保存结果
        output_file = Path(audio_path).stem + "_说话人识别结果.txt"
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(labeled_segments)
            
        print(f"结果已保存到: {output_file}")
        
    except Exception as e:
        print(f"发生错误: {str(e)}")

if __name__ == "__main__":
    if len(sys.argv) > 2:
        audio_file = sys.argv[1]
        segments_json = sys.argv[2]
        main(audio_file, segments_json)
    else:
        print("使用方法: python recognize_speakers.py 音频文件.mp3 语音片段.json") 