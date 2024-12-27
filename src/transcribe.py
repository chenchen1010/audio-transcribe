import sys
from pathlib import Path
from whisper_transcriber import WhisperTranscriber
from speaker_recognizer import SpeakerRecognizer

def main(audio_path):
    try:
        # 1. 转录音频为文本
        transcriber = WhisperTranscriber()
        segments = transcriber.transcribe(audio_path)
        
        # 2. 识别说话人
        recognizer = SpeakerRecognizer()
        labeled_segments = recognizer.recognize_speakers(audio_path, segments)
        
        # 3. 保存结果
        output_file = Path(audio_path).stem + "_转录结果.txt"
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(labeled_segments)
            
        print(f"结果已保存到: {output_file}")
        
    except Exception as e:
        print(f"发生错误: {str(e)}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        audio_file = sys.argv[1]
        main(audio_file)
    else:
        print("请将音频文件拖放到此脚本上运行")