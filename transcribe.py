import whisper
import sys
from pathlib import Path
import re
from pydub import AudioSegment
import librosa
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from tqdm import tqdm
import time

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

def trim_audio(audio_path):
    """
    检查音频格式并获取时长，如果需要则转换为MP3格式
    """
    try:
        # 加载音频
        audio = AudioSegment.from_file(audio_path)
        duration_ms = len(audio)  # 获取音频总时长（毫秒）
        
        # 如果不是MP3格式，则转换
        if Path(audio_path).suffix.lower() != '.mp3':
            temp_path = f"temp_{Path(audio_path).stem}.mp3"
            # 导出完整音频
            audio = audio[:duration_ms]  # 使用完整时长
            audio.export(temp_path, format="mp3")
            return temp_path
            
        return audio_path
        
    except Exception as e:
        print(f"处理音频时发生错误: {str(e)}")
        return audio_path

def extract_audio_features(audio_path, start_time, end_time, sr=16000):
    # 加载音频片段
    y, sr = librosa.load(audio_path, sr=sr, offset=start_time, duration=end_time-start_time)
    
    features = {}
    # 音高特征
    pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
    features['pitch'] = float(np.mean(pitches[magnitudes > np.median(magnitudes)]))
    
    # 频谱质心 - 表示声音的"亮度"
    features['spectral_centroid'] = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))
    
    # 声音响度
    features['rms_energy'] = float(np.mean(librosa.feature.rms(y=y)))
    
    # 过零率 - 表示声音的"粗糙度"
    features['zero_crossing_rate'] = float(np.mean(librosa.feature.zero_crossing_rate(y)))
    
    # MFCC特征 - 表示音色
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    features['mfccs'] = mfccs.mean(axis=1).tolist()
    
    return features

def calculate_similarity(features1, features2):
    weights = {
        'pitch': 0.15,
        'spectral_centroid': 0.1,
        'rms_energy': 0.1,
        'zero_crossing_rate': 0.05,
        'mfccs': 0.6
    }
    
    similarity = 0
    for key, weight in weights.items():
        if key == 'mfccs':
            # MFCC使用余弦相似度
            similarity += weight * np.dot(features1[key], features2[key]) / (
                np.linalg.norm(features1[key]) * np.linalg.norm(features2[key]))
        else:
            # 其他特征使用欧氏距离的负值
            similarity -= weight * abs(features1[key] - features2[key])
    return similarity

def transcribe_audio(audio_path):
    try:
        print("正在处理音频...")
        audio_file = trim_audio(audio_path)
        
        # 获取音频时长（分钟）
        audio = AudioSegment.from_file(audio_file)
        duration_minutes = len(audio) / 1000 / 60  # 转换为分钟
        
        print(f"音频时长: {duration_minutes:.1f}分钟")
        print("正在加载模型...")
        model = whisper.load_model("medium")
        
        print(f"正在转录文件: {audio_path}")
        start_time = time.time()
        
        result = model.transcribe(
            audio_file,
            language="zh",
            task="transcribe",
            fp16=False,
            verbose=False,  # 关闭默认的进度显示
            beam_size=5,
            best_of=5,
            temperature=0.0,
            condition_on_previous_text=True,
            initial_prompt="这是一段多人对话的中文音频。"
        )
        
        # 删除临时文件
        if audio_file.startswith("temp_"):
            Path(audio_file).unlink(missing_ok=True)
        
        output_file = Path(audio_path).stem + "_转录结果.txt"
        
        # 第一遍：收集所有语音片段的特征
        print("\n正在分析说话人特征...")
        all_segments_features = []
        total_segments = len(result["segments"])
        
        # 使用tqdm创建进度条
        pbar = tqdm(total=total_segments, desc="处理进度")
        
        for i, segment in enumerate(result["segments"]):
            start_sec = segment.get("start", 0)
            end_sec = segment.get("end", 0)
            
            # 更新进度条描述
            elapsed_time = time.time() - start_time
            progress = (i + 1) / total_segments
            estimated_total = elapsed_time / progress if progress > 0 else 0
            remaining_time = estimated_total - elapsed_time
            
            pbar.set_description(
                f"处理进度 [{time.strftime('%M:%S', time.gmtime(start_sec))} - "
                f"{time.strftime('%M:%S', time.gmtime(end_sec))}] "
                f"预计剩余时间: {time.strftime('%M:%S', time.gmtime(remaining_time))}"
            )
            
            features = extract_audio_features(audio_path, start_sec, end_sec)
            all_segments_features.append({
                'features': features,
                'start': start_sec,
                'end': end_sec,
                'text': segment["text"]
            })
            
            pbar.update(1)
        
        pbar.close()
        
        print("\n正在聚类分析说话人...")
        # 准备特征矩阵
        feature_matrix = []
        for segment in all_segments_features:
            features = []
            features.extend(segment['features']['mfccs'])
            features.append(segment['features']['pitch'])
            features.append(segment['features']['spectral_centroid'])
            feature_matrix.append(features)
        
        # 使用层次聚类
        clustering = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=1.0,
            linkage='ward'
        )
        labels = clustering.fit_predict(feature_matrix)
        
        # 根据聚类结果生成转录文本
        print("\n正在生成转录文本...")
        formatted_text = ""
        for i, segment in enumerate(all_segments_features):
            speaker_id = f"说话人{labels[i] + 1}"
            text = clean_text(segment['text'])
            if text:
                time_info = f"[{time.strftime('%M:%S', time.gmtime(segment['start']))} - {time.strftime('%M:%S', time.gmtime(segment['end']))}]"
                formatted_text += f"{speaker_id} {time_info}：{text}\n\n"
        
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(formatted_text)
            
        total_time = time.time() - start_time
        print(f"\n转录完成！用时: {time.strftime('%M:%S', time.gmtime(total_time))}")
        print(f"结果已保存到: {output_file}")
        print(f"共识别出 {len(set(labels))} 位说话人")
        
    except Exception as e:
        print(f"发生错误: {str(e)}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        audio_file = sys.argv[1]
        transcribe_audio(audio_file)
    else:
        print("请将音频文件拖放到此脚本上运行")