import librosa
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from tqdm import tqdm
import time

class SpeakerRecognizer:
    def __init__(self):
        self.weights = {
            'pitch': 0.2,
            'spectral_centroid': 0.15,
            'rms_energy': 0.05,
            'zero_crossing_rate': 0.05,
            'mfccs': 0.55
        }
    
    def extract_features(self, audio_path, start_time, end_time, sr=16000):
        """提取音频特征"""
        y, sr = librosa.load(audio_path, sr=sr, offset=start_time, duration=end_time-start_time)
        
        features = {}
        pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
        features['pitch'] = float(np.mean(pitches[magnitudes > np.median(magnitudes)]))
        features['spectral_centroid'] = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))
        features['rms_energy'] = float(np.mean(librosa.feature.rms(y=y)))
        features['zero_crossing_rate'] = float(np.mean(librosa.feature.zero_crossing_rate(y)))
        features['mfccs'] = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13).mean(axis=1).tolist()
        
        return features
    
    def calculate_similarity(self, features1, features2):
        """计算特征相似度"""
        similarity = 0
        for key, weight in self.weights.items():
            if key == 'mfccs':
                similarity += weight * np.dot(features1[key], features2[key]) / (
                    np.linalg.norm(features1[key]) * np.linalg.norm(features2[key]))
            else:
                similarity -= weight * abs(features1[key] - features2[key])
        return similarity
    
    def recognize_speakers(self, audio_path, segments):
        """识别说话人"""
        print("\n正在分析说话人特征...")
        all_segments_features = []
        
        # 提取特征
        with tqdm(total=len(segments), desc="处理进度") as pbar:
            for segment in segments:
                start_time = segment.get("start", 0)
                end_time = segment.get("end", 0)
                features = self.extract_features(audio_path, start_time, end_time)
                
                all_segments_features.append({
                    'features': features,
                    'start': start_time,
                    'end': end_time,
                    'text': segment["text"]
                })
                pbar.update(1)
        
        # 准备特征矩阵
        feature_matrix = []
        for segment in all_segments_features:
            features = []
            features.extend(segment['features']['mfccs'])
            features.append(segment['features']['pitch'])
            features.append(segment['features']['spectral_centroid'])
            feature_matrix.append(features)
        
        # 聚类分析
        clustering = AgglomerativeClustering(
            n_clusters=2,
            linkage='average'
        )
        labels = clustering.fit_predict(feature_matrix)
        
        # 生成结果
        formatted_text = ""
        current_speaker = None
        current_text = []
        start_time = None
        
        for i, segment in enumerate(all_segments_features):
            speaker_id = f"说话人{labels[i] + 1}"
            
            # 说话人改变时才添加标签和分段
            if speaker_id != current_speaker:
                # 输出上一个说话人的内容
                if current_text:
                    formatted_text += f"{current_speaker} [{time.strftime('%M:%S', time.gmtime(start_time))}]：{''.join(current_text)}\n\n"
                
                # 开始新的说话人
                current_speaker = speaker_id
                current_text = [segment['text']]
                start_time = segment['start']
            else:
                # 同一说话人，直接拼接文本
                current_text.append(segment['text'])
        
        # 输出最后一个说话人的内容
        if current_text:
            formatted_text += f"{current_speaker} [{time.strftime('%M:%S', time.gmtime(start_time))}]：{''.join(current_text)}\n\n"
        
        return formatted_text 