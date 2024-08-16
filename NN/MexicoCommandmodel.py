import os
import numpy as np
import librosa
import joblib
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

current_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(current_dir, 'chinotrain')  # Update to new directory
test_audio = os.path.join(current_dir, 'static', 'audio', 'user_input.wav')
def extract_mfcc(file_path, n_mfcc=13):
    """
    从音频文件中提取MFCC特征
    """
    y, sr = librosa.load(file_path, sr=None)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    mfcc_scaled = StandardScaler().fit_transform(mfcc.T)  # 标准化
    return mfcc_scaled

def train_gmm_models(data_folder):
    """
    训练GMM模型
    """
    models = {}
    for label in os.listdir(data_folder):
        class_folder = os.path.join(data_folder, label)
        if not os.path.isdir(class_folder):
            continue
        
        # 获取所有音频文件路径
        file_paths = [os.path.join(class_folder, f) for f in os.listdir(class_folder) if f.endswith('.wav')]
        
        # 提取所有文件的MFCC特征
        features = np.vstack([extract_mfcc(fp) for fp in file_paths])
        
        # 训练GMM模型
        gmm = GaussianMixture(n_components=8, covariance_type='diag', max_iter=200, random_state=0)
        gmm.fit(features)
        
        # 保存模型
        models[label] = gmm
        model_path = f"gmm_models/{label}.pkl"
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        joblib.dump(gmm, model_path)
        
        print(f"Trained and saved GMM for label: {label}")
    
    return models

def recognize_audio(models, test_file_path):
    """
    对测试音频进行识别
    """
    mfcc_features = extract_mfcc(test_file_path)
    scores = {label: model.score(mfcc_features) for label, model in models.items()}
    recognized_label = max(scores, key=scores.get)
    return recognized_label

def main(training_data_folder, test_file_path):
    # 训练GMM模型
    print("Training GMM models...")
    models = train_gmm_models(training_data_folder)
    
    # 识别测试音频
    print("Recognizing test audio...")
    recognized_label = recognize_audio(models, test_file_path)
    print(f"Recognized label: {recognized_label}")

# 调用主程序
if __name__ == "__main__":
    main(data_path, test_audio)
