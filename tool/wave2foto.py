import os
import librosa
import matplotlib.pyplot as plt
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
source_folder = os.path.join(current_dir, 'chinotrainV2')  # Update to new directory
target_folder = os.path.join(current_dir, 'wave_fotoV2')

# 建立目標資料夾
if not os.path.exists(target_folder):
    os.makedirs(target_folder)

# 遍歷每一個資料夾
for folder_num in range(1, 4):
    folder_name = str(folder_num)
    folder_path = os.path.join(source_folder, folder_name)
    target_subfolder = os.path.join(target_folder, folder_name)

    # 建立目標子資料夾
    if not os.path.exists(target_subfolder):
        os.makedirs(target_subfolder)

    # 遍歷資料夾中的每一個音檔
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.wav'):
            file_path = os.path.join(folder_path, file_name)

            # 讀取音檔
            y, sr = librosa.load(file_path, sr=None)

            # # 找到前後靜音部分並去除
            # non_silent_intervals = librosa.effects.split(y, top_db=80)
            # start_sample, end_sample = non_silent_intervals[0][0], non_silent_intervals[-1][1]
            # non_silent_audio = y[start_sample:end_sample]

            # 繪製波形圖
            plt.figure(figsize=(1, 1), dpi=100)  # 設定圖片大小為 100x100 像素
            plt.axis('off')  # 去除座標軸
            librosa.display.waveshow(y, sr=sr)
            plt.subplots_adjust(left=0, right=1, top=1, bottom=0)  # 去除邊框

            # 儲存圖片
            image_name = os.path.splitext(file_name)[0] + '.png'
            image_path = os.path.join(target_subfolder, image_name)
            plt.savefig(image_path, bbox_inches='tight', pad_inches=0)
            plt.close()

print('所有音檔已轉換並儲存成圖片')