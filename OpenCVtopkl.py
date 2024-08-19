import cv2
import numpy as np
import pickle
import os
# 讀取圖片
current_dir = os.path.dirname(os.path.abspath(__file__))
image_path = os.path.join(current_dir,'PKL2image.png')
image = cv2.imread(image_path, cv2.IMREAD_COLOR)

# 將圖片轉換為灰度格式
image_array = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 將圖片資料轉換為一維 numpy array
data_array = image_array.flatten()

# 提取出有效的資料部分
data_bytes = data_array.tobytes()

# 將 bytes 轉換回原始資料
data_restored = pickle.loads(data_bytes)

# 確保還原的資料與原始資料一致
restored_file_path = os.path.join(current_dir,'PKL','2.pkl')
with open(restored_file_path, 'wb') as f:
    pickle.dump(data_restored, f)

print(f"Data has been restored and saved to {restored_file_path}.")