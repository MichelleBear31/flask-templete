# -*- coding: utf-8 -*-
import cv2
import numpy as np
import os

# Hash值对比
def cmpHash(hash1, hash2, shape=(10, 10)):
    n = 0
    # hash长度不同则返回-1代表传参出错
    if len(hash1) != len(hash2):
        return -1
    # 遍历判断
    for i in range(len(hash1)):
        # 相等则n计数+1，n最终为相似度
        if hash1[i] == hash2[i]:
            n = n + 1
    return n / (shape[0] * shape[1])

# 差值感知算法
def dHash(img, shape=(10, 10)):
    # 缩放10*11
    img = cv2.resize(img, (shape[0] + 1, shape[1]))
    # 转换灰度图
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hash_str = ''
    # 每行前一个像素大于后一个像素为1，相反为0，生成哈希
    for i in range(shape[0]):
        for j in range(shape[1]):
            if gray[i, j] > gray[i, j + 1]:
                hash_str = hash_str + '1'
            else:
                hash_str = hash_str + '0'
    return hash_str

# 均值哈希算法
def aHash(img, shape=(10, 10)):
    # 缩放为10*10
    img = cv2.resize(img, shape)
    # 转换为灰度图
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # s为像素和初值为0，hash_str为hash值初值为''
    s = 0
    hash_str = ''
    # 遍历累加求像素和
    for i in range(shape[0]):
        for j in range(shape[1]):
            s = s + gray[i, j]
    # 求平均灰度
    avg = s / 100
    # 灰度大于平均值为1相反为0生成图片的hash值
    for i in range(shape[0]):
        for j in range(shape[1]):
            if gray[i, j] > avg:
                hash_str = hash_str + '1'
            else:
                hash_str = hash_str + '0'
    return hash_str

# 感知哈希算法(pHash)
def pHash(img, shape=(10, 10)):
    # 缩放32*32
    img = cv2.resize(img, (32, 32))  # , interpolation=cv2.INTER_CUBIC

    # 转换为灰度图
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 将灰度图转为浮点型，再进行dct变换
    dct = cv2.dct(np.float32(gray))
    # opencv实现的掩码操作
    dct_roi = dct[0:10, 0:10]

    hash = []
    avreage = np.mean(dct_roi)
    for i in range(dct_roi.shape[0]):
        for j in range(dct_roi.shape[1]):
            if dct_roi[i, j] > avreage:
                hash.append(1)
            else:
                hash.append(0)
    return hash

def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    img1_path = os.path.join(current_dir, 'VGG_MobileNet_mfcc_images', '1', '1.png')
    img2_path = os.path.join(current_dir, 'VGG_MobileNet_mfcc_images', '2', '3.png')
    print(img1_path,img2_path)
    # Read images
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)

    # Ensure images are read correctly
    if img1 is None or img2 is None:
        print("Error: One or both images could not be read. Check file paths.")
        return

    # Calculate hashes and compare
    hash1 = dHash(img1)
    hash2 = dHash(img2)
    n = cmpHash(hash1, hash2)
    print('差值哈希算法相似度：', n)

    hash1 = aHash(img1)
    hash2 = aHash(img2)
    n = cmpHash(hash1, hash2)
    print('均值哈希算法相似度：', n)

    hash1 = pHash(img1)
    hash2 = pHash(img2)
    n = cmpHash(hash1, hash2)
    print('感知哈希算法相似度：', n)

if __name__ == "__main__":
    main()