# -*- coding: UTF-8 -*-
'''
@File    ：CarDect.py
@IDE     ：PyCharm 
@Author  ：LiJun
@Date    ：2022/11/23 15:33 
'''
import cv2
import imutils
import numpy as np
import pytesseract

img = cv2.resize(cv2.imread("img.png"), (900, 700))


# cv2读取图片通道默认为BGR，转换为GRAY灰度图
grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

'''
 cv2.GaussianBlur：高斯滤波（平滑处理图像，同时更多保留图像的总体灰度分布特征）
 待处理图像，滤波核大小（处理过程中其领域图像的高度和宽度，它的核值为奇数），x轴和y轴的标准差
 @P1：传入图像，@P2：滤波和大小，@P3：x轴标准差，@P4：y轴标准差
'''

guassianImg = cv2.GaussianBlur(grayImg, (3, 3), 0, 0)

# 边缘检测，低于x1的像素点不会被认为是边缘，高于x2的像素点会被认为是边缘；若在x1与x2之间的像素点是x2像素点的相邻则被认为是边缘
# @P1：传入图片，@P2：x1，@P3：x2
cannyImg = cv2.Canny(guassianImg, 100, 200)


'''
 cv2.findContours：图像上寻找轮廓
 参数：
 @P1：传入二值化图像，@P2：轮廓检索方式，一般是检测外轮廓cv2.RETR_EXTERNAL，@P3：轮廓的近似方式，cv2.CHAIN_APPROX_SIMPLE为保留轮廓的终点坐标，近似矩阵的左上角顶点坐标及宽高
 返回值：
 cnts：轮廓的点集列表，存储图像中每个轮廓的数组，hierarchy：存储轮廓之间的层级关系，是一个N*M大小的矩阵，N是轮廓数量，M固定等于4
'''
cnts = cv2.findContours(cannyImg, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# 返回cnts中的countors（轮廓）
countors = imutils.grab_contours(cnts)

'''
 对countors进行排序，排序的条件为cv2.contoureArea函数，从大到小取前十个
 cv2.contoureArea函数：计算轮廓的面积
'''
countors = sorted(countors, key=cv2.contourArea, reverse=True)[:10]

# 创建车牌定位列表
screenCnt = None

print(countors)
# 遍历轮廓列表
for c in countors:

    '''
     @P1:输入的向量，二维点；@P2：指示曲线是否封闭的标识符，一般设置为True
     cv2.arcLength：用于计算封闭轮廓的周长或曲线的长度
    '''
    peri = cv2.arcLength(c, True)

    '''
     cv2.approxPolyDP：输入一组曲线点集合，输出折线点集合，大概意思就是将曲折的线条变成平滑的线条
     @P1：输入的点集，@P2：指定精度，原始曲线与近似曲线之间的最大距离，@P3：若为True，则表示近似曲线是闭合的，首尾相连
    '''
    approx = cv2.approxPolyDP(c, 0.018 * peri, True)

    # 如果有四个端点，则可以确定其为车牌的形状
    if len(approx) == 4:
        # 将当前列表赋值給车牌定位列表
        screenCnt = approx
        break

# 如果检测车牌列表为空，则将detected元素赋0
if screenCnt is None:
    detected = 0
    print("None!")
else:
    # 反之为1
    detected = 1

'''
 cv2.drawContours：轮廓填充
 @P1：目标图像，@P2：轮廓坐标，@P3：轮廓索引，负数绘制所有轮廓，@P4：轮廓颜色，@P5：轮廓线条宽度，负数填充轮廓内部
'''
if detected:
    recognizedImg = cv2.drawContours(img, [screenCnt], 0, (0, 0, 255), 2)
    cv2.imshow("rec", recognizedImg)


# 创建一个相同尺寸的黑色背景图
mask = np.zeros(grayImg.shape, np.uint8)
cv2.drawContours(mask, [screenCnt], -1, 255, -1)
'''
 cv2.bitwise_and：提取图像为，掩膜在输入图像上的图像
 @P1，@P2：输入图像，@P3:掩膜
'''
splitImg = cv2.bitwise_and(img, img, mask=mask)

# 获得掩膜为255的x，y两坐标
x, y = np.where(mask == 255)
# print(x, y)
# 获得
topX, topY = np.min(x), np.min(y)
# print(topX, topY)
bottomX, bottomY = np.max(x), np.max(y)
# 获取灰度图的车牌位置图片
croppedImg = grayImg[topX: bottomX+1, topY:bottomY+1]

# https://blog.csdn.net/qq_31362767/article/details/107891185
# 在灰度图上进行OCR识别获得车牌号文本
text = pytesseract.image_to_string(croppedImg, config='--psm 11')
print("Detected license plate Number is:", text)
cv2.imshow('cropped', croppedImg)
# cv2.imshow('mask', mask)
# cv2.imshow('split', splitImg)
# cv2.imshow('canny', cannyImg)
# cv2.imshow('gray', grayImg)
# cv2.imshow("img", img)

cv2.waitKey()
cv2.destroyAllWindows()
