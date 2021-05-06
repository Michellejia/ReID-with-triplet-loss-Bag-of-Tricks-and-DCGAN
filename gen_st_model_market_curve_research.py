# -*- coding: utf-8 -*-

from __future__ import print_function, division

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
from torchvision import datasets
import os
import scipy.io
import math
######################################################################
# Options
# --------
parser = argparse.ArgumentParser(description='Training')
parser.add_argument('--data_dir',default="E://graduation thesis//Spatial-Temporal-Re-identification-master//Spatial-Temporal-Re-identification-master//dataset//market_rename",type=str, help='./train_data')
parser.add_argument('--name', default='ft_ResNet50_market_rename_pcb', type=str, help='save model path')

opt = parser.parse_args()
name = opt.name
data_dir = opt.data_dir


def get_id(img_path):
    camera_id = []
    labels = []
    frames = []
    for path, v in img_path:
        filename = path.split('\\')[-1]
        label = filename[0:4]
        camera = filename.split('c')[1]
        # frame = filename[9:16]
        frame = filename.split('_')[2][1:]
        if label[0:2]=='-1':
            labels.append(-1)
        else:
            labels.append(int(label))
        camera_id.append(int(camera[0]))
        frames.append(int(frame))
    return camera_id, labels, frames

#labels均为0-750
def spatial_temporal_distribution(camera_id, labels, frames):
    class_num=751
    max_hist = 5000
    spatial_temporal_sum = np.zeros((class_num,8))                       
    spatial_temporal_count = np.zeros((class_num,8))
    eps = 0.0000001
    interval = 100.0
    
    for i in range(len(camera_id)):
        label_k = labels[i]                 #### not in order, done
        cam_k = camera_id[i]-1              ##### ### ### ### ### ### ### ### ### ### ### ### # from 1, not 0
        frame_k = frames[i]
        #每个label对应出现在某个相机的frame(时间)
        spatial_temporal_sum[label_k][cam_k]=spatial_temporal_sum[label_k][cam_k]+frame_k
        #每个label出现在某个相机当中的次数
        spatial_temporal_count[label_k][cam_k] = spatial_temporal_count[label_k][cam_k] + 1
    spatial_temporal_avg = spatial_temporal_sum/(spatial_temporal_count+eps)          # spatial_temporal_avg: 751 ids, 8cameras, center point
    
    distribution = np.zeros((8,8,max_hist))
    for i in range(class_num):
        for j in range(8-1):
            for k in range(j+1,8):
                if spatial_temporal_count[i][j]==0 or spatial_temporal_count[i][k]==0:
                    continue 
                st_ij = spatial_temporal_avg[i][j]
                st_ik = spatial_temporal_avg[i][k]
                if st_ij>st_ik:
                    diff = st_ij-st_ik
                    hist_ = int(diff/interval)
                    distribution[j][k][hist_] = distribution[j][k][hist_]+1     # [big][small]
                else:
                    diff = st_ik-st_ij
                    hist_ = int(diff/interval)
                    distribution[k][j][hist_] = distribution[k][j][hist_]+1
    
    #归一化了一下，让每条包络和为1
    sum_ = np.sum(distribution,axis=2)
    for i in range(8):
        for j in range(8):
            distribution[i][j][:]=distribution[i][j][:]/(sum_[i][j]+eps)
    
    return distribution                    # [to][from], to xxx camera, from xxx camera

def gaussian_func(x, u, o=0.1):
    if (o == 0):
        print("In gaussian, o shouldn't equel to zero")
        return 0
    temp1 = 1.0 / (o * math.sqrt(2 * math.pi))
    temp2 = -(math.pow(x - u, 2)) / (2 * math.pow(o, 2))
    return temp1 * math.exp(temp2)

def gaussian_func2(x, u, o=50):
    temp1 = 1.0 / (o * math.sqrt(2 * math.pi))
    temp2 = -(np.power(x - u, 2)) / (2 * np.power(o, 2))
    return temp1 * np.exp(temp2)

# def gauss_smooth(arr):
#     print(gaussian_func(0,0))
#     for u, element in enumerate(arr):
#         print(u," ",element)
#         if element != 0:
#             for index in range(0, 3000):
#                 arr[index] = arr[index] + element * gaussian_func(index, u)

#     sum = 0
#     for v in arr:
#         sum = sum + v
#     if sum==0:
#         return arr
#     for i in range(0,3000):
#         arr[i] = arr[i] / sum
#     return arr

# class SavGol(object):
#     def __init__(self, window_size=11, rank=2):
#         assert window_size % 2 == 1
#         self.window_size = window_size
#         self.rank = rank

#         self.size = int((self.window_size - 1) / 2)
#         self.mm = self.create_matrix(self.size)
#         self.data_seq = []

#     def create_matrix(self, size):
#         line_seq = np.linspace(-size, size, 2*size+1)
#         #rank_seqs: [array([1., 1., 1., 1..., 1., 1.]), array([-5., -4., -3.... 4.,  5.])]
#         rank_seqs = [line_seq**j for j in range(self.rank)]
#         #rank_seqs变array为矩阵
#         rank_seqs = np.mat(rank_seqs)
#         kernel = (rank_seqs.T * (rank_seqs * rank_seqs.T).I) * rank_seqs
#         mm = kernel[self.size].T
#         return mm

#     def update(self, data):
#         self.data_seq.append(data)
#         #self.data_seq = data
#         if len(self.data_seq) > self.window_size:
#             del self.data_seq[0]
#         padded_data = self.data_seq.copy()
#         if len(padded_data) < self.window_size:
#             left = int((self.window_size-len(padded_data))/2)
#             right = self.window_size-len(padded_data)-left
#             for i in range(left):
#                 padded_data.insert(0, padded_data[0])
#             for i in range(right):
#                 padded_data.insert(
#                     len(padded_data), padded_data[len(padded_data)-1])
#         return (np.mat(padded_data)*self.mm).item()

"""
* 创建系数矩阵X
* size - 2×size+1 = window_size
* rank - 拟合多项式阶次
* x - 创建的系数矩阵
"""
def create_x(size, rank):
    x = []
    for i in range(2 * size + 1):
        m = i - size
        row = [m**j for j in range(rank)]
        x.append(row) 
    x = np.mat(x)
    return x

"""
 * Savitzky-Golay平滑滤波函数
 * data - list格式的1×n纬数据
 * window_size - 拟合的窗口大小
 * rank - 拟合多项式阶次
 * ndata - 修正后的值
"""
def savgol(data, window_size, rank):
    m = (window_size - 1) / 2
    odata = data[:]
    # 处理边缘数据，首尾增加m个首尾项
    m = int(m)
    for i in range(m):
        odata = np.insert(odata, 0, odata[0])
        odata = np.insert(odata, len(odata), odata[len(odata)-1])
    # 创建X矩阵
    x = create_x(m, rank)
    # 计算加权系数矩阵B
    b = (x * (x.T * x).I) * x.T
    a0 = b[m]
    a0 = a0.T
    # 计算平滑修正后的值
    ndata = []
    for i in range(len(data)):
        y = [odata[i + j] for j in range(window_size)]
        y1 = np.mat(y) * a0
        y1 = float(y1)
        ndata.append(y1)
    return ndata

# faster gauss_smooth
def gauss_smooth2(arr,o):
    hist_num = len(arr)
    vect= np.zeros((hist_num,1))
    for i in range(hist_num):
        vect[i,0]=i
    # gaussian_vect= gaussian_func2(vect,0,1)
    # o=50
    approximate_delta = 3*o     #  when x-u>approximate_delta, e.g., 6*o, the gaussian value is approximately equal to 0.
    gaussian_vect= gaussian_func2(vect,0,o)
    matrix = np.zeros((hist_num,hist_num))
    for i in range(hist_num):
        k=0
        for j in range(i,hist_num):
            if k>approximate_delta:
                continue
            matrix[i][j]=gaussian_vect[j-i] 
            k=k+1  
    matrix = matrix+matrix.transpose()
    for i in range(hist_num):
        matrix[i][i]=matrix[i][i]/2
    # for i in range(hist_num):
    #     for j in range(i):
    #         matrix[i][j]=gaussian_vect[j]     
    xxx = np.dot(matrix,arr)
    return xxx


transform_train_list = [
        transforms.Resize(144, interpolation=3),
        transforms.RandomCrop((256,128)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]


image_datasets = {x: datasets.ImageFolder( os.path.join(data_dir,x) ,transform_train_list) for x in ['train_all']}
train_path = image_datasets['train_all'].imgs
train_cam, train_label, train_frames = get_id(train_path)

train_label_order = []
for i in range(len(train_path)):
    train_label_order.append(train_path[i][1]) 


# distribution = spatial_temporal_distribution(train_cam, train_label, train_frames)
distribution = spatial_temporal_distribution(train_cam, train_label_order, train_frames)

#for i in range(0,8):
#    for j in range(0,8):
#        print("gauss "+str(i)+"->"+str(j))
#        distribution[i][j] = gauss_smooth2(distribution[i][j],1)

#sg = SavGol()
for i in range(0,8):
    for j in range(0,8):
        print("gauss "+str(i)+"->"+str(j))
        distribution[i][j] = savgol(distribution[i][j], 11, 4)

import matplotlib.pyplot as plt
x1 = np.arange(0, 5000)
plt.figure(figsize=(10,5))

from scipy.interpolate import make_interp_spline
xnew1 = np.linspace(x1.min(),x1.max(),5000) #300 represents number of points to make between T.min and T.max
for i in range(1,8):
    y1 = distribution[0][i][:]
    power_smooth1 = make_interp_spline(x1,y1)(xnew1)
    plt.plot(xnew1,power_smooth1,label = 'cam1_to_cam{}'.format(i+1))
 
#x2 = np.arange(-5000,0)
#xnew2 = np.linspace(x2.min(),x2.max(),5000) 
#for i in range(1,8):
#   y2 = distribution[i][0][:]
#   power_smooth2 = make_interp_spline(x2,y2)(xnew2)
#   plt.plot(xnew2,power_smooth2,label = 'cam1_to_cam{}'.format(i+1))
plt.legend(loc='upper right', frameon=True)
plt.xlabel("Time Interval") #xlabel、ylabel：分别设置X、Y轴的标题文字。
plt.ylabel("Frequency")
plt.title("Spatial-Temporal Model") # title：设置子图的标题。
plt.xlim(-5,100)
plt.ylim(0,0.23)# xlim、ylim：分别设置X、Y轴的显示范围。
   

#result = {'distribution':distribution}
#scipy.io.savemat('model/'+name+'/'+'pytorch_result2.mat',result)
