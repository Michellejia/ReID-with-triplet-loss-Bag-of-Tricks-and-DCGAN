import scipy.io
import torch
import numpy as np
import time
import argparse
import os
import math
import pickle
import torchvision.transforms as transforms
import logging
import argparse
import sys
from torchvision import datasets
from utils import pdist_np as pdist

parser = argparse.ArgumentParser(description='evaluate')
parser.add_argument('--name',default='ft_ResNet50_market_pcb_r', type=str, help='0,1,2,3...or last')
parser.add_argument('--alpha', default=5, type=float, help='alpha')
parser.add_argument('--smooth', default=50, type=float, help='smooth')
opt = parser.parse_args()
name = opt.name
alpha=opt.alpha
smooth=opt.smooth

def parse_args():
    parse = argparse.ArgumentParser()
    parse.add_argument(
            '--gallery_embs',
            dest = 'gallery_embs',
            type = str,
            default = './res/emb_gallery.pkl',
            help = 'path to embeddings of gallery dataset'
            )
    parse.add_argument(
            '--query_embs',
            dest = 'query_embs',
            type = str,
            default = './res/emb_query.pkl',
            help = 'path to embeddings of query dataset'
            )
    parse.add_argument(
            '--cmc_rank',
            dest = 'cmc_rank',
            type = int,
            default = 1,
            help = 'path to embeddings of query dataset'
            )

    return parse.parse_args()

#######################################################################
# Evaluate
def evaluate(qf,ql,qc,qfr,gf,gl,gc,gfr,distribution):
    query = qf
    score = np.dot(gf,query)
    #score = pdist(gf, query)

    # spatial temporal scores: qfr,gfr, qc, gc
    # TODO
    interval = 100
    score_st = np.zeros(len(gc))
    for i in range(len(gc)):
        if qfr>gfr[i]:
            diff = qfr-gfr[i]
            hist_ = int(diff/interval)
            pr = distribution[qc-1][gc[i]-1][hist_]
        else:
            diff = gfr[i]-qfr
            hist_ = int(diff/interval)
            pr = distribution[gc[i]-1][qc-1][hist_]
        score_st[i] = pr

    # ========================
    score  = 1/(1+np.exp(-alpha*score))*1/(1+2*np.exp(-alpha*score_st))
    index = np.argsort(-score)  #from large to small 数组值从小到大的索引值

    #返回非0的数组元组的索引
    query_index = np.argwhere(gl==ql)
    camera_index = np.argwhere(gc==qc)

    #把在a中的但是不在b中的元素按a中的顺序排序，并且不合并重复的元素
    good_index = np.setdiff1d(query_index, camera_index, assume_unique=True)
    junk_index1 = np.argwhere(gl==-1)
    junk_index2 = np.intersect1d(query_index, camera_index)
    junk_index = np.append(junk_index2, junk_index1) #.flatten())
    CMC_tmp = compute_mAP(index, good_index, junk_index)
    return CMC_tmp


def compute_mAP(index, good_index, junk_index):
    ap = 0
    cmc = torch.IntTensor(len(index)).zero_()
    if good_index.size==0:   # if empty
        cmc[0] = -1
        return ap,cmc

    # remove junk_index
    mask = np.in1d(index, junk_index, invert=True)
    index = index[mask]

    # find good_index index
    ngood = len(good_index)
    mask = np.in1d(index, good_index)
    rows_good = np.argwhere(mask==True)
    rows_good = rows_good.flatten()
    
    cmc[rows_good[0]:] = 1
    for i in range(ngood):
        d_recall = 1.0/ngood
        precision = (i+1)*1.0/(rows_good[i]+1)
        if rows_good[i]!=0:
            old_precision = i*1.0/rows_good[i]
        else:
            old_precision=1.0
        ap = ap + d_recall*(old_precision + precision)/2

    return ap, cmc

def gaussian_func(x, u, o=50):
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


def gauss_smooth(arr):
    hist_num = len(arr)
    vect= np.zeros((hist_num,1))
    for i in range(hist_num):
        vect[i,0]=i
    # gaussian_vect= gaussian_func2(vect,0,1)
    gaussian_vect= gaussian_func2(vect,0,50)
    matrix = np.zeros((hist_num,hist_num))
    # matrix = np.eye(hist_num)
    for i in range(hist_num):
        for j in range(i,hist_num):
            matrix[i][j]=gaussian_vect[j-i]    
    matrix = matrix+matrix.transpose()
    for i in range(hist_num):
        matrix[i][i]=matrix[i][i]/2
    # for i in range(hist_num):
    #     for j in range(i):
    #         matrix[i][j]=gaussian_vect[j]     
    xxx = np.dot(matrix,arr)
    return xxx

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

######################################################################
# result = scipy.io.loadmat('pytorch_result.mat')
# result = scipy.io.loadmat('model/'+name+'/'+'pytorch_result.mat')
# query_feature = result['query_f']
# query_cam = result['query_cam'][0]
# query_label = result['query_label'][0]
# query_frames = result['query_frames'][0]


# gallery_feature = result['gallery_f']
# gallery_cam = result['gallery_cam'][0]
# gallery_label = result['gallery_label'][0]
# gallery_frames = result['gallery_frames'][0]

def get_id(img_path):
    camera_id = []
    labels = []
    frames = []
    pth = []
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
        pth.append(path)
    return camera_id, labels, frames, pth

data_transforms = transforms.Compose([
        transforms.Resize((288,144), interpolation=3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
############### Ten Crop        
        #transforms.TenCrop(224),
        #transforms.Lambda(lambda crops: torch.stack(
         #   [transforms.ToTensor()(crop) 
          #      for crop in crops]
           # )),
        #transforms.Lambda(lambda crops: torch.stack(
         #   [transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(crop)
          #       for crop in crops]
          # ))
])

image_datasets = {x: datasets.ImageFolder(os.path.join("E://graduation thesis//Spatial-Temporal-Re-identification-master//Spatial-Temporal-Re-identification-master//dataset//market_rename",x),data_transforms) for x in ['gallery','query']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=8,
                                             shuffle=False, num_workers=0) for x in ['gallery','query']}

class_names = image_datasets['query'].classes
use_gpu = torch.cuda.is_available()

gallery_path = image_datasets['gallery'].imgs
query_path = image_datasets['query'].imgs

gallery_cam,gallery_label, gallery_frames,gallery_pth = get_id(gallery_path)
query_cam,query_label, query_frames, query_pth = get_id(query_path)

import pandas as pd
gallery_pth = pd.DataFrame(gallery_pth)
query_pth = pd.DataFrame(query_pth)
gallery_pth.to_csv("gallery_pth.csv")
query_pth.to_csv("query_pth.csv")

gallery_frames = np.array(gallery_frames)
query_frames = np.array(query_frames)

## logging
FORMAT = '%(levelname)s %(filename)s:%(lineno)d: %(message)s'
logging.basicConfig(level=logging.INFO, format=FORMAT, stream=sys.stdout)
logger = logging.getLogger(__name__)
## load embeddings
logger.info('loading gallery embeddings')
with open('E:\\graduation thesis\\code modification\\triplet-reid-pytorch-master-BagofTricks(BNNeck + CenterLoss + Smoothing Research)\\res\\BNNeck(use this one)\\emb_gallery.pkl', 'rb') as fr:
    gallery_dict = pickle.load(fr)
    #emb_gallery, lb_ids_gallery, lb_cams_gallery = gallery_dict['embeddings'], gallery_dict['label_ids'], gallery_dict['label_cams']
    gallery_feature, gallery_label, gallery_cam = gallery_dict['embeddings'], gallery_dict['label_ids'], gallery_dict['label_cams']
logger.info('loading query embeddings')
with open('E:\\graduation thesis\\code modification\\triplet-reid-pytorch-master-BagofTricks(BNNeck + CenterLoss + Smoothing Research)\\res\\BNNeck(use this one)\\emb_query.pkl', 'rb') as fr:
    query_dict = pickle.load(fr)
    #emb_query, lb_ids_query, lb_cams_query = query_dict['embeddings'], query_dict['label_ids'], query_dict['label_cams']
    query_feature, query_label, query_cam = query_dict['embeddings'], query_dict['label_ids'], query_dict['label_cams']

query_feature=query_feature.transpose()/np.power(np.sum(np.power(query_feature,2),axis=1),0.5)
query_feature=query_feature.transpose()
print('query_feature:',query_feature.shape)
gallery_feature=gallery_feature.transpose()/np.power(np.sum(np.power(gallery_feature,2),axis=1),0.5)
gallery_feature=gallery_feature.transpose()
print('gallery_feature:',gallery_feature.shape)

# query_feature=query_feature.transpose()/np.power(np.sum(np.power(query_feature,2),axis=1),0.5)
# query_feature=query_feature.transpose()
# print('query_feature:',query_feature.shape)
# gallery_feature=gallery_feature.transpose()/np.power(np.sum(np.power(gallery_feature,2),axis=1),0.5)
# gallery_feature=gallery_feature.transpose()
# print('gallery_feature:',gallery_feature.shape)


#############################################################

result2 = scipy.io.loadmat('E:\\graduation thesis\\code modification\\triplet-reid-pytorch-master-BagofTricks(BNNeck + CenterLoss + Smoothing Research)\\res\\BNNeck(use this one)\\pytorch_result2.mat')
distribution = result2['distribution']

############################################################
for i in range(0,8):
    for j in range(0,8):
        print("gauss "+str(i)+"->"+str(j))
        # gauss_smooth(distribution[i][j])
        distribution[i][j] = gauss_smooth2(distribution[i][j], smooth)


eps = 0.0000001
sum_ = np.sum(distribution,axis=2)
for i in range(8):
    for j in range(8):
        distribution[i][j][:]=distribution[i][j][:]/(sum_[i][j]+eps)        
#############################################################

CMC = torch.IntTensor(len(gallery_label)).zero_()
ap = 0.0
#print(query_label)
for i in range(len(query_label)):
    ap_tmp, CMC_tmp = evaluate(query_feature[i],query_label[i],query_cam[i],query_frames[i], gallery_feature,gallery_label,gallery_cam,gallery_frames,distribution)
    if CMC_tmp[0]==-1:
        continue
    CMC = CMC + CMC_tmp
    ap += ap_tmp
    print(i, CMC_tmp[0])
    # if i%10==0:
    #     print('i:',i) 

CMC = CMC.float()
CMC = CMC/len(query_label) #average CMC
print('top1:%f top5:%f top10:%f mAP:%f'%(CMC[0],CMC[4],CMC[9],ap/len(query_label)))
print('alpha,smooth:',alpha,smooth)

result = {'CMC':CMC.numpy()}

scipy.io.savemat('model/'+'CMC_duke_two_stream_add5.mat',result)


