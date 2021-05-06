import scipy.io
import torch
import numpy as np
import time
from  re_ranking import re_ranking
import argparse
import os
import math
import logging
import sys
import pickle

parser = argparse.ArgumentParser(description='evaluate')
parser.add_argument('--name',default='ft_ResNet50_duke_pcb_r_c', type=str, help='0,1,2,3...or last')
opt = parser.parse_args()
name = opt.name

#######################################################################
# Evaluate
def evaluate(score,ql,qc,gl,gc):
    index = np.argsort(score)  #from small to large
    #index = index[::-1]
    # good index
    query_index = np.argwhere(gl==ql)
    camera_index = np.argwhere(gc==qc)

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

######################################################################
# result = scipy.io.loadmat('E:\\graduation thesis\\code modification\\triplet-reid-pytorch-master-BagofTricks(BNNeck + CenterLoss + Smoothing Research)\\model\\pytorch_result.mat')
# query_feature = result['query_f']
# query_cam = result['query_cam'][0]
# query_label = result['query_label'][0]
# gallery_feature = result['gallery_f']
# gallery_cam = result['gallery_cam'][0]
# gallery_label = result['gallery_label'][0]

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

mat_path = 'E:\\graduation thesis\\code modification\\triplet-reid-pytorch-master-BagofTricks(BNNeck + CenterLoss + Smoothing Research)\\model\\all_scores.mat'
all_scores = scipy.io.loadmat(mat_path)                   #important
all_dist =  all_scores['all_scores']
print('all_dist shape:',all_dist.shape)
print('query_cam shape:',query_cam.shape)

CMC = torch.IntTensor(len(gallery_label)).zero_()
ap = 0.0
#re-ranking
print('calculate initial distance')
# q_g_dist = np.dot(query_feature, np.transpose(gallery_feature))
# q_q_dist = np.dot(query_feature, np.transpose(query_feature))
# g_g_dist = np.dot(gallery_feature, np.transpose(gallery_feature))

since = time.time()
re_rank = re_ranking(len(query_cam), all_dist)
time_elapsed = time.time() - since
print('Reranking complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
for i in range(len(query_label)):
    ap_tmp, CMC_tmp = evaluate(re_rank[i,:],query_label[i],query_cam[i],gallery_label,gallery_cam)
    if CMC_tmp[0]==-1:
        continue
    CMC = CMC + CMC_tmp
    ap += ap_tmp
    #print(i, CMC_tmp[0])

CMC = CMC.float()
CMC = CMC/len(query_label) #average CMC
print('top1:%f top5:%f top10:%f mAP:%f'%(CMC[0],CMC[4],CMC[9],ap/len(query_label)))
