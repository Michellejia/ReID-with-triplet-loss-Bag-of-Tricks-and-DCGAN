# -*- coding: utf-8 -*-
"""
Created on Sat May 16 20:00:18 2020

@author: Miche
"""
  
# Improting Image class from PIL module  
from PIL import Image  
  
# Opens a image in RGB mode  
im = Image.open(r"E:\\graduation thesis\\code modification\\triplet-reid-pytorch-master-BagofTricks(BNNeck + CenterLoss + Smoothing Research)\\triplet-reid-pytorch-master\\fake_image.png")  
  
# Size of the image in pixels (size of orginal image)  
# (This is not mandatory)  
width, height = im.size  
  
# Setting the points for cropped image  
#left = 4
#top = height / 5
#right = 154
#bottom = 3 * height / 5
  
# Cropped image of above dimension  
# (It will not change orginal image)  
#im1 = im.crop((left, top, right, bottom)) 
width1 = int(383)
height1 = int(383*2.5) 
im1 = im.resize((width1, height1)) 
# Shows the image in image viewer  
im1.show() 

