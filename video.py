#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 30 22:43:53 2019

@author: Uday Rallabhandi
"""

import cv2

#image parameters
folder_path=r'D:/Steve/GiaHanXinhDep/data/' #path to folder containing images
number_of_images=158

#create list of image objects
images=[]
for i in range(number_of_images):
    image_name="gs_%02d.png" % i #makes a string of length 3 with 0 as padding character
    images.append(cv2.imread(folder_path+image_name)) #creates the image object

#video parameters
fps=12 #can have this be num_images/time wanted
height,width,layers=images[0].shape #gets uploaded image size (shape of representing numpy array)
size=(width, height) #image size must match video initalization
filename='f035k060.avi' #only writes .avi files
fourcc = cv2.VideoWriter_fourcc('D','I','V','X') #windows fourCC code (for compression format)

#create video object
video=cv2.VideoWriter(filename, fourcc, fps, size)

#write the images to the video object
for frame in images:
    video.write(frame) #write method appends the frames together in the desired fps
    
video.release() #closes the video file
