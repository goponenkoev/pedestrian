#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
from PIL import Image, ImageDraw
from matplotlib import pyplot as plt
import numpy as np
import cv2, os, math

kernel = np.array([[1,1,1], [1,1,1], [1,1,1]])
kernel = 0.11 * kernel

kernelSharpen = np.array([[0,-1,0], [-1,5,-1], [0,-1,0]])
kernelBright = np.array([[-0.1,0.2,-0.1], [0.2,3,0.2], [-0.1,0.2,-0.1]])
kernelBlackout = np.array([[-0.1,0.1,-0.1], [0.1,0.5,0.1], [-0.1,0.1,-0.1]])
kernelBright = 0.5*kernelBright

def inside(r, q):
    h = 15
    rx, ry, rw, rh = r
    qx, qy, qw, qh = q
    return rx > qx - h and ry > qy - h and rx + rw < qx + qw + h and ry + rh < qy + qh + h

def equal(r, q):
    h = 35
    rx, ry, rw, rh = r
    qx, qy, qw, qh = q
    a = rx > (qx - h) and rx < (qx + h)
    b = ry > (qy - h) and ry < (qy + h)
    c = rw > (qw - h) and rw < (qw + h)
    d = rh > (qh - h) and rh < (qh + h)
    #print(a,b,c,d)
    return  a and b and c and d

def clear(final_found):
   for ri, r in enumerate(final_found):
        for qi, q in enumerate(final_found):
            if qi != ri and equal(r, q):
                final_found.pop(qi)
                break

def clearInside(final_found):
   for ri, r in enumerate(final_found):
        for qi, q in enumerate(final_found):
            if qi != ri and inside(r, q):
                final_found.pop(qi)
                break             

def draw_detections(img, rects, thickness = 1):
    for x, y, w, h in rects:
        # the HOG detector returns slightly larger rectangles than the real objects.
        # so we slightly shrink the rectangles to get a nicer output.
        pad_w, pad_h = int(0.15*w), int(0.05*h)
        cv2.rectangle(img, (x+pad_w, y+pad_h), (x+w-pad_w, y+h-pad_h), (0, 255, 0), thickness)

def draw_detections1(img, rects, angle, x0, y0, thickness = 1):
    for x, y, w, h in rects:
        # the HOG detector returns slightly larger rectangles than the real objects.
        # so we slightly shrink the rectangles to get a nicer output.
        pad_w, pad_h = int(0.15*w), int(0.05*h)
        print(x0,y0, angle)
        rx = x - x0
        ry = y - y0
        c = math.cos(math.radians(math.fabs(angle)))
        s = math.sin(math.radians(math.fabs(angle)))
        print(c,s)
        x1 = x0 + rx * c - ry * s
        y1 = y0 + rx * s + ry * c
        print('x1 ',x1,' y1 ',y1,' x ',x,' y ',y)
        if (x1 < 0):
            x1 = 0
        if (y1 < 0):
            y1 = 0
        cv2.rectangle(img, (int(x1)+pad_w, int(y1)+pad_h), (int(x1)+w-pad_w, int(y1)+h-pad_h), (255, 200, 0), thickness)

def plot(title, img, i):
    plt.subplot(2, 2, i)
    plt.title(title)
    plt.imshow(img, 'gray')
    plt.gca().get_xaxis().set_visible(False)
    plt.gca().get_yaxis().set_visible(False)

def withPicture1(path):
    image_paths = [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.jpg')]
    print(image_paths)
    for imgpath in image_paths:
        i = imgpath.rfind('/')
        if i == -1:
            break
        name = imgpath[(i+1):]
        src = cv2.imread(imgpath)
        affin = (15,10,0,-10,-15)
        final_found = []
        for i in range(len(affin)):
            img = cv2.copyMakeBorder(src,0,0,0,0,cv2.BORDER_REPLICATE);
            imgtest = cv2.copyMakeBorder(src,0,0,0,0,cv2.BORDER_REPLICATE);
            (h, w) = img.shape[:2]
            center = (w / 2, h / 2)
            M = cv2.getRotationMatrix2D(center, affin[i], 1.0)
            #print(M)
            img = cv2.warpAffine(img, M, (w, h))

            for j in range(1):
                img = cv2.filter2D(img, -1, kernel)  
            #img = cv2.filter2D(img, -1, kernelSharpen)  
            height, width = img.shape[:2]
            #res = cv2.resize(img,(2*width, 2*height), interpolation = cv2.INTER_CUBIC)
            
            found, w1 = hog.detectMultiScale(img, winStride=(8,8), padding=(16,16), scale=1.05)
            found_filtered = []
            
            for ri, r in enumerate(found):
                final_found.append(r) 
        
            draw_detections(img, found)
            cv2.imshow(str(i), img)
            pathWrite = path + 'res1/' + str(i) + str(name)
            cv2.imwrite(pathWrite, img)

            #draw_detections1(imgtest, found, affin[i], w, h)
            #cv2.imshow('test1', imgtest)
            ch = cv2.waitKey()

        #clear(final_found)
        clearInside(final_found)
        draw_detections(src, final_found)
        pathWrite = path + 'res1/v1' + str(name)
        cv2.imwrite(pathWrite, src)
        ch = cv2.waitKey()

def withPicture2(path):
    image_paths = [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.jpg')]
    print(image_paths)
    for imgpath in image_paths:
        i = imgpath.rfind('/')
        if i == -1:
            break
        name = imgpath[(i+1):]
        src = cv2.imread(imgpath)
        affin = (10,0,-10)
        #found_filtered = []
        final_found = []
        
        img = cv2.copyMakeBorder(src,0,0,0,0,cv2.BORDER_REPLICATE);

        for i in range(1):
            img = cv2.filter2D(img, -1, kernel) 
            #img = cv2.filter2D(img, -1, kernelBlackout)
            #img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
       
        found, w = hog.detectMultiScale(img, winStride=(8,8), padding=(16,16), scale=1.05)
        found_filtered = []
        
        for ri, r in enumerate(found):
            for qi, q in enumerate(found):
                if ri != qi and inside(r, q):
                    break
                else:
                    found_filtered.append(r)
        
        draw_detections(img, found)
        draw_detections(img, found_filtered, 1)
        
        cv2.imshow('frame',img)
        pathWrite = path + 'res2/v2' + str(name)
        cv2.imwrite(pathWrite, img)
        ch = cv2.waitKey()

def wuthVideo(path):
    #video

    cap = cv2.VideoCapture('./vtest.avi')

    if (cap.isOpened() == False): 
         print("Error opening video stream or file")

    while(cap.isOpened()):
        ret, img = cap.read()
        #img = cv2.resize(src, (640, 400))
        #height, width = img.shape[:2]
        #res = cv2.resize(img,(2*width, 2*height), interpolation = cv2.INTER_CUBIC)
        for i in range(1):
            img = cv2.filter2D(img, -1, kernel) 
            #img = cv2.filter2D(img, -1, kernelBlackout)
            #img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
       
        found, w = hog.detectMultiScale(img, winStride=(8,8), padding=(16,16), scale=1.05)
        found_filtered = []
        
        for ri, r in enumerate(found):
            for qi, q in enumerate(found):
                if ri != qi and inside(r, q):
                    break
                else:
                    found_filtered.append(r)
        
        draw_detections(img, found)
        draw_detections(img, found_filtered, 1)
        
        cv2.imshow('frame',img)
        #plt.imshow(img, 'frame')
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
if __name__ == '__main__':
    import sys
    from glob import glob
    import itertools as it

    print(__doc__)

    path = './'

    nbins = 9
    cellSize = (8, 8)
    blockSize = (16, 16)
    blockStride = (8, 8)
    winSize = (64, 128)
    winStride = (1, 1)

    hog = cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins)

    #hog = cv2.HOGDescriptor()
    hog.setSVMDetector( cv2.HOGDescriptor_getDefaultPeopleDetector())
    assert(hog.checkDetectorSize())

    #
    plt.subplot()
    plt.title('test')
    plt.gca().get_xaxis().set_visible(False)
    plt.gca().get_yaxis().set_visible(False)
    #
    withPicture2(path)
        
    cv2.destroyAllWindows()