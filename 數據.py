import random
import os
import pickle

import cv2
import numpy as np

import koad

def 源():
    for 圖 in koad.遍歷讀圖('D:\數據集\yandere[370001~466745]\ori', 僅3通道=True):
        圖=koad.壓(圖,420)
        # cv2.imshow('',圖)
        # cv2.waitKey()
        for 圖64 in koad.切分(圖,64,dropout=0.5):
            yield 圖64

def 同隨機():
    a=np.random.randint(0,2,size=(4096))
    return a,a.copy()

class 器(koad.數據準備器):
    def __init__(self):
        super().__init__(
            數據源生成器 = 源(),
            數據處理函數 = lambda img: (img/256,*同隨機())
        )
    def 重整格式(self,l):
        l = super().重整格式(l)
        return [l[0],l[1]],[l[2]]



if __name__=='__main__':
    for x,y,z in 器().生成數據():
        # print(x,y,z)
        # print(x.shape,y.shape,z.shape)
        # input()
        # cv2.imshow('',x)
        print(x)
        print(y,y.var(),z.mean())
        print(z,z.var(),z.mean())
        # input()
        # cv2.waitKey()