import os
import sys
import numpy as np
import keras
from keras.callbacks import *
import keras.backend as K

import 數據
import 模型

def 測試(model,path=''):
    import cv2
    g_test=數據.生成數據組(批大小=20)
    x,y=next(g_test)

    if path:
        try:
            os.mkdir(path)
        except:
            None

    for n,圖 in enumerate(model.predict(x)):
        圖=(圖*255).astype(np.uint8)
        cv2.imwrite(path+'%s.png'%n, 圖)
    for n,圖 in enumerate(x):                                                             
        圖=(圖*255).astype(np.uint8)
        cv2.imwrite(path+'%s_ori.png'%n, 圖)
    
# class 存檔(keras.callbacks.Callback):
#     def on_train_begin(self, logs={}):
#         self.記錄點=[0,0.005,0.006,0.007,0.008,0.009,0.01,0.012,0.015,0.02,0.03,0.05,0.1]

#     def on_epoch_end(self, batch, logs={}):
#         存檔(self.model)

#——————————————————————
def 訓練(model):
    g=數據.器().生成數據組(批大小=16)
    回調=[
        ModelCheckpoint('model.h5',save_weights_only=True,monitor='loss',save_best_only=True),
        ReduceLROnPlateau(monitor='loss', factor=0.93, patience=6),
    ]
    model.fit_generator(g,800,callbacks=回調,epochs=99999)
    # model.fit_generator(g,500,epochs=99999)

def 存檔(model):
    try:
        os.mkdir('model')
    except:
        None
    model.save_weights('model.h5')

def 讀檔():
    全模型,嵌入,解碼=模型.製造模型()
    if not os.path.isfile('model.h5'):
        print('重新訓練')
    else:
        全模型.load_weights('model.h5', by_name=True)
        print('繼續訓練')
    return 全模型

if __name__=='__main__':
    model=讀檔()
    訓練(model)
