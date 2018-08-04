import numpy as np
from keras.models import *
from keras.layers import *
from keras.layers.local import *
from keras.optimizers import * 
from keras.utils import plot_model

def BN():
    return BatchNormalization()

def relu():
    return PReLU(alpha_initializer='ones',shared_axes=[1,2])

def mult_res(inputs,n=0,bn=True):
    x=inputs
    for i in range(n):
        x=res(x,bn)
    if bn and n==0: 
        x=BN()(x)
    return x

def res(inputs,bn=True):
    n=int(inputs.shape[-1])
    
    # x=CoordinateChannel2D()(inputs)
    x=(inputs)

    c1=relu()(Conv2D(n,3, padding='same')(x))
    c2=relu()(Conv2D(n,3, padding='same')(c1))
    m=add([c2,inputs])
    if bn:
        m=BN()(m)
    return m

def down(t,inputs,bn=True):
    a=Conv2D(t,5,strides=2, padding='same')(inputs)
    c1=relu()(a)
    return (mult_res(c1,bn=bn))

def up(t,inputs,bn=True):
    c1=(relu()(Conv2DTranspose(t,5,strides=2, padding='same')(inputs)))
    return (mult_res(c1,bn=bn))

def 圖像卷積(inputs):
    c32 = down(32,inputs)
    c16 = down(64,c32)
    c8 = down(128,c16)
    c4 = down(256,c8,bn=False)

    return c4

def 圖像卷積模型():
    i=Input(shape=(64,64,3))
    c32 = down(32,i)
    c16 = down(64,c32)
    c8 = down(128,c16)
    c4 = down(256,c8,bn=False)
    o = Activation('sigmoid')(Flatten()(c4))
    return Model(inputs=i,outputs=o)

def 反卷到水印(inputs,水印強度=0.016,噪聲強度=1/1024):

    c8 = up(256,inputs)
    c16 = up(128,c8)
    c32 = up(64,c16)
    c64 = up(32,c32,bn=False)

    predict = Activation('sigmoid')(Conv2D(3,1)(c64))

    l=Lambda(lambda x: x*水印強度)(predict)

    return GaussianNoise(噪聲強度)(l)

def 製造模型():
    圖輸入 = Input(shape=(64,64,3))
    碼輸入 = Input(shape=(64*64,))
    特徵 = 圖像卷積(圖輸入)
    
    特徵和碼 = concatenate([特徵,Reshape([4,4,256])(碼輸入)])
    水印 = 反卷到水印(特徵和碼)
    
    終圖 = add([圖輸入,水印])

    解碼 = 圖像卷積模型()
    解碼的碼 = 解碼(終圖)

    全模型 = Model(inputs=[圖輸入,碼輸入],outputs=解碼的碼)
    嵌入 = Model(inputs=[圖輸入,碼輸入],outputs=終圖)
    全模型.summary()

    adam = Adam(lr=1e-4,amsgrad=True)
    rms = RMSprop(lr=4e-4)
    全模型.compile(loss='binary_crossentropy',optimizer=adam, metrics=['binary_accuracy','mse'])

    plot_model(全模型, to_file='model.png', show_shapes=True, show_layer_names=True)
    plot_model(解碼, to_file='model2.png', show_shapes=True, show_layer_names=True)
    return 全模型, 嵌入, 解碼

if __name__=='__main__':
    製造模型()