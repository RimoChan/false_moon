import 模型
import 編碼
import cv2
import numpy as np
import random

def shuffle(s):
    random.seed(3)
    l=len(s)
    for _ in range(2*l):
        x=random.randint(0,l-1)
        y=random.randint(0,l-1)
        s[x],s[y]=s[y],s[x]
def deshuffle(s):
    random.seed(3)
    l=len(s)
    q=[]
    for _ in range(2*l):
        x=random.randint(0,l-1)
        y=random.randint(0,l-1)
        q.append((x,y))
    for x,y in q[::-1]:
        s[x],s[y]=s[y],s[x]


def imwrite(name,img):
    img=np.minimum(img, 1.0)
    img=np.maximum(img, 0.0)
    img=(img*255).astype(np.uint8)
    cv2.imwrite(name,img)

全模型,嵌入,解碼=模型.製造模型()
全模型.load_weights('model.h5')
######################

img=cv2.imread('test/kazeno.png')/256
img=(img-0.5)+0.5
r,c,_=img.shape
容量 = 4096*(r//64)*(c//64)
print('可用容量:',容量)
img2=img.copy()
r,c=img2.shape[:2]
l=64
with open('test/灰姑娘.txt',encoding='utf8') as f:
    s=f.read()
b字=編碼.二化(s)
print('嵌入容量:',len(b字))
b字=np.concatenate([b字,np.zeros(shape=(容量-len(b字),))])
s2=b字.copy()
shuffle(s2)
# s2=np.zeros(10)
for x in range(0,r-l,l):
    for y in range(0,c-l,l):
        roi = img2[x:x+l,y:y+l]
        yao = s2[:4096]
        # if len(yao)<4096:
        #     yao=np.concatenate([yao,np.zeros(shape=(4096-len(yao),))])

        roi = 嵌入.predict([
            roi.reshape([1,64,64,3]),
            yao.reshape([1,4096])
        ])
        s2=s2[4096:]
        img2[x:x+l,y:y+l] = roi.reshape([64,64,3])
d=0.5+(img2-img)*80
imwrite('test/d.png',0.5+(img2-img)*20)
imwrite('test/kazeno2.png',img2)

######################

img2=cv2.imread('test/kazeno2.png')/256
a=[]
for x in range(0,r-l,l):
    for y in range(0,c-l,l):
        roi = img2[x:x+l,y:y+l]
        a.append(解碼.predict(roi.reshape([1,64,64,3])))

a=np.concatenate(a).flatten()
deshuffle(a)

print(((b字>0.5)==(a>0.5)).sum()/len(b字))

with open('t.txt','w',encoding='utf8') as f:
    f.write(編碼.字化(a))