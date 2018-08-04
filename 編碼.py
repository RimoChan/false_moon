import struct
import numpy
import zlib
import lzma
import random
from reedsolo import RSCodec

RS器=RSCodec(100)

def 二化(s):
    b=s.encode('utf32')
    壓縮=lzma.compress(b)
    rs=RS器.encode(壓縮)
    al=''.join([bin(b)[2:].zfill(8) for b in rs])
    return numpy.array([int(i) for i in al])

def 字化(s2):
    s2=[str(int(i>0.5)) for i in s2]
    al=''.join(s2)
    s=[]
    for i in range(0,len(al),8):
        s.append(int(al[i:i+8],base=2))
    b=bytes(s)
    解rs=RS器.decode(s)
    解壓縮=lzma.decompress(解rs)
    字=解壓縮.decode('utf32')
    return 字

def test():
    with open('test/灰姑娘.txt',encoding='utf8') as f:
        s=f.read()
    
    s2=二化(s)
    print(len(s2))
    for i in range(len(s)):
        if random.random()<1/100:
            s2[i]=1-s2[i]
    
    with open('t.txt','w',encoding='utf8') as f:
        f.write(字化(s2))

if __name__=='__main__':
    for i in range(250):
        print(i,end=' ')
        try:
            test()
        except Exception as e:
            print('fail')