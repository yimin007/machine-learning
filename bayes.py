import torch
import torchvision
import numpy as np
from collections import Counter
import pandas as pd
import struct
import matplotlib.pyplot as plt
from PIL import Image
import math


# 训练集文件
train_images_idx3_ubyte_file = r'D:\notebook\Statistics learning\data\MNIST\raw\train-images-idx3-ubyte'
# 训练集标签文件
train_labels_idx1_ubyte_file = r'D:\notebook\Statistics learning\data\MNIST\raw\train-labels-idx1-ubyte'
# 测试集文件
test_images_idx3_ubyte_file =r'D:\notebook\Statistics learning\data\MNIST\raw\t10k-images-idx3-ubyte'
# 测试集标签文件
test_labels_idx1_ubyte_file =r'D:\notebook\Statistics learning\data\MNIST\raw\t10k-labels-idx1-ubyte'


def decode_idx3_ubyte(idx3_ubyte_file):

    # 读取二进制数据
    bin_data = open(idx3_ubyte_file, 'rb').read()
    # 解析文件头信息，依次为魔数、图片数量、每张图片高、每张图片宽
    offset = 0
    fmt_header = '>iiii' #因为数据结构中前4行的数据类型都是32位整型，所以采用i格式，但我们需要读取前4行数据，所以需要4个i。我们后面会看到标签集中，只使用2个ii。
    magic_number, num_images, num_rows, num_cols = struct.unpack_from(fmt_header, bin_data, offset)
    #print('魔数:%d, 图片数量: %d张, 图片大小: %d*%d' % (magic_number, num_images, num_rows, num_cols))
    image_size = num_rows * num_cols
    offset += struct.calcsize(fmt_header)  #获得数据在缓存中的指针位置，从前面介绍的数据结构可以看出，读取了前4行之后，指针位置（即偏移位置offset）指向0016。
    #print(offset)
    fmt_image = '>' + str(image_size) + 'B'  #图像数据像素值的类型为unsigned char型，对应的format格式为B。这里还有加上图像大小784，是为了读取784个B格式数据，如果没有则只会读取一个值（即一副图像中的一个像素值）
    #print(fmt_image,offset,struct.calcsize(fmt_image))
    images = np.empty((num_images, num_rows, num_cols))
    for i in range(num_images):
        images[i] = np.array(struct.unpack_from(fmt_image, bin_data, offset)).reshape((num_rows, num_cols))
        offset += struct.calcsize(fmt_image)
    return images

def decode_idx1_ubyte(idx1_ubyte_file):
    bin_data = open(idx1_ubyte_file, 'rb').read()
    offset = 0
    fmt_header = '>ii'
    magic_number, num_images = struct.unpack_from(fmt_header, bin_data, offset)
    #print('魔数:%d, 图片数量: %d张' % (magic_number, num_images))
    offset += struct.calcsize(fmt_header)
    fmt_image = '>B'
    labels = np.empty(num_images)
    for i in range(num_images):
        labels[i] = struct.unpack_from(fmt_image, bin_data, offset)[0]
        offset += struct.calcsize(fmt_image)
    return labels

def load_train_images(idx_ubyte_file=train_images_idx3_ubyte_file):
    return decode_idx3_ubyte(idx_ubyte_file)

def load_train_labels(idx_ubyte_file=train_labels_idx1_ubyte_file):
    return decode_idx1_ubyte(idx_ubyte_file)

def load_test_images(idx_ubyte_file=test_images_idx3_ubyte_file):
    return decode_idx3_ubyte(idx_ubyte_file)

def load_test_labels(idx_ubyte_file=test_labels_idx1_ubyte_file):
    return decode_idx1_ubyte(idx_ubyte_file)

train_images = load_train_images()
train_labels = load_train_labels()
test_images = load_test_images()
test_labels = load_test_labels()

class Bayes(object):
    def __init__(self,data,label):
        self.data=data
        self.label=label
        m,n,p=np.shape(self.data)
        self.featurenum=np.prod(np.shape(data)[1:])
        self.data=np.reshape(self.data,(m,n*p))  
        self.propmatrix=[]
        self.labelprop=[]
        self.classnum=2        
        self.num=m
        self.fnum=n*p
        self.lnum=10
    def binary(self,inputdata):
        self.classnum=2
        #若进行测试，需输入二维行数据
        num=inputdata.shape[0]
        for ind in range(num):
            meanvalue=inputdata[ind].mean()
            #inputdata[ind]=[int(i/maxvalue*self.classnum) for i in inputdata[ind]]
            inputdata[ind]=np.array([0 if i<meanvalue else 1 for i in inputdata[ind]])            
        return inputdata
    def train(self,classnum=2):
        self.classnum=classnum
        self.data=self.binary(self.data)
        labelcount=Counter(self.label)
        self.lnum=len(labelcount)
        # 先验概率，分别计算标签及属性
        self.labelprop=np.array([(labelcount[i])/float(self.num) for i in range(self.lnum)])
        self.propmatrix=np.empty((self.lnum,self.fnum))
        for ii in range(self.lnum):
            numList=np.squeeze(self.data[np.where(self.label==ii)])
            numlabelii=labelcount[ii]
            #for jj in range(self.fnum):                
                #numCount=Counter(numList[:,jj])
                #self.propmatrix[ii,jj,:]=[(numCount[i]+1)/float(numlabelii+self.classnum) for i in range(self.classnum)]
            self.propmatrix[ii,:]=(numlist.sum(axis=0)+1)/float(numlabelii+self.classnum)
            #self.propmatrix[ii,:,0]=1-self.propmatrix[ii,:,1]
    def test(self,X):
        XB=np.squeeze(self.binary(X.reshape(1,-1)))
        prop=np.empty((self.lnum,1))
        for i in range(self.lnum):
            prop[i]=sum([math.log(self.propmatrix[i,j]) if j==1 else math.log(1-self.propmatrix[i,j]) for j in range(self.fnum)])
            prop[i]+=math.log(self.labelprop[i]) 
        label=np.argmax(prop)
        return label,prop


bayes=Bayes(train_images,train_labels)
bayes.train(2)

accuracy=0
for i in range(test_images.shape[0]):
    X=test_images[i]
    label,prop=bayes.test(X)
    accuracy+=label==test_labels[i]
accuracy=accuracy/float(np.size(test_labels))
print(accuracy)


def Binarization(images):
    for i in range(images.shape[0]):
        imageMean = images[i].mean()
        images[i] = np.array([0 if x < imageMean else 1 for x in images[i]])
    return images
def Bayes_train(train_x, train_y):    
    #先验概率P(0), P(1)....
    totalNum = train_x.shape[0]
    classNum = Counter(train_y)
    prioriP = np.array([classNum[i]/totalNum for i in range(10)])
    
    #后验概率
    posteriorNum = np.empty((10, train_x.shape[1]))
    posteriorP = np.empty((10, train_x.shape[1]))
    for i in range(10):
        posteriorNum[i] = train_x[np.where(train_y == i)].sum(axis = 0)  
        #拉普拉斯平滑      
        posteriorP[i] = (posteriorNum[i] + 1) / (classNum[i] + 2)   
    return prioriP, posteriorP

def Bayes_pret(test_x, test_y, prioriP, posteriorP):
    pret = np.empty(test_x.shape[0])
    for i in range(test_x.shape[0]):
        prob = np.empty(10)
        for j in range(10):
            temp = sum([math.log(1-posteriorP[j][x]) if test_x[i][x] == 0 else math.log(posteriorP[j][x]) for x in range(test_x.shape[1])])
            prob[j] = np.array(math.log(prioriP[j]) + temp)
        pret[i] = np.argmax(prob)
    return pret, (pret == test_y).sum()/ test_y.shape[0]

train_x_data = train_images
train_y = train_labels
train_x = np.resize(train_x_data, (train_x_data.shape[0], train_x_data.shape[1]*train_x_data.shape[2]))
train_x = Binarization(train_x)

test_x_data = test_images
# test_x = imageResize(test_x)
test_y = test_labels
test_x = np.resize(test_x_data, (test_x_data.shape[0], test_x_data.shape[1]*test_x_data.shape[2]))
test_x = Binarization(test_x)

prioriP, posteriorP = Bayes_train(train_x, train_y)
accuracy = Bayes_pret(test_x, test_y, prioriP, posteriorP)

print(accuracy)






