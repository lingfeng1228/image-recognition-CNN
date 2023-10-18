# 导入必要的包
#import torch
#import torchvision
#from torch import nn, optim
#from torch.utils.data import DataLoader
#from torchvision import models, datasets, transforms

# %pylab inline  # 魔法方法用于显示 plt.show()

#root = 'D:/shenjingwangluo/output/train/'
#dataset=torchvision.datasets.ImageFolder(root)
#print(dataset)

import os
url1 = r"D:/shenjingwangluo/output/output3/val"
k = 0
for filename in os.listdir("D:/shenjingwangluo/output/output3/val"):
    int(k)
    k = k + 1
    f = url1+'/'+filename
    num = 0
    k = str(k)
    for i in os.listdir(f):

        newname = f + '/' + str(k) + '-' + str(num)+'.png'
        num = int(num)
        k = int(k)
        num = num + 1
        oldname = f+'/'+i
        os.rename(oldname, newname)
        print(oldname, '->', newname)

