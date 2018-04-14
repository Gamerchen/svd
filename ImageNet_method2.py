# -*-coding:utf-8-*-
import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.autograd import Variable
from torch import optim
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from numpy.linalg import svd
import torchvision.datasets as datasets
import sys
import time

learning_rate = 1e-3
batch_size = 100
epoches = 30

k_elements = 4

transform = transforms.Compose([
    transforms.RandomSizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean = [ 0.485, 0.456, 0.406 ],
                         std = [ 0.229, 0.224, 0.225 ]),
])

trainset=list(datasets.ImageFolder('./train',transform))


trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=1)

# build network
class vgg19(nn.Module):
    def __init__(self):
        super(vgg19, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(256, 512, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, stride=1, padding=1),
            nn.ReLU(),
            #nn.MaxPool2d(2, 2),
            nn.Conv2d(512, 512, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, stride=1, padding=1),
            nn.ReLU(),
            #nn.MaxPool2d(2, 2)
        )
        
        self.classifier=torch.nn.Sequential()
        self.classifier.add_module('fc1',nn.Linear(512*28*1, 4096))
        self.classifier.add_module('fc2',nn.Linear(4096, 4096))
        self.classifier.add_module('fc3',nn.Linear(4096, 10))


    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return F.log_softmax(out)

vgg19=vgg19()
pretrained_dict = torch.load('vgg19.pth')
model_dict = vgg19.state_dict()  
print(vgg19)
# 1. filter out unnecessary keys
pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
# 2. overwrite entries in the existing state dict
model_dict.update(pretrained_dict)
# 3. load the new state dict
vgg19.load_state_dict(model_dict)
        
for param in list(vgg19.parameters())[:-6]:
    param.requires_grad = False
# 替换最后的全连接层， 改为训练10类
# 新构造的模块的参数默认requires_grad为True
vgg19.cuda()

criterian = nn.CrossEntropyLoss(size_average=False)
# 只优化最后的分类层
optimizer = optim.SGD(vgg19.classifier.parameters(), lr=1e-2, momentum=0.9)

# train
for i in range(epoches):
    svd_time_cost = 0.
    start = time.time()
    
    running_loss = 0.
    running_acc = 0.
    for (img, label) in trainloader:
        img = img.cuda()
        img_re = torch.zeros(100,3,224,2*k_elements).cuda()
        
		#SVD
        svd_start = time.time()
        for i in range(len(img)):
            for j in range(len(img[i])):
                A=img[i][j]
                U, s, V = torch.svd(A)
                Sigma = torch.zeros((A.shape[0], A.shape[1])).cuda()
                Sigma[:A.shape[0], :A.shape[0]] = torch.diag(s)
                # first k singular values
                Sigma = Sigma[:, :k_elements].float().cuda()
                V = V[:k_elements, :]
                V = torch.transpose(V,0,1)
                # reconstruct
                x = np.dot(U,Sigma)
                x = torch.from_numpy(x).cuda()
                
                temp = torch.cat((x,V),1)
                img_re[i][j].copy_(temp)
                
        svd_end = time.time()
        svd_time_cost +=svd_end-svd_start 

        img_re = Variable(img_re).cuda()
        label = Variable(label).cuda()    
        
        optimizer.zero_grad()
        output = vgg19(img_re)
        loss = criterian(output, label)
        # backward
        loss.backward()
        optimizer.step()

        running_loss += loss.data[0]
        _, predict = torch.max(output, 1)
        correct_num = (predict == label).sum()
        running_acc += correct_num.data[0]

    running_loss /= len(trainset)
    running_acc /= len(trainset)
    
    end = time.time()
    
    epoch_time_cost = end - start - svd_time_cost

    print("[%d/%d] Loss: %.5f, Acc: %.2f, Time: %.2f" %(i+1, epoches, running_loss, 100*running_acc, epoch_time_cost))
