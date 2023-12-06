import torch
import torch.nn as nn


class Generator(nn.Module):

    def __init__(self):
        super().__init__()

        self.feedforward=nn.Linear(100,4*4*1024)
        self.conv1=nn.ConvTranspose2d(1024,512,(4,4),2,1)
        self.conv2=nn.ConvTranspose2d(512,256,(4,4),2,1)
        self.conv3=nn.ConvTranspose2d(256,128,(4,4),2,1)
        self.conv4=nn.ConvTranspose2d(128,3,(4,4),2,1)
        self.relu=nn.ReLU()
        self.tanh=nn.Tanh()

    def forward(self,x):

        x=self.feedforward(x)

        batch_size,_=x.shape
        x=torch.reshape(x,(batch_size,1024,4,4))
        x=self.relu(x)

        x=self.conv1(x)
        x=self.relu(x)

        x=self.conv2(x)
        x=self.relu(x)

        x=self.conv3(x)
        x=self.relu(x)

        x=self.conv4(x)
        x=self.tanh(x)

        return x
    
class Discriminator(nn.Module):

    def __init__(self):
        super().__init__()

        self.conv1=nn.Conv2d(3,128,(4,4),2,1)
        self.conv2=nn.Conv2d(128,256,(4,4),2,1)
        self.conv3=nn.Conv2d(256,512,(4,4),2,1)
        self.conv4=nn.Conv2d(512,1024,(4,4),2,1)
        self.feedforward=nn.Linear(4*4*1024,2)
        self.leakyrelu=nn.LeakyReLU(0.2)
        self.softmax=nn.Softmax(1)

    def forward(self,x):

        x=self.conv1(x)
        x=self.leakyrelu(x)

        x=self.conv2(x)
        x=self.leakyrelu(x)

        x=self.conv3(x)
        x=self.leakyrelu(x)

        x=self.conv4(x)
        x=self.leakyrelu(x)

        batch_size,h,w,c=x.shape
        x=torch.reshape(x,(batch_size,h*w*c))

        x=self.feedforward(x)
        x=self.softmax(x)

        return x




