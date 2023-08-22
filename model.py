import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

class conv(nn.Module):
    def __init__(self, inchannel, outchannel, kernel_size, stride):
        super(conv,self).__init__()
        self.ks = kernel_size
        self.conv = nn.Conv2d(inchannel, outchannel, kernel_size, stride)
        self.bn = nn.BatchNorm2d(outchannel)
    
    def forward(self, x):
        p = int(np.floor((self.ks-1)/2))
        x = F.pad(input=x, pad = (p,p,p,p))
        x = self.bn(self.conv(x))
        return F.elu(x, inplace=True)

class convblock(nn.Module):
    def __init__(self, inchannel, outchannel, kernelsize):
        super(convblock, self).__init__()
        self.conv1 = conv(inchannel, outchannel, kernelsize, 2)
        self.conv2 = conv(outchannel, outchannel, kernelsize, 1)

    def forward(self, x):
        x = self.conv2(self.conv1(x))
        return x

class upconv(nn.Module):
    def __init__(self, inchannel, outchannel):
        super(upconv, self).__init__()
        self.upconv = conv(inchannel, outchannel, kernel_size=3, stride=1)
    
    def forward(self, x):
        x = nn.functional.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        return self.upconv(x)

class iconv(nn.Module):
    def __init__(self, inchannel, outchannel):
        super(iconv, self).__init__()
        self.iconv = conv(inchannel, outchannel, kernel_size=3, stride=1)
    
    def forward(self, x):
        return self.iconv(x)

class getdisp(nn.Module):
    def __init__(self, inputchannel) -> None:
        super(getdisp, self).__init__()
        self.conv = nn.Conv2d(in_channels= inputchannel, out_channels=2, kernel_size= 3, stride=1)
        self.bn = nn.BatchNorm2d(2) # 2 disps left2right and right2left
        self.ac = torch.nn.Sigmoid()

    def forward(self, x):
        p = 1
        p2d = (p, p, p, p)
        x = self.bn(self.conv(F.pad(x,p2d)))
        disp = 0.3 * self.ac(x) # https://github.com/mrharicot/monodepth/issues/96
        return disp

class monomodel(nn.Module):
    def __init__(self):
        super(monomodel, self).__init__()

        # encoder
        self.convblock1 = convblock(3,32,7)
        self.convblock2 = convblock(32, 64, 5)
        self.convblock3 = convblock(64, 128, 3)
        self.convblock4 = convblock(128, 256, 3)
        self.convblock5 = convblock(256, 512, 3)
        self.convblock6 = convblock(512, 512, 3)
        self.convblock7 = convblock(512, 512, 3)

        # decoder
        self.upconv7 = upconv(512,512)
        self.iconv7 = iconv(1024, 512)
        self.upconv6 = upconv(512,512)
        self.iconv6 = iconv(1024, 512)
        self.upconv5 = upconv(512,256)
        self.iconv5 = iconv(512, 256)
        self.upconv4 = upconv(256,128)
        self.iconv4 = iconv(256, 128)
        self.disp4 = getdisp(128)
        self.upconv3 = upconv(128,64)
        self.iconv3 = iconv(130, 64)
        self.disp3 = getdisp(64)
        self.upconv2 = upconv(64,32)
        self.iconv2 = iconv(66, 32)
        self.disp2= getdisp(32)
        self.upconv1 = upconv(32,16)
        self.iconv1 = iconv(18, 16)
        self.disp1 = getdisp(16)

    def forward(self, x):
        # shortcut
        x1 = self.convblock1(x)
        x2 = self.convblock2(x1)
        x3 = self.convblock3(x2)
        x4 = self.convblock4(x3)
        x5 = self.convblock5(x4)
        x6 = self.convblock6(x5)
        x7 = self.convblock7(x6)

        up7 = self.upconv7(x7)
        cat7 = torch.cat((up7, x6), 1)
        out7 = self.iconv7(cat7)

        up6 = self.upconv6(out7)
        cat6 = torch.cat((up6, x5), 1)
        out6 = self.iconv6(cat6)

        up5 = self.upconv5(out6)
        cat5 = torch.cat((up5, x4), 1)
        out5 = self.iconv5(cat5)
        # 4 outscales
        up4 = self.upconv4(out5)
        cat4 = torch.cat((up4, x3), 1)
        out4 = self.iconv4(cat4)
        disp4 = self.disp4(out4)

        up3 = self.upconv3(out4)
        updisp4 = F.interpolate(disp4, scale_factor= 2, mode= 'bilinear', align_corners= True)
        cat3 = torch.cat((up3, x2, updisp4), 1)
        out3 = self.iconv3(cat3)
        disp3 = self.disp3(out3)

        up2 = self.upconv2(out3)
        updisp3 = F.interpolate(disp3, scale_factor= 2, mode= 'bilinear', align_corners= True)
        cat2 = torch.cat((up2, x1, updisp3), 1)
        out2 = self.iconv2(cat2)
        disp2 = self.disp2(out2)

        up1 = self.upconv1(out2)
        updisp2 = F.interpolate(disp2, scale_factor= 2, mode= 'bilinear', align_corners= True)
        cat1 = torch.cat((up1, updisp2), 1)
        out1 = self.iconv1(cat1)
        disp1 = self.disp1(out1)
        return  [disp1, disp2, disp3, disp4] # from big to small
    
if __name__ == '__main__':
    # the input width and height (512, 256) is ok for the monomodel, when change the number may cause error.
    # you can change the outputs of each layer to solve the problem
    from transforms import image_transforms
    from dataset import Kittiset
    from torch.utils.data import DataLoader
    data_root = 'E:/dataset/KITTI/raw'
    trainset = Kittiset(data_root, mode = 'train', transform=image_transforms(mode='train'))
    valset = Kittiset(data_root, mode= 'val', transform=image_transforms(mode='train'))
    trainloader = DataLoader(trainset, batch_size=1, shuffle= True, pin_memory=True)
    m = monomodel() 
    for data in trainloader:
        limg = data['limg']
        rimg = data['rimg']
        disps = m(limg)
        print(disps[1].size())
        break