# the inputs of the loss function are 4-scales disps and left&right images
import torch
import torch.nn as nn
import torch.nn.functional as F
class Monoloss(nn.modules.Module):
    def __init__(self, n=4, SSIM_w=0.85, disp_gradient_w=0.1, lr_w=1.0):
        super(Monoloss, self).__init__()
        self.n = n
        self.ssimw = SSIM_w
        self.dispgradientw = disp_gradient_w
        self.lrw = lr_w
    
    def gradient_x(self, img):
        # Pad input to keep output size consistent
        img = F.pad(img, (0, 1, 0, 0), mode="replicate")
        gx = img[:, :, :, :-1] - img[:, :, :, 1:]  # NCHW
        return gx

    def gradient_y(self, img):
        # Pad input to keep output size consistent
        img = F.pad(img, (0, 0, 0, 1), mode="replicate")
        gy = img[:, :, :-1, :] - img[:, :, 1:, :]  # NCHW
        return gy

    def smoothloss(self, disps, stacks):
        disp_gradients_x = [self.gradient_x(d) for d in disps]
        disp_gradients_y = [self.gradient_y(d) for d in disps]

        image_gradients_x = [self.gradient_x(img) for img in stacks]
        image_gradients_y = [self.gradient_y(img) for img in stacks]

        weights_x = [torch.exp(-torch.mean(torch.abs(g), 1,
                     keepdim=True)) for g in image_gradients_x]
        weights_y = [torch.exp(-torch.mean(torch.abs(g), 1,
                     keepdim=True)) for g in image_gradients_y]

        smoothness_x = [disp_gradients_x[i] * weights_x[i]
                        for i in range(self.n)]
        smoothness_y = [disp_gradients_y[i] * weights_y[i]
                        for i in range(self.n)]
        
        smoothness = [torch.mean(torch.abs(smoothness_x[i]) + torch.abs(smoothness_y[i]))
                for i in range(self.n)] # mean and abs


        smoothloss = sum(smoothness[i]/2**i for i in range(self.n))
        return self.dispgradientw * smoothloss

    def apply_disparity(self, img, disp):
        # Using disps to sample from images
        batch_size, _, height, width = img.size()

        # Original coordinates of pixels
        x_base = torch.linspace(0, 1, width).repeat(batch_size,
                    height, 1).type_as(img)
        y_base = torch.linspace(0, 1, height).repeat(batch_size,
                    width, 1).transpose(1, 2).type_as(img)

        # Apply shift in X direction
        x_shifts = disp[:, 0, :, :]  # Disparity is passed in NCHW format with 1 channel
        flow_field = torch.stack((x_base + x_shifts, y_base), dim=3)
        # In grid_sample coordinates are assumed to be between -1 and 1
        output = F.grid_sample(img, 2*flow_field - 1, mode='bilinear',
                               padding_mode='zeros', align_corners= True)

        return output
   
    def SSIM(self, x, y):
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        mu_x = nn.AvgPool2d(3, 1)(x)
        mu_y = nn.AvgPool2d(3, 1)(y)
        mu_x_mu_y = mu_x * mu_y
        mu_x_sq = mu_x.pow(2)
        mu_y_sq = mu_y.pow(2)

        sigma_x = nn.AvgPool2d(3, 1)(x * x) - mu_x_sq
        sigma_y = nn.AvgPool2d(3, 1)(y * y) - mu_y_sq
        sigma_xy = nn.AvgPool2d(3, 1)(x * y) - mu_x_mu_y

        SSIM_n = (2 * mu_x_mu_y + C1) * (2 * sigma_xy + C2)
        SSIM_d = (mu_x_sq + mu_y_sq + C1) * (sigma_x + sigma_y + C2)
        SSIM = SSIM_n / SSIM_d

        return torch.clamp((1 - SSIM) / 2, 0, 1)

    def matchingLoss(self, ngleftdisps, rightstacks, rightdisps, leftstacks):
        # generate 4 new images and compute the loss
        leftnewstacks  = [self.apply_disparity(rightstacks[i], ngleftdisps[i]) for i in range(4)]
        rightnewstacks = [self.apply_disparity(leftstacks[i], rightdisps[i]) for i in range(4)]
        # L1 loss
        L1L = sum([torch.mean(torch.abs(leftnewstacks[i] - leftstacks[i]))
                   for i in range(self.n)])
        L1R = sum([torch.mean(torch.abs(rightnewstacks[i] - rightstacks[i]))
                   for i in range(self.n)])
        
        # SSIM 
        ssimL = sum([torch.mean(self.SSIM(leftnewstacks[i], leftstacks[i])) for i in range(self.n)])
        ssimR = sum([torch.mean(self.SSIM(rightnewstacks[i], rightstacks[i])) for i in range(self.n)])

        matchingloss = (1-self.ssimw) *(L1L + L1R) + self.ssimw * (ssimL + ssimR)
        return matchingloss
    
    def consistencyloss(self, dispsl, dispsr):
        # using disps to generate each other
        newdispsr = [self.apply_disparity(dispsl[i], dispsr[i]) for i in range(self.n)]
        newdispsl = [self.apply_disparity(dispsr[i], -dispsl[i]) for i in range(self.n)]

        # compute consitency
        lcons = sum(torch.mean(torch.abs(newdispsl[i] - dispsl[i])) for i in range(self.n))
        rcons = sum(torch.mean(torch.abs(newdispsr[i] - dispsr[i])) for i in range(self.n))
        consistencyloss = lcons + rcons
        return self.lrw * consistencyloss
    
    def forward(self, disps, targets):
        """
        Args:
            input [disp1, disp2, disp3, disp4]
            target [left, right]

        Return:
            (float): The loss
        """
        # get 4 scales disps, each disp has 2 channel, which reprensents left&right disp
        leftdisps = []
        rightdisps = []
        for d in disps:
            leftdisps.append(d[:,0,:,:].unsqueeze(1)) # keep 4 dim
            rightdisps.append(d[:,1,:,:].unsqueeze(1)) # from big to small
        
        # get 2 target images, left and right, then produce 4 scales images
        leftimage = targets[0]
        rightimage = targets[1]
        leftstacks = []
        rightstacks = []
        h = leftimage.size()[2]
        w = leftimage.size()[3]

        for i in range(self.n):
            ratio = 2 ** i
            ratiow = w // ratio
            ratioh = h // ratio
            ratioimg = F.interpolate(leftimage, size=(ratioh,ratiow), mode= 'bilinear', align_corners= True)
            leftstacks.append(ratioimg)
            ratioimg = F.interpolate(rightimage, size=(ratioh,ratiow), mode= 'bilinear', align_corners= True)
            rightstacks.append(ratioimg) # from big to small

        # matching loss
        ngleftdisps = [-leftdisps[i] for i in range(self.n)]
        matchingloss = self.matchingLoss(ngleftdisps, rightstacks, rightdisps, leftstacks)
        #print(Lmatchingloss, Rmatchingloss)

        # smoothness loss
        Lsmoothloss = self.smoothloss(leftdisps, leftstacks)
        Rsmoothloss = self.smoothloss(rightdisps, rightstacks)
        smoothloss = Lsmoothloss+Rsmoothloss
        #print(smoothloss)

        # consistency loss
        consistency = self.consistencyloss(leftdisps, rightdisps)
        #print(consistency)

        # total loss
        Loss  =  smoothloss + matchingloss + consistency
        return Loss

if __name__ == '__main__':
    from model import monomodel
    from dataset import Kittiset
    from transforms import image_transforms
    import matplotlib.pyplot as plt
    from torch.utils.data import DataLoader

    data_root = 'E:/dataset/KITTI/raw'
    trainset = Kittiset(data_root, mode = 'train', transform=image_transforms(mode='train'))
    valset = Kittiset(data_root, mode= 'val', transform=image_transforms(mode='train'))
    trainloader = DataLoader(trainset, batch_size=2, shuffle= False, pin_memory=True)
    m = monomodel() 
    for data in trainloader:
        limg = data['limg']
        rimg = data['rimg']
        disps = m(limg)
        target = [limg, rimg]
        break
    lossfc = Monoloss()
    valloss = lossfc(disps, target)
    print(valloss)

    
