import os
from PIL import Image
from torch.utils.data import Dataset

# You must change your own Kitti dataset path before training

class Kittiset(Dataset):
    def __init__(self, data_root, mode, transform = None):
        super(Kittiset, self).__init__()
        self.root = data_root
        self.mode = mode
        self.transform = transform
        self.datalist = []
        if mode == 'train':
            with open('./eigen_full/train_files.txt', 'r') as trainlist:
                for line in trainlist.readlines():
                    _, _ , side =line.split(' ')
                    if side == 'l\n':
                        self.datalist.append(line)  
            trainlist.close()

        if mode == 'val':
            with open('./eigen_full/val_files.txt', 'r') as vallist:
                for line in vallist.readlines():
                    _, _ , side =line.split(' ')
                    if side == 'l\n':
                        self.datalist.append(line)
            vallist.close()

        if mode == 'test':
            with open('./eigen_full/test_files.txt', 'r') as testlist:
                for line in testlist.readlines():
                    self.datalist.append(line)
            testlist.close()
        
    def __len__(self):
        return self.datalist.__len__()

    def __getitem__(self, index):
        filedict, imagename, side = self.datalist[index].split(' ')
        leftroot = os.path.join(self.root, filedict, "image_02/data/{:010d}.png".format(int(imagename)))
        limg = Image.open(leftroot).convert('RGB')
        if self.mode != 'test': # train or val
            rightroot = os.path.join(self.root, filedict, "image_03/data/{:010d}.png".format(int(imagename)))
            rimg = Image.open(rightroot).convert('RGB')          
            stereos = {'limg': limg, 'rimg': rimg}
            if self.transform:
                stereos = self.transform(stereos)
                return stereos
            else:
                return stereos
        if self.mode == 'test':
            if self.transform:
                limg = self.transform(limg)
                return limg
            else:
                return limg
    
if __name__ == '__main__':
    data_root = 'E:/dataset/KITTI/raw'
    trainloader = Kittiset(data_root, mode = 'train')
    valloader = Kittiset(data_root, mode= 'val')
    import matplotlib.pyplot as plt
    from transforms import image_transforms

    img = trainloader.__getitem__(0)

    #print(trainloader.__len__())
    #testloader = Kittiset(data_root, mode= 'test',  transform= image_transforms(mode= 'test'))
    plt.imshow(trainloader.__getitem__(1)['limg'])
    #img = testloader.__getitem__(1)
    #print(img.size())
    #left = img[0,:,:,:]
    
    #plt.imshow(img)

    plt.show()

