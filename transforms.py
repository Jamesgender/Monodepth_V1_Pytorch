import torch
import torchvision.transforms as transforms
import numpy as np


def image_transforms(mode='train', augment_parameters=[0.8, 1.2, 0.5, 2.0, 0.8, 1.2],
                     do_augmentation=True, transformations=None,  size=(256, 512)):
    if mode == 'train':
        data_transform = transforms.Compose([
            ResizeImage(train=True, size=size),
            RandomFlip(do_augmentation),
            ToTensor(train=True),
            AugmentImagePair(augment_parameters, do_augmentation)
        ])
        return data_transform
    elif mode == 'test':
        data_transform = transforms.Compose([
            ResizeImage(train=False, size=size),
            ToTensor(train=False),
            DoTest(),
        ])
        return data_transform
    elif mode == 'custom':
        data_transform = transforms.Compose(transformations)
        return data_transform
    else:
        print('Wrong mode')


class ResizeImage(object):
    def __init__(self, train=True, size=(256, 512)):
        self.train = train
        self.transform = transforms.Resize(size)

    def __call__(self, sample):
        if self.train:
            limg = sample['limg']
            rimg = sample['rimg']
            new_rimg = self.transform(rimg)
            new_limg = self.transform(limg)
            sample = {'limg': new_limg, 'rimg': new_rimg}
        else:
            limg = sample
            new_limg = self.transform(limg)
            sample = new_limg
        return sample


class DoTest(object):
    def __call__(self, sample):
        new_sample = torch.stack((sample, torch.flip(sample, [2]))) # For PP
        return new_sample


class ToTensor(object):
    def __init__(self, train):
        self.train = train
        self.transform = transforms.ToTensor()

    def __call__(self, sample):
        if self.train:
            limg = sample['limg']
            rimg = sample['rimg']
            new_rimg = self.transform(rimg)
            new_limg = self.transform(limg)
            sample = {'limg': new_limg,
                      'rimg': new_rimg}
        else:
            limg = sample
            sample = self.transform(limg)
        return sample


class RandomFlip(object):
    def __init__(self, do_augmentation):
        self.transform = transforms.RandomHorizontalFlip(p=1)
        self.do_augmentation = do_augmentation

    def __call__(self, sample):
        limg = sample['limg']
        rimg = sample['rimg']
        k = np.random.uniform(0, 1, 1)
        if self.do_augmentation:
            if k > 0.5:
                fliped_left = self.transform(rimg)
                fliped_right = self.transform(limg)
                sample = {'limg': fliped_left, 'rimg': fliped_right}
        else:
            sample = {'limg': limg, 'rimg': rimg}
        return sample


class AugmentImagePair(object):
    def __init__(self, augment_parameters, do_augmentation):
        self.do_augmentation = do_augmentation
        self.gamma_low = augment_parameters[0]  # 0.8
        self.gamma_high = augment_parameters[1]  # 1.2
        self.brightness_low = augment_parameters[2]  # 0.5
        self.brightness_high = augment_parameters[3]  # 2.0
        self.color_low = augment_parameters[4]  # 0.8
        self.color_high = augment_parameters[5]  # 1.2

    def __call__(self, sample):
        limg = sample['limg']
        rimg = sample['rimg']
        p = np.random.uniform(0, 1, 1)
        if self.do_augmentation:
            if p > 0.5:
                # randomly shift gamma
                random_gamma = np.random.uniform(self.gamma_low, self.gamma_high)
                limg_aug = limg ** random_gamma
                rimg_aug = rimg ** random_gamma

                # randomly shift brightness
                random_brightness = np.random.uniform(self.brightness_low, self.brightness_high)
                limg_aug = limg_aug * random_brightness
                rimg_aug = rimg_aug * random_brightness

                # randomly shift color
                random_colors = np.random.uniform(self.color_low, self.color_high, 3)
                for i in range(3):
                    limg_aug[i, :, :] *= random_colors[i]
                    rimg_aug[i, :, :] *= random_colors[i]

                # saturate
                limg_aug = torch.clamp(limg_aug, 0, 1)
                rimg_aug = torch.clamp(rimg_aug, 0, 1)

                sample = {'limg': limg_aug, 'rimg': rimg_aug}

        else:
            sample = {'limg': limg, 'rimg': rimg}
        return sample
