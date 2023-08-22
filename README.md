# Monodepth-Pytorch
Repo by Jamesgender.(https://github.com/Jamesgender)
This repo is inspired by an amazing work of [Clément Godard](http://www0.cs.ucl.ac.uk/staff/C.Godard/), [Oisin Mac Aodha](http://vision.caltech.edu/~macaodha/) and [Gabriel J. Brostow](http://www0.cs.ucl.ac.uk/staff/g.brostow/) , for Unsupervised Monocular Depth Estimation.
The most valuable and helpful reference code is here[OniroAI](https://github.com/OniroAI/MonoDepth-PyTorch), I use their transforms.py and some other functions.
Original code and paper could be found via the following links:
1. [Original repo](https://github.com/mrharicot/monodepth)
2. [Original paper](https://arxiv.org/abs/1609.03677)

# Purpose
Just for learning. Monodepth is one of the most important MDE works, so I learn to write the code.

# Dataset
[Kitti raw](http://www.cvlibs.net/datasets/kitti/raw_data.php) dataset was used for training.
The dataset is too big to upload, so if you are not able to get all data, I recommend you download part of the dataset. The model was still OK for 8000 imgs training-set and 2000 imgs validing-set when I first using the model.

Example data folder structure, notice that I don't split the training imgs and val imgs to different folders. The imgs are split by txt files.
...
Kitti
├── 2011_09_26
|   ├── 2011_09_26_drive_0001_sync
|   ├── 2011_09_26_drive_0002_sync
│   ├── ...
├── 2011_09_28
├── ...

# Training

"""
python train.py --data_root (your data path) --model_path (pretrain model path)
"""
And you can change some opts in train.py on your own or just using '--' to tell the python.
I didn’t using pretrain model at first because it's hard to get from ImageNet. So the learning rate is 0.01 from(https://github.com/OniroAI/MonoDepth-PyTorch/issues/1). And I didn't using multi GPUs to train, about 15mins per epoch on one GTX 3090. 

After the first training, I used the first ckpt as pretrain model for second training. 

I can provide my epochs50lr0.01 train model as pretrain model for you.

# Testing
The eigen split has already been provided in eigen full folder, which has all test files' paths.
You need to change the ckpt root(the checkpoint path), and the data root again.
Using test.ipynb to test one of the eigen test imgs & compute metrics.
If you like my work, please click the star, thank you!

And maybe follow my bilibili.
(https://space.bilibili.com/2000310)

# 