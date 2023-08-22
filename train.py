import torch
import numpy as np
import time
import tqdm
from torch.utils.data import DataLoader
from torch import optim
from dataset import Kittiset
from loss import Monoloss
from model import monomodel
from argparse import ArgumentParser
from transforms import image_transforms

def args():
    parser = ArgumentParser(description= 'Monodepth V1 Pytorch repo')
    parser.add_argument('--data_root',
                        help='path to Kitti dataset', default= 'E:/dataset/KITTI/raw')
    parser.add_argument('--mode',
                        help='train or test', default= 'train')
    parser.add_argument('--model_path', help='path to the pretrained model')
    parser.add_argument('--out_dict', help='path to save checkpoints',default= './outs')

    parser.add_argument('--input_height', type=int, help='input height',
                        default=256)
    parser.add_argument('--input_width', type=int, help='input width',
                        default=512)
    parser.add_argument('--batch_size', default = 48)
    parser.add_argument('--lr', default= 0.01)
    parser.add_argument('--num_of_works', default= 4)
    parser.add_argument('--epochs', default= 50)
    parser.add_argument('--adjust_lr', default=True,
                        help='apply learning rate decay or not\
                        (default: True)'
                        )
    parser.add_argument('--device',
                        default='cuda:0',
                        help='choose cpu or cuda:0 device"'
                        )
    parser.add_argument('--use_multiple_gpu',default= False)
    parser.add_argument('--print_freq', default=50,
                        help= 'print frequency')

    args = parser.parse_args()
    return args

def adjust_learning_rate(optimizer, epoch, learning_rate):
    """Sets the learning rate to the initial LR\
        decayed by 2 every 10 epochs after 30 epoches"""

    if epoch >= 30 and epoch < 40:
        lr = learning_rate / 2
    elif epoch >= 40:
        lr = learning_rate / 4
    else:
        lr = learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def post_process_disparity(disp):
    (_, h, w) = disp.shape
    l_disp = disp[0, :, :]
    r_disp = np.fliplr(disp[1, :, :])
    m_disp = 0.5 * (l_disp + r_disp)
    (l, _) = np.meshgrid(np.linspace(0, 1, w), np.linspace(0, 1, h))
    l_mask = 1.0 - np.clip(20 * (l - 0.05), 0, 1)
    r_mask = np.fliplr(l_mask)
    return r_mask * l_disp + l_mask * r_disp + (1.0 - l_mask - r_mask) * m_disp

def train(opt, trainloader, valloader, model, lossfc, optimizer):
    print('Start Training')
    print('Epoch: ' , opt.epochs,'|', 'Batchsize: ', opt.batch_size,'|', 'Learningrate: ', opt.lr ,'|',
          'num_batch', len(trainloader),'|','num_val_batch', len(valloader))

    best_val_loss = float('Inf')
    model.train()
    for epoch in range(opt.epochs):
        if opt.adjust_lr:
            adjust_learning_rate(optimizer, epoch, opt.lr)
        
        starttime = time.time()
        runningloss = 0.0
        model.train()
        print('epoch {}'.format(epoch+1))
        for i, data in enumerate(trainloader):
            # load data
            if i % opt.print_freq == 0:
                print('batch : {}'.format(i))
            limg = data['limg'].to(opt.device)
            rimg = data['rimg'].to(opt.device)
            # set optimizer
            optimizer.zero_grad()

            disps = model(limg)
            target = [limg, rimg]
            trainloss = lossfc(disps, target)
            trainloss.backward()
            optimizer.step()
            runningloss += trainloss.item()

        running_val_loss = 0.0
        model.eval()
        print('val')
        with torch.no_grad():
            for i, data in enumerate(valloader):
                if i % 5 == 0:
                    print('batch : {}'.format(i))
                limg = data['limg'].to(opt.device)
                rimg = data['rimg'].to(opt.device)
                disps = model(limg)
                target = [limg, rimg]
                valloss =lossfc(disps, target)
                running_val_loss += valloss.item()
        
        runningloss /= len(trainloader) 
        running_val_loss /= len(valloader) 
        print ('Epoch:', epoch + 1, 'train_loss:', runningloss, 'val_loss:', running_val_loss,
                'time:', round(time.time() - starttime, 3),'s',)
        
        if running_val_loss < best_val_loss:
                print('save best ckpt')
                torch.save(model.state_dict(), opt.out_dict+ '/best.pth')
                best_val_loss = running_val_loss
                print('Model_saved')

def test():
    pass

def main(opt):
    data_root = opt.data_root
    device = opt.device
    # get model
    model = monomodel()
    print('Using Monomodel')
    model = model.to(device)

    if opt.model_path:
        model.load_state_dict(torch.load(opt.model_path))
        print('Using Pretrain Model')

    if opt.use_multiple_gpu:
        model = torch.nn.DataParallel(model)
    

    # get loss and optimizer 
    lossfc = Monoloss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=opt.lr)

    trainset = Kittiset(data_root, mode = 'train', transform=image_transforms(mode='train'))
    valset = Kittiset(data_root, mode= 'val', transform=image_transforms(mode='train'))
    trainloader = DataLoader(trainset, batch_size=opt.batch_size, shuffle= True, pin_memory=True)
    valloader = DataLoader(valset, batch_size=opt.batch_size, shuffle= True, pin_memory=True)

    output_directory = opt.out_dict
    input_height = opt.input_height
    input_width = opt.input_width

    if 'cuda' in device:
            torch.cuda.synchronize()

    train(opt, trainloader, valloader, model, lossfc, optimizer)


if __name__ == '__main__':
    opt = args()
    if opt.mode == 'train':
        print('mode : Train')
        main(opt)
    else:
        print('mode : Test')
        test()
