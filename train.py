import datetime
import os

import torch
from torch import nn
from torch import optim
from torch.autograd import Variable
from torch.backends import cudnn
from torch.utils.data import DataLoader
from torchvision import transforms
import sys
import joint_transforms
from config import sbu_training_root
from dataset import ImageFolder
from misc import AvgMeter, check_mkdir
from model import DSDNet
from utils import MyBceloss12_n,MyWcploss
cudnn.benchmark = True

torch.cuda.set_device(0)

ckpt_path = './ckpt'
exp_name = 'SBU_MODEL'
# exp_name=sys.argv[1]

args = {
    'iter_num': 5000,
    'train_batch_size': 10,
    'last_iter': 0,
    'lr': 5e-3,
    'lr_decay': 0.9,
    'weight_decay': 1e-3,
    'momentum': 0.9,
    'snapshot': '',
    'scale':320
}

joint_transform = joint_transforms.Compose([
    joint_transforms.Resize((args['scale'], args['scale'])),
    joint_transforms.RandomHorizontallyFlip()
])
val_joint_transform = joint_transforms.Compose([
    joint_transforms.Resize((args['scale'], args['scale']))
])
img_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
target_transform = transforms.ToTensor()
to_pil = transforms.ToPILImage()

train_set = ImageFolder(sbu_training_root, joint_transform, img_transform, target_transform)
train_loader = DataLoader(train_set, batch_size=args['train_batch_size'], num_workers=10, shuffle=True)

bce_logit = MyBceloss12_n().cuda()
bce_logit_dst = MyWcploss().cuda()
log_path = os.path.join(ckpt_path, exp_name, str(datetime.datetime.now()) + '.txt')


def main():
    net = DSDNet().cuda().train()

    optimizer = optim.SGD([
        {'params': [param for name, param in net.named_parameters() if name[-4:] == 'bias'],
         'lr': 2 * args['lr']},
        {'params': [param for name, param in net.named_parameters() if name[-4:] != 'bias'],
         'lr': args['lr'], 'weight_decay': args['weight_decay']}
    ], momentum=args['momentum'])

    if len(args['snapshot']) > 0:
        print('training resumes from \'%s\'' % args['snapshot'])
        net.load_state_dict(torch.load(os.path.join(ckpt_path, exp_name, args['snapshot'] + '.pth')))
        optimizer.load_state_dict(torch.load(os.path.join(ckpt_path, exp_name, args['snapshot'] + '_optim.pth')))
        optimizer.param_groups[0]['lr'] = 2 * args['lr']
        optimizer.param_groups[1]['lr'] = args['lr']

    check_mkdir(ckpt_path)
    check_mkdir(os.path.join(ckpt_path, exp_name))
    open(log_path, 'w').write(str(args) + '\n\n')
    train(net, optimizer)


def train(net, optimizer):
    curr_iter = args['last_iter']
    while True:

        train_loss_record_shad, loss_fuse_record_shad, loss_down1_record_shad = AvgMeter(), AvgMeter(), AvgMeter()
        loss_down2_record_shad, loss_down3_record_shad, loss_down4_record_shad = AvgMeter(), AvgMeter(), AvgMeter()

        train_loss_record_dst1, loss_fuse_record_dst1, loss_down1_record_dst1 = AvgMeter(), AvgMeter(), AvgMeter()
        loss_down2_record_dst1, loss_down3_record_dst1, loss_down4_record_dst1 = AvgMeter(), AvgMeter(), AvgMeter()

        train_loss_record_dst2, loss_fuse_record_dst2, loss_down1_record_dst2 = AvgMeter(), AvgMeter(), AvgMeter()
        loss_down2_record_dst2, loss_down3_record_dst2, loss_down4_record_dst2 = AvgMeter(), AvgMeter(), AvgMeter()
        train_loss_record = AvgMeter()

        for i, data in enumerate(train_loader):
            optimizer.param_groups[0]['lr'] = 2 * args['lr'] * (1 - float(curr_iter) / args['iter_num']
                                                                ) ** args['lr_decay']
            optimizer.param_groups[1]['lr'] = args['lr'] * (1 - float(curr_iter) / args['iter_num']
                                                            ) ** args['lr_decay']

            inputs, labels, labels_dst1, labels_dst2 = data
            batch_size = inputs.size(0)
            inputs = Variable(inputs).cuda()
            labels = Variable(labels).cuda()
            labels_dst1 = Variable(labels_dst1).cuda()
            labels_dst2 = Variable(labels_dst2).cuda()

            optimizer.zero_grad()

            fuse_pred_shad, pred_down1_shad, pred_down2_shad, pred_down3_shad, pred_down4_shad, \
            fuse_pred_dst1, pred_down1_dst1, pred_down2_dst1, pred_down3_dst1, pred_down4_dst1, \
            fuse_pred_dst2, pred_down1_dst2, pred_down2_dst2, pred_down3_dst2, pred_down4_dst2, \
            pred_down0_dst1, pred_down0_dst2, pred_down0_shad = net(inputs)

            loss_fuse_shad = bce_logit(fuse_pred_shad, labels, labels_dst1, labels_dst2)
            loss1_shad = bce_logit(pred_down1_shad, labels, labels_dst1, labels_dst2)
            loss2_shad = bce_logit(pred_down2_shad, labels, labels_dst1, labels_dst2)
            loss3_shad = bce_logit(pred_down3_shad, labels, labels_dst1, labels_dst2)
            loss4_shad = bce_logit(pred_down4_shad, labels, labels_dst1, labels_dst2)
            loss0_shad = bce_logit(pred_down0_shad, labels, labels_dst1, labels_dst2)

            loss_fuse_dst1 = bce_logit_dst(fuse_pred_dst1, labels_dst1)
            loss1_dst1 = bce_logit_dst(pred_down1_dst1, labels_dst1)
            loss2_dst1 = bce_logit_dst(pred_down2_dst1, labels_dst1)
            loss3_dst1 = bce_logit_dst(pred_down3_dst1, labels_dst1)
            loss4_dst1 = bce_logit_dst(pred_down4_dst1, labels_dst1)
            loss0_dst1 = bce_logit_dst(pred_down0_dst1, labels_dst1)
            loss_fuse_dst2 = bce_logit_dst(fuse_pred_dst2, labels_dst2)
            loss1_dst2 = bce_logit_dst(pred_down1_dst2, labels_dst2)
            loss2_dst2 = bce_logit_dst(pred_down2_dst2, labels_dst2)
            loss3_dst2 = bce_logit_dst(pred_down3_dst2, labels_dst2)
            loss4_dst2 = bce_logit_dst(pred_down4_dst2, labels_dst2)
            loss0_dst2 = bce_logit_dst(pred_down0_dst2, labels_dst2)

            loss_shad = loss_fuse_shad + loss1_shad + loss2_shad + loss3_shad + loss4_shad +loss0_shad
            loss_dst1 = loss_fuse_dst1 + loss1_dst1 + loss2_dst1 + loss3_dst1 + loss4_dst1 +loss0_dst1
            loss_dst2 = loss_fuse_dst2 + loss1_dst2 + loss2_dst2 + loss3_dst2 + loss4_dst2 +loss0_dst2
            loss = loss_shad + 2*loss_dst1 + 2*loss_dst2

            loss.backward()

            optimizer.step()

            train_loss_record.update(loss.data, batch_size)
            train_loss_record_shad.update(loss_shad.data, batch_size)
            loss_fuse_record_shad.update(loss_fuse_shad.data, batch_size)
            loss_down1_record_shad.update(loss1_shad.data, batch_size)
            loss_down2_record_shad.update(loss2_shad.data, batch_size)
            loss_down3_record_shad.update(loss3_shad.data, batch_size)
            loss_down4_record_shad.update(loss4_shad.data, batch_size)


            train_loss_record_dst1.update(loss_dst1.data, batch_size)
            loss_fuse_record_dst1.update(loss_fuse_dst1.data, batch_size)
            loss_down1_record_dst1.update(loss1_dst1.data, batch_size)
            loss_down2_record_dst1.update(loss2_dst1.data, batch_size)
            loss_down3_record_dst1.update(loss3_dst1.data, batch_size)
            loss_down4_record_dst1.update(loss4_dst1.data, batch_size)


            train_loss_record_dst2.update(loss_dst2.data, batch_size)
            loss_fuse_record_dst2.update(loss_fuse_dst2.data, batch_size)
            loss_down1_record_dst2.update(loss1_dst2.data, batch_size)
            loss_down2_record_dst2.update(loss2_dst2.data, batch_size)
            loss_down3_record_dst2.update(loss3_dst2.data, batch_size)
            loss_down4_record_dst2.update(loss4_dst2.data, batch_size)



            curr_iter += 1

            log = '[iter %d], [train loss %.5f], [loss_train_shad %.5f], [loss_train_dst1 %.5f], [loss_train_dst2 %.5f], [lr %.13f]' % \
                  (curr_iter, train_loss_record.avg, train_loss_record_shad.avg, train_loss_record_dst1.avg,
                   train_loss_record_dst2.avg,optimizer.param_groups[1]['lr'])

            print(log)
            open(log_path, 'a').write(log + '\n')

            if curr_iter == 4500:
                torch.save(net.state_dict(), os.path.join(ckpt_path, exp_name, '%d.pth' % curr_iter))


            if curr_iter == args['iter_num']:
                torch.save(net.state_dict(), os.path.join(ckpt_path, exp_name, '%d.pth' % curr_iter))
                return


if __name__ == '__main__':
    main()
