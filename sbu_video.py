import numpy as np
import os

import torch
from PIL import Image
from torch.autograd import Variable
from torchvision import transforms

from config import sbu_testing_root
from misc import check_mkdir, crf_refine
from model import DSDNet
import sys
import random

import cv2

ckpt_path = './ckpt'
exp_name='models'

args = {
    'snapshot': '5000_SBU',
    'scale': 320
}

img_transform = transforms.Compose([
    transforms.Resize((args['scale'],args['scale'])),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

to_test = {'sbu': sbu_testing_root}

to_pil = transforms.ToPILImage()


def main():
    net = DSDNet().cpu()
    cap = cv2.VideoCapture('../test_footage/samples/cici_flat_small.mov')
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_array = []
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))

    if len(args['snapshot']) > 0:
        print('load snapshot \'%s\' for testing' % args['snapshot'])
        net.load_state_dict(torch.load(os.path.join(ckpt_path, exp_name, args['snapshot'] + '.pth'), 
            map_location=torch.device('cpu')))
    
    colour = [random.randint(100, 255), random.randint(100, 255), random.randint(100, 255)]
 
    net.eval()
    with torch.no_grad():
        clip_count = 0
        for i in range(frame_count-5):
            if i%50 == 0:
                colour = [random.randint(100, 255), random.randint(100, 255), random.randint(100, 255)]
            success,image = cap.read()
            img = Image.fromarray(image)
            print('predicting for frame %d of %d' % (i, frame_count))
            w, h = img.size
            img_var = Variable(img_transform(img).unsqueeze(0)).cpu()
            res = net(img_var)
            prediction = np.array(transforms.Resize((h, w))(to_pil(res.data.squeeze(0).cpu())))
            prediction = crf_refine(np.array(img.convert('RGB')), prediction)
            frame = np.array(prediction)

            #add colour mask
            ret, mask = cv2.threshold(frame, 230, 255,cv2.THRESH_BINARY_INV)
            bw_mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
            col_img = np.zeros(frame.shape, frame.dtype)
            col_img[:,:] = (colour[0], colour[1], colour[2])

            col_mask = cv2.bitwise_and(col_img, col_img, mask=bw_mask)
            bgr_mask = np.where(col_mask < 10, 255, col_mask)
            cv2.addWeighted(bgr_mask, 0.65, frame, 0.35, 0, image)

            cv2.imshow('', image)
            k = cv2.waitKey(30) & 0xff
            if k == 27:
                break

            frame_array.append(image)

            if i%1000 == 0 and i != 0:
                clip_count = clip_count + 1
                print('writing video out, wait a sec...')
                out = cv2.VideoWriter('../test_footage/distraction_aware/cici_flat/clip' + str(clip_count) 
                    + '.avi',cv2.VideoWriter_fourcc(*'DIVX'), fps, (frame_width, frame_height))
                print('frame array is', len(frame_array))
                for j in range(len(frame_array)):
                    out.write(frame_array[j])
                frame_array = []

                out.release()

if __name__ == '__main__':
    main()
