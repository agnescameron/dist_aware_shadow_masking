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
    cap = cv2.VideoCapture('Datasets/SBU/SBU-Test/Cici-Shadow/living-room-small.mp4')
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_array = []
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))

    if len(args['snapshot']) > 0:
        print('load snapshot \'%s\' for testing' % args['snapshot'])
        net.load_state_dict(torch.load(os.path.join(ckpt_path, exp_name, args['snapshot'] + '.pth'), map_location=torch.device('cpu')))

    net.eval()
    with torch.no_grad():

        for i in range(3):
            success,image = cap.read()
            img = Image.fromarray(image)
            print('predicting for frame %d of %d' % (i, frame_count))
            w, h = img.size
            img_var = Variable(img_transform(img).unsqueeze(0)).cpu()
            res = net(img_var)
            prediction = np.array(transforms.Resize((h, w))(to_pil(res.data.squeeze(0).cpu())))
            prediction = crf_refine(np.array(img.convert('RGB')), prediction)

            # prediction.save(
            #     os.path.join('Output/SBU/vid_test_1', str(i))+'.png')
            frame = np.array(prediction)
            frame_array.append(frame)

    print('writing video out, wait a sec...')
    out = cv2.VideoWriter('../test_footage/distraction_aware/test_out.avi',cv2.VideoWriter_fourcc(*'DIVX'), fps, (frame_width, frame_height))
    for j in range(len(frame_array)):
        out.write(frame_array[j])

    out.release()

if __name__ == '__main__':
    main()
