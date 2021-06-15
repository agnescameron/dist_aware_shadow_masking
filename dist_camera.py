import numpy as np
import random
import picamera
import subprocess
import RPi.GPIO as GPIO
import time
import os
from glob import glob
import board
import neopixel

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


camera = picamera.PiCamera()
camera.resolution = (640, 480)
camera.framerate = 10
GPIO.setmode(GPIO.BCM)
pixels = neopixel.NeoPixel(board.D18, 12, brightness=0.9)

#setup LEDs
red_led = 13
yellow_led = 6
green_led = 5
button_in = 19

GPIO.setup(button_in, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)

GPIO.setup(red_led, GPIO.OUT)
GPIO.setup(yellow_led, GPIO.OUT)
GPIO.setup(green_led, GPIO.OUT)

GPIO.output(red_led, GPIO.HIGH)
GPIO.output(yellow_led, GPIO.HIGH)
GPIO.output(green_led, GPIO.LOW)

def record_and_write():
	GPIO.output(green_led, GPIO.HIGH)
	GPIO.output(yellow_led, GPIO.LOW)

	try:
		os.remove('./temp/video.h264')
	except OSError:
		pass

	try:
		os.remove('./temp/video.mp4')
	except OSError:
		pass

	open('./temp/vid_1.txt', 'w').close()
	open('./temp/vid_2.txt', 'w').close()

	for f in glob ('./temp/*.avi'):
	   os.remove(f)

	#make output path with recording timestamp
	timestamp = round(time.time())
	output_path = os.path.join('./output/', str(timestamp))
	os.mkdir(output_path)

	print("started recording")
	pixels.fill((255, 255, 255))
	pixels.show()
	camera.start_recording('./temp/video.h264')
	camera.wait_recording(1)

	while True:
	    camera.wait_recording(0.05)
	    if GPIO.input(button_in) == GPIO.HIGH:
	        break

	print("finished filming")
	pixels.fill((0, 0, 0))
	pixels.show()

	GPIO.output(yellow_led, GPIO.HIGH)
	GPIO.output(red_led, GPIO.LOW)

	camera.stop_recording()
	command = "MP4Box -add ./temp/video.h264 ./temp/video.mp4"
	subprocess.call([command], shell=True)

	#copy the output video to the folder
	command2 = "cp ./temp/video.mp4 ./output/video_orig.mp4"
	subprocess.call([command2], shell=True)

	print("written recording, now converting frames")

	shadow_threshold = 0.5
	filter_size = (9,9)
	filter_threshold = 105

    net = DSDNet().cpu()
    cap = cv2.VideoCapture('../test_footage/samples/cici_flat_small.mov')
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))

    frame_array_1 = []
    frame_array_2 = []

    if len(args['snapshot']) > 0:
        print('load snapshot \'%s\' for testing' % args['snapshot'])
        net.load_state_dict(torch.load(os.path.join(ckpt_path, exp_name, args['snapshot'] + '.pth'), 
            map_location=torch.device('cpu')))
    
    colour = [random.randint(100, 255), random.randint(100, 255), random.randint(100, 255)]
 
    net.eval()
    with torch.no_grad():
        clip_count = 0

        #loop over video
		for i in range(frame_count-5):
			if i%50 == 0:
				print("frame", i, "of", frame_count)
				colour = [random.randint(100, 255), random.randint(100, 255), random.randint(100, 255)]

			GPIO.output(red_led, GPIO.HIGH)
			time.sleep(0.2)
			GPIO.output(red_led, GPIO.LOW)

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

			frame_array_1.append(image)
			frame_array_2.append(image)

			if i%20 == 0 and i != 0:
				clip_count = clip_count + 1
				print('writing video out, wait a sec...')

				#write out clip in style 1
				out_1 = cv2.VideoWriter(os.path.join('./temp/', 'video_1_' + str(clip_count) 
					+ '.avi'),cv2.VideoWriter_fourcc(*'DIVX'), fps, (frame_width, frame_height))

				for j in range(len(frame_array_1)):
					out_1.write(frame_array_1[j])

				out_1.release()

				with open('./temp/vid_1.txt', 'a') as file:
						file.write(f"file 'video_1_{str(clip_count)}.avi'\n")

				# write out clip in style 2
				out_2 = cv2.VideoWriter(os.path.join('./temp/', 'video_2_' + str(clip_count) 
					+ '.avi'),cv2.VideoWriter_fourcc(*'DIVX'), fps, (frame_width, frame_height))

				for j in range(len(frame_array_2)):
					out_2.write(frame_array_2[j])

				out_2.release()

				with open('./temp/vid_2.txt', 'a') as file:
				        file.write(f"file 'video_2_{str(clip_count)}.avi'\n")

				frame_array_1 = []
				frame_array_2 = []
				clip_count = clip_count + 1

		cap.release()

		#write out remaining videos
		out_1 = cv2.VideoWriter(os.path.join('./temp/', 'video_1_' + str(clip_count) 
			+ '.avi'),cv2.VideoWriter_fourcc(*'DIVX'), fps, (frame_width, frame_height))

		for j in range(len(frame_array_1)):
			out_1.write(frame_array_1[j])

		with open('./temp/vid_1.txt', 'a') as file:
			file.write(f"file 'video_1_{str(clip_count)}.avi'")

		out_1.release()

		out_2 = cv2.VideoWriter(os.path.join('./temp/', 'video_2_' + str(clip_count) 
			+ '.avi'),cv2.VideoWriter_fourcc(*'DIVX'), fps, (frame_width, frame_height))

		for j in range(len(frame_array_2)):
			out_2.write(frame_array_2[j])

		with open('./temp/vid_2.txt', 'a') as file:
			file.write(f"file 'video_2_{str(clip_count)}.avi'")

		out_2.release()

		cv2.destroyAllWindows()

		print('merging clips, cleaning up...')

		# join video 1
		subprocess.call(["ffmpeg -f concat -safe 0 -i ./temp/vid_1.txt -c copy " + os.path.join(output_path, "video_1.mp4")], 
			stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT, shell=True)

		# join video 2
		subprocess.call(["ffmpeg -f concat -safe 0 -i ./temp/vid_2.txt -c copy " + os.path.join(output_path, "video_2.mp4")], 
			stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT, shell=True)

		print("output is in", output_path)
		GPIO.output(red_led, GPIO.HIGH)

if __name__ == "__main__":
	run_count = 0
	while True:
		if GPIO.input(button_in) == GPIO.HIGH:
			try:
				record_and_write()
				GPIO.output(green_led, GPIO.LOW)
				run_count = run_count + 1
				print("ready to record, run count is", run_count)
			except Exception as e:
				print("hit exception, cleaning up")
				GPIO.cleanup()
				pixels.fill((0, 0, 0))
				pixels.show()
				sys.exit(1)
	GPIO.cleanup()
