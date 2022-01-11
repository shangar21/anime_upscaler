import torch
import torchvision
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer
import cv2

# These names are kinda random, because in python you can't have a variable name start with a number
# CINEMA is basically 4K
# DESKTOP is 1440p
# TV is 1080p 
CINEMA = (3840, 2160)
DESKTOP = (2560, 1440)
TV = (1920, 1080)

def upscale(model_path, im_path):
	model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=6, num_grow_ch=32, scale=4)
	upsampler = RealESRGANer(scale=4, model_path=model_path, model=model, tile=0, tile_pad=10, pre_pad=0, half=False)
	img = cv2.imread(im_path, cv2.IMREAD_UNCHANGED)
	output, _ = upsampler.enhance(img, outscale=4)
	return output

# model_path = '/home/ssm/Documents/Real-ESRGAN/experiments/pretrained_models/RealESRGAN_x4plus_anime_6B.pth'
# print('upscaling...')
# output = upscale(model_path,'test_img/random_test_frame.jpg')
# cv2.imwrite('test_img/upscaled_frame.jpg', output)