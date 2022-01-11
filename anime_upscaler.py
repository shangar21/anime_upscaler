import frame_esrgan
import cv2
import moviepy.editor as mpe
from tqdm import tqdm
import os
import argparse
import shutil

def extract_frames(vid_path):
	vid = cv2.VideoCapture(vid_path)
	images = []
	success, image = vid.read()
	while success:
		images.append(image)
		success, image = vid.read()
	return images

def create_temp_folder(vid_path):
	if os.path.exists('tmp'):
		folder_name = vid_path.split('/')[-1].split('.')[0]
		os.mkdir('tmp/{}'.format(folder_name))
	else:
		os.mkdir('tmp')
		create_temp_folder(vid_path)

def clear_temp(temp_path):
	shutil.rmtree(temp_path)

def setup_frames(vid_path):
	folder_name = vid_path.split('/')[-1].split('.')[0]
	images = extract_frames(vid_path)
	create_temp_folder(vid_path)
	os.mkdir('tmp/{}/original'.format(folder_name))
	for i in tqdm(range(len(images))):
		cv2.imwrite('tmp/{}/original'.format(folder_name)+'/frame_{}.jpg'.format(i), images[i])
	os.mkdir('tmp/{}/upscaled'.format(folder_name))

def upscale(vid_path):
	folder_name = vid_path.split('/')[-1].split('.')[0]
	print('extracting frames...')
	setup_frames(vid_path)		
	print('upscaling...')
	for i in tqdm(os.listdir('tmp/{}/original'.format(folder_name))):
		out = frame_esrgan.upscale(args.model_path, 'tmp/{}/original/{}'.format(folder_name, i))
		cv2.imwrite('tmp/{}/upscaled/{}'.format(folder_name, i), out)

def combine_frames(video_path, new_video_path):
	folder_name = video_path.split('/')[-1].split('.')[0]
	images = [img for img in os.listdir('tmp/{}/upscaled'.format(folder_name))]
	height, width, layers = cv2.imread('tmp/{}/upscaled/frame_0.jpg'.format(folder_name)).shape
	fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
	video = cv2.VideoWriter(new_video_path, fourcc, 30, (width, height))
	for i in tqdm(range(len(images))):
		video.write(cv2.imread('tmp/{}/upscaled/frame_{}.jpg'.format(folder_name, i)))
	cv2.destroyAllWindows()
	video.release()

def extract_audio(video_path, audio_path):
	clip = mpe.VideoFileClip(video_path)
	clip.audio.write_audiofile(audio_path)

def copy_audio(original_video_path, new_video_path):
	folder_name = original_video_path.split('/')[-1].split('.')[0]
	clip = mpe.VideoFileClip(original_video_path)
	new_clip = mpe.VideoFileClip(new_video_path)
	final_clip = new_clip.set_audio(clip.audio)
	final_clip.write_videofile(new_video_path)

parser = argparse.ArgumentParser()
parser.add_argument('model_path', type=str, help='REQUIRED: specify path of the model being used')
args = parser.parse_args()

extract_audio('test_img/test_clip.mp4', 'test_img/test_clip.mp3')
# copy_audio('test_img/test_clip.mp4', 'test_img/test_clip_upscaled_copy.mp4')
# combine_frames('test_img/test_clip.mp4', 'test_img/test_clip_upscaled_copy.mp4')
# audio = upscale('test_img/test_clip.mp4')
# clear_temp('tmp/test_clip')
# 'test_img/test_clip_upscaled.mp4'