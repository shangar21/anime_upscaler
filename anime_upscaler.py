import frame_esrgan
import cv2
from tqdm import tqdm
import os
import argparse
import shutil
import image_slicer
from image_slicer import join
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import subprocess


parser = argparse.ArgumentParser()
parser.add_argument('-m', '--model_path', type=str, help='REQUIRED: specify path of the model being used')
parser.add_argument('-i', '--input', type=str, help='REQUIRED: specify path of the image you want to upscale')
parser.add_argument('-o', '--output', type=str, help='REQUIRED: specify path where you want to save image')
parser.add_argument('-s', '--slice', nargs='?', type=int, const=4, help='OPTIONAL: specify weather to split frames, recommended to use to help with VRAM unless you got a fucken quadro or something' )
parser.add_argument('-a', '--audio', action='store_true', help='OPTIONAL: specify weather you want to copy audio from source as well')
parser.add_argument('-c', '--clear_temp', action='store_true', help='OPTIONAL: specify weather you want to clear temporary folder with upscaled frames after you are finished with final video')
args = parser.parse_args()

def extract_frames(vid_path, save=''):
    vid = cv2.VideoCapture(vid_path)
    images = []
    count = 0
    success, image = vid.read()
    while success:
        if not save:
            images.append(image)
        else:
            print('saving frame {}...'.format(count))
            cv2.imwrite(save.format(count), image)
            print('done saving frame {}...'.format(count))
        success, image = vid.read()
        count += 1
    return images

def get_fps(vid_path):
    vid = cv2.VideoCapture(vid_path)
    return vid.get(cv2.CAP_PROP_FPS)

def create_temp_folder(vid_path):
    if os.path.exists('tmp'):
        folder_name = vid_path.split('/')[-1].split('.')[0]
        os.mkdir('tmp/{}'.format(folder_name))
    else:
        os.mkdir('tmp')
        create_temp_folder(vid_path)

def get_dir(path):
    if not os.path.exists(path):
        os.mkdir(path)
    return path

def setup_frames(vid_path, slice=None):
    folder_name = vid_path.split('/')[-1].split('.')[0]
    images = extract_frames(vid_path)
    create_temp_folder(vid_path)
    os.mkdir('tmp/{}/original'.format(folder_name))
    slices = []
    for i in tqdm(range(len(images))):
        cv2.imwrite('tmp/{}/original'.format(folder_name)+'/frame_{}.png'.format(i), images[i])
    os.mkdir('tmp/{}/upscaled'.format(folder_name))

def upscale(vid_path, slice=None):
    folder_name = vid_path.split('/')[-1].split('.')[0]
    print('extracting frames...')
    setup_frames(vid_path)
    print('upscaling...')
    for i in tqdm(os.listdir('tmp/{}/original'.format(folder_name))):
        if slice:
            out = frame_esrgan.upscale_slice(args.model_path,  'tmp/{}/original/{}'.format(folder_name, i), slice)
        else:
            out = frame_esrgan.upscale(args.model_path, 'tmp/{}/original/{}'.format(folder_name, i))
        cv2.imwrite('tmp/{}/upscaled/{}'.format(folder_name, i), out)

def combine_frames(video_path, new_video_path):
    folder_name = video_path.split('/')[-1].split('.')[0]
    images = [img for img in os.listdir('tmp/{}/upscaled'.format(folder_name))]
    height, width, layers = cv2.imread('tmp/{}/upscaled/frame_0.png'.format(folder_name)).shape
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    fps = get_fps(video_path)
    video = cv2.VideoWriter(new_video_path, fourcc, fps, (width, height))
    for i in tqdm(range(len(images))):
        video.write(cv2.imread('tmp/{}/upscaled/frame_{}.png'.format(folder_name, i)))
    cv2.destroyAllWindows()
    video.release()

def copy_audio(original_video_path, new_video_path, new_name=''):
    #ffmpeg -i input_0.mp4 -i input_1.mp4 -c copy -map 0:v:0 -map 1:a:0 -shortest out.mp4
    tmp_name = new_video_path.split('.')[0] + '_tmp.' + new_video_path.split('.')[-1]
    subprocess.run([
        'ffmpeg',
        '-i',
        new_video_path,
        '-i',
        original_video_path,
        '-c',
        'copy',
        '-map',
        '0:v:0',
        '-map',
        '1:a:0',
        '-shortest',
        tmp_name
    ])

    os.replace(tmp_name, new_video_path)


if __name__ == '__main__':
    if args.model_path and args.input and args.output:
        try:
            upscale(args.input, slice=args.slice)
            combine_frames(args.input, args.output)
            if args.audio:
                copy_audio(args.input, args.output)
            if args.clear_temp:
                shutil.rmtree('tmp')
        except Exception as e:
            print(e)
            shutil.rmtree('tmp')
