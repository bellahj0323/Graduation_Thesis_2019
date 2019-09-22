import os
import cv2
import argparse
import numpy as np
from tqdm import tqdm
from PIL import Image

#argparse
parser = argparse.ArgumentParser()
parser.add_argument('--load_path', help='load_path', type=str)
parser.add_argument('--save_path', help='save_path', type=str)
args = parser.parse_args()


def get_path():
  load_path = args.load_path
  save_path = args.save_path
  return load_path, save_path

def get_video_list(load_path):
  video_list = os.listdir(load_path)
  return video_list
  # Train001...Train100

def make_dir(save_path):
  save_path = os.path.abspath(save_path)
  if not os.path.isdir(save_path):
    os.mkdir(save_path)
    
def get_video_frame(video_name, load_path, save_path):
  save_path = os.path.abspath(save_path)
  video_save_path = os.path.join(save_path, video_name)
  video_save_path = os.path.abspath(video_save_path)
  
  if not os.path.isdir(video_save_path):
    os.mkdir(video_save_path)
    
  # load frames
  video_load_path = os.path.join(load_path, video_name)
  video_load_path = os.path.abspath(video_load_path)
  frames = os.listdir(video_load_path)
  # 001.tif ...
  
  for f in frames:
    frame = cv2.resize(frame, dsize=(182,120), interpolation=cv2.INTER_LINEAR)
    os.chdir(video_save_path)
    cv2.imwrite('{}.png'.format(f[:-4]), frame)
    
    
def main():
  load_path, save_path = get_path()
  video_list = get_video_list(load_path)
  make_dir(save_path)
  for i in video_list:
    get_video_frame(i, load_path, save_path)
    
  print('전체 완료')
  
if __name__ == '__main__':
  main()
