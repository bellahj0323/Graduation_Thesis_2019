import os
import cv2
import numpy as np
import pandas as pd
from PIL import Image

class Dataset:
  np.random.seed(123)
  def __init__(self, directory, offset, seq, batch_size, batch_per_video):
    self.directory = directory
    self.train_dir = directory + 'Train'
    self.val_dir = directory + 'Validation'
    self.videos = [os.path.join(self.train_dir, j) for j in os.listdir(self.train_dir)]
                                               # list of videos ex) Train001, Train002, ...
    self.valvideos = [os.path.join(self.val_dir, j) for j in os.listdir(self.val_dir)]
                                               # list of videos ex) Val001, Val002, ...
    self.batch_size = batch_size
    self.batch_per_video = batch_per_video
    self.seq = seq  # list
    self.offset = offset 
    
  def _load_frame(self, frame_path):
    frame = Image.open(frame_path) # (158, 238)
    frame = frame.resize((128, 128))
    arr = np.array(frame)
    arr = np.expand_dims(arr, -1) # (128, 128, 1)
    arr = arr.astype('float32')
    arr /= 255
    return arr
    
  def random_frames(self, version):
    if version == 1 :
      video_idx = np.random.randint(low=0, high=len(self.videos))
      video = self.videos[video_idx]
    elif version == 0 :
      video_idx = np.random.randint(low=0, high=len(self.valvideos))
      video = self.valvideos[video_idx]
      
    # video 선택 했으니까 frame np.array로 불러오기
    video_len = int(len(os.listdir(video))) - 2 - self.offset - self.seq[-1]
    frames = [os.path.join(video, '%03d.tif' %i) for i in range(video_len)]
    
    idx_y = np.random.randint(self.offset, len(frames), self.batch_per_video)
    idx_x = [[y - self.offset + x for x in self.seq] for y in idx_y]
    
    frame_y = np.array([self._load_frame(frames[i]) for i in idx_y])
    frame_x = []
    
    
    for x in zip(*idx_x):
      temp = np.array([self._load_frame(frames[j]) for j in x])
      frame_x.append(temp)
    return frame_x, frame_y
    # [[1st, 1st, 1st],[2nd,2nd,2nd],[3rd,3rd,3rd]]
  
    
  def train_loader(self):
    while True:
      for i in range(int(self.batch_size/self.batch_per_video)):
        x, y = self.random_frames(1)
        if i == 0:
          batch_x = x
          batch_y = y
        else:
          batch_x = np.concatenate((batch_x, x), axis=1)
          batch_y = np.concatenate((batch_y, y), axis=0)
          
      batch_x = list(batch_x)
      yield batch_x, batch_y

  def validation_loader(self):
    while True:
      x, y = self.random_frames(0)
      x = list(x)

      yield x, y
    
      
  def test_loader(self, video_idx):
    video = self.videos[video_idx]
    print("Test 할 동영상은 %s 입니다." %video)
    # video 선택 했으니까 frame np.array로 불러오기    
    video_len = int(len(os.listdir(video))) - 2 - self.offset - self.seq[-1]
    frames = [os.path.join(video, '%03d.tif' %i) for i in range(video_len)]
    
    idx_y = np.arange(self.offset , len(frames))
    idx_x = [[y - self.offset + x for x in self.seq] for y in idx_y]
    
    frame_y = np.array([self._load_frame(frames[i]) for i in idx_y])
    frame_x = []
    for x in zip(*idx_x):
      temp = np.array([self._load_frame(frames[i]) for i in x])
      frame_x.append(temp)
      
    return frame_x, frame_y
  
