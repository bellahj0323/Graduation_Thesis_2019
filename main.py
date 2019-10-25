import argparse
import pandas as pd
import numpy as np
import scipy.stats as sp
import matplotlib.pyplot as plt

import keras
import keras.layers as L
import keras.backend as K
from keras.callbacks import ModelCheckpoint

from dataload import Dataset
from model import ConvLSTM


# argparse
parser = argparse.ArgumentParser()
parser.add_argument('--train', type=str, default='train')
parser.add_argument('--data_path', type=str)
parser.add_argument('--save_path', type=str)

parser.add_argument('--seq', nargs='+', type=int, default=[1, 5, 9])
parser.add_argument('--offset', type=int, default=15)

parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--steps_per_epoch', type=int, default=100)
parser.add_argument('--batch_size', type=int, default=4)
parser.add_argument('--batch_per_video', type=int, default=1)

args = parser.parse_args()

def train(dataload, model, epochs, steps_per_epoch, save_path):
  checkpoint = ModelCheckpoint('{}.h5'.format(save_path), monitor='loss', verbose=1, save_best_only=True, mode='min')
  callbacks_list = [checkpoint]
  
  history = model.fit_generator(
    generator=dataload,
    epochs = epochs,
    steps_per_epoch = steps_per_epoch,
    callbacks = callbacks_list
  )
  history_df = pd.DataFrame(history.history)
  history_df.to_csv('{}.csv'.format(save_path), index=False)
  
  
def test(model, x, y, batch_size):
  for i in range(int(len(y) / batch_size)):
    batch_x = [j[i:i+batch_size] for j in x]
    batch_y = y[i:i+batch_size]
    pred = model.predict(batch_x)
    if i == 0:
      result = pred
    else:
      result = np.concatenate((result, pred), axis=0)
      
  return result
  
  
def main(args):
  dataset = Dataset(args.data_path, args.seq, args.offset, args.batch_size, args.batch_per_video)
  optimizer = keras.optimizers.Adam(lr=1e-4)
  model = ConvLSTM()
  
  if args.train == 'train':
    dataload = dataset.train_loader()
    train(dataload, model, args.epochs, args.steps_per_epoch, args.save_path)
    x, y = next(dataload)
    pred = model.predict(x)
    
  elif args.train == 'test':
    video_idx = int(input('test할 동영상 인덱스를 입력하세요.'))
    x, y = dataset.test_loader(video_idx)
    model = load_model(model, args.save_path)
    pred = test(model, x, y, args.batch_size)
    
if __name__ == '__main__':
  main(args)
  
