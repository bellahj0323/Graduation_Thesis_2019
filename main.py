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
from model2 import CConvLSTM


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

parser.add_argument('--model_type', type=int, default=0)
# 0=model.py, 1=model2.py

args = parser.parse_args()

def make_image(pred, real):
    fig = plt.figure(figsize=(13, 13))
    ax1 = fig.add_subplot(2, 1, 1)
    ax2 = fig.add_subplot(2, 1, 2)

    ax1.imshow(pred[0][:,:,0])
    ax2.imshow(real[0][:,:,0])
    plt.savefig('train.png')

def train(dataload, model, epochs, steps_per_epoch, save_path):
  checkpoint = ModelCheckpoint('{}.h5'.format(save_path), monitor='loss', verbose=1, save_best_only=True, mode='min')
  callbacks_list = [checkpoint]
  
  history = model.fit_generator(
    generator=dataload,
    epochs = epochs,
    steps_per_epoch = steps_per_epoch,
    callbacks = callbacks_list,
    #use_multiprocessing=True
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
  dataset = Dataset(args.data_path, args.offset, args.seq, args.batch_size, args.batch_per_video)
  optimizer = keras.optimizers.Adam(lr=1e-4)
  if(args.model_type == 0):
    model = ConvLSTM(optimizer)
  elif(args.model_type == 1):
    model = CConvLSTM(optimizer)
  
  if args.train == 'train':
    dataload = dataset.train_loader()
    train(dataload, model, args.epochs, args.steps_per_epoch, args.save_path)
    
    # save model
    model_json = model.to_json()
    with open('{}.json'.format(args.save_path), 'w') as json_file:
      json_file.write(model_json)
    model.save_weights('{}.h5'.format(args.save_path))
    
    # check trained well
    x, y = next(dataload)
    pred = model.predict(x)
    make_image(pred, y)
    
  elif args.train == 'test':
    video_idx = int(input('test할 동영상 인덱스를 입력하세요.'))
    x, y = dataset.test_loader(video_idx)
    # loading model
    try:
      with open('{}.json'.format(args.load_path), 'r') as f:
        test_model = model_from_json(f.read())
    except:
      test_model = model
    test_model.load_weights('{}.h5'.format(args.load_path))
    pred = test(test_model, x, y, args.batch_size)
    
if __name__ == '__main__':
  main(args)
  
