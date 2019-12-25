from keras.models import Model, Sequential
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.convolutional_recurrent import ConvLSTM2D
from keras.layers.normalization import BatchNormalization
from keras.layers import Input
from keras.layers import Reshape
from keras.layers import Concatenate
import numpy as np
import pylab as plt

def CConvLSTM(optimizer, layer_num, channel_num):
  input_shape=(256, 256, 1)
  input1 = Input(shape=input_shape)
  input2 = Input(shape=input_shape)
  input3 = Input(shape=input_shape)

  encoder = Sequential(name='encoder')

  #for i in range(layer_num):
  #  if i == 0:
  #    encoder.add(Conv2D(channel_num*(2**i), (3,3), strides=2, activation='relu', padding='same', kernel_initializer='he_normal', input_shape=input_shape))
  #  else:
  #    encoder.add(Conv2D(channel_num*(2**i), (3,3), strides=2, activation='relu', padding='same', kernel_initializer='he_normal'))
  #encoder.add(Conv2D(channel_num*(2**i), (3,3), strides=1, activation='relu', padding='same', kernel_initializer='he_normal'))

  encoder.add(Conv2D(channel_num, (3,3), strides=1, activation='relu', padding='same', kernel_initializer='he_normal'))
  encoder.add(Conv2D(channel_num*2, (3,3), strides=1, activation='relu', padding='same', kernel_initializer='he_normal'))

  encoded1 = encoder(input1)
  encoded2 = encoder(input2)
  encoded3 = encoder(input3)

  reshape = (1, *encoder.output_shape[1:])
  reshaped1 = Reshape(reshape)(encoded1)
  reshaped2 = Reshape(reshape)(encoded2)
  reshaped3 = Reshape(reshape)(encoded3)

  print(encoded1.shape)
  print(reshaped1.shape)

  concat = Concatenate(axis=1)([reshaped1, reshaped2, reshaped3])
  print(concat.shape)
  convlstm = ConvLSTM2D(channel_num, (3,3), strides=1, padding='same', activation='relu', kernel_initializer='he_normal', return_sequences=False)(concat)

  decoder_shape = (i.value for i in convlstm.get_shape()[1:])
  decoder = Sequential(name='decoder')

  #for i in range(layer_num):
  #  if i == 0:
  #    decoder.add(Conv2DTranspose(channel_num*(2**(layer_num-i)), (3,3), strides=2, activation='relu', padding='same', kernel_initializer='he_normal', input_shape=decoder_shape))
  #  else:
  #    decoder.add(Conv2DTranspose(channel_num*(2**(layer_num-i)), (3,3), strides=2, activation='relu', padding='same', kernel_initializer='he_normal'))
  #  decoder.add(Conv2D(channel_num*(2**(layer_num-i)), (3,3), strides=1, activation='relu', padding='same', kernel_initializer='he_normal'))

  decoder.add(Conv2D(channel_num*2, (3,3), strides=1, activation='relu', padding='same', kernel_initializer='he_normal'))
  decoder.add(Conv2D(channel_num, (3,3), strides=1, activation='relu', padding='same', kernel_initializer='he_normal')) 
  decoder.add(Conv2D(1, (3,3), strides=1, activation='sigmoid', padding='same', kernel_initializer='he_normal'))

  output = decoder(convlstm)

  model = Model(inputs=[input1, input2, input3], outputs=output)
  model.compile(optimizer=optimizer, loss='mean_squared_error')
  print(model.summary())
  
  return model

