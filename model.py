from keras.models import Model, Sequential
from keras.layers.convolutional import Conv3D, Conv2DTranspose
from keras.layers.convolutional_recurrent import ConvLSTM2D
from keras.layers.normalization import BatchNormalization

# (n_frames, rows, cols, channels) input shape
# (None, 158, 238, 1) -> resized (None, 128, 128, 1)

def ConvLSTM(optimizer):

  model = Sequential()

  
  model.add(ConvLSTM2D(strides=2, padding='same', activation='relu',
                        kernel_initializer='he_normal', kernel_size=(3,3),
                        input_shape=(None, 158, 238, 1), return_sequences=False))

  model.add(BatchNormalization())
  
  model.compile(optimizer=optimizer, loss='mean_squared_error')
  
  return model
