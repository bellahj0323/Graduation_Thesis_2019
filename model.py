from keras.models import Model, Sequential
from keras.layers.convolutional import Conv3D, Conv2DTranspose
from keras.layers.convolutional_recurrent import ConvLSTM2D
from keras.layers.normalization import BatchNormalization

# (n_frames, rows, cols, channels) input shape
# (None, 158, 238, 1) -> resized (None, 128, 128, 1)

def ConvLSTM(optimizer):

  model = Sequential()

  model.add(ConvLSTM2D(filters=20, kernel_size=(3,3), strides=1, padding='same', activation='relu',
                        kernel_initializer='he_normal',data_format='channels_last',
                        input_shape=(None, 128, 128, 1), return_sequences=True))

  model.add(ConvLSTM2D(filters=1, kernel_size=(3,3), strides=2, padding='same', activation='relu',
                     data_format='channels_last', kernel_initializer='he_normal', return_sequences=False))

  model.add(BatchNormalization())
  
  model.add(Conv2DTranspose(filters=1, kernel_size=(3,3)
                          ,strides=2, padding='same', data_format='channels_last'
                          ,activation='relu',kernel_initializer='he_normal'))
  
  model.compile(optimizer=optimizer, loss='mean_squared_error')
  
  return model
