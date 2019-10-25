from keras.models import Model, Sequential
from keras.layers.convolutional import Conv3D, Conv2DTranspose
from keras.layers.convolutional_recurrent import ConvLSTM2D
from keras.layers.normalization import BatchNormalization

# (n_frames, rows, cols, channels) input shape
# (None, 158, 238, 1)

def ConvLSTM():

  model = Sequential()

  model.add(Conv3D(filters=10, kernel_size=(3,3,3)
                   ,strides=2, padding='same', data_format='channels_last'
                   ,activation='relu', kernel_initializer='he_normal'
                  ,input_shape=(None,158,238,1)))

  model.add(ConvLSTM2D(filters=32, kernel_size=(3,3)
                       ,activation='relu',data_format='channels_last'
                       ,padding='same',return_sequences=False))

  model.add(BatchNormalization())

  model.add(Conv2DTranspose(filters=1, kernel_size=(3,3)
                            ,strides=2, padding='same', data_format='channels_last'
                            ,activation='relu',kernel_initializer='he_normal'))
  
  model.compile(optimizer='adadelta', loss='binary_crossentropy')
  
  return model
