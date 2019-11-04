from __future__ import print_function
import tensorflow as tf
from keras.datasets import cifar100
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dropout, Activation, Convolution2D, GlobalAveragePooling2D, merge, MaxPooling2D
from keras.utils import np_utils
from keras.optimizers import SGD
from keras import backend as K
from keras.models import Model
from keras.layers.core import Lambda
from keras.callbacks import ModelCheckpoint
from keras import regularizers
import pandas



import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


# import tensorflow as tf
# from keras.backend.tensorflow_backend import set_session
# config = tf.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.7
# set_session(tf.Session(config=config))


K.set_image_dim_ordering('tf')
def make_parallel(model, gpu_count):
    def get_slice(data, idx, parts):
        shape = tf.shape(data)
        size = tf.concat(0, [ shape[:1] // parts, shape[1:] ])
        stride = tf.concat(0, [ shape[:1] // parts, shape[1:]*0 ])
        start = stride * idx
        return tf.slice(data, start, size)

    outputs_all = []
    for i in range(len(model.outputs)):
        outputs_all.append([])

    #Place a copy of the model on each GPU, each getting a slice of the batch
    for i in range(gpu_count):
        with tf.device('/gpu:%d' % i):
            with tf.name_scope('tower_%d' % i) as scope:

                inputs = []
                #Slice each input into a piece for processing on this GPU
                for x in model.inputs:
                    input_shape = tuple(x.get_shape().as_list())[1:]
                    slice_n = Lambda(get_slice, output_shape=input_shape, arguments={'idx':i,'parts':gpu_count})(x)
                    inputs.append(slice_n)                

                outputs = model(inputs)
                
                if not isinstance(outputs, list):
                    outputs = [outputs]
                
                #Save all the outputs for merging back together later
                for l in range(len(outputs)):
                    outputs_all[l].append(outputs[l])

    # merge outputs on CPU
    # with tf.device('/cpu:0'):0.91060
    #     merged = []
    #     for outputs in outputs_all:
    #         merged.append(merge(outputs, mode='concat', concat_axis=0))
    #
    #     return Model(input=model.inputs, output=merged)

batch_size = 64
nb_classes = 100
nb_epoch = 3300


rows, cols = 32, 32

channels = 3

(X_train, y_train), (X_test, y_test) = cifar100.load_data()
print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

print (X_train.shape[1:])

Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)



model = Sequential()

model.add(Convolution2D(96, 3, 3, border_mode = 'same', input_shape=(32, 32, 3)))
model.add(Activation('relu'))
model.add(Dropout(0.2))
# model.add(PReLU(alpha_initializer='zeros'))
model.add(Convolution2D(96, 3, 3,border_mode='same', kernel_regularizer=regularizers.l2(0.001)))
model.add(Activation('relu'))
# model.add(PReLU(alpha_initializer='zeros'))
model.add(Convolution2D(96, 3, 3, border_mode='same', subsample = (2,2)))
model.add(Dropout(0.5))

model.add(Convolution2D(192, 3, 3, border_mode = 'same', kernel_regularizer=regularizers.l2(0.002)))
model.add(Activation('relu'))
# model.add(PReLU(alpha_initializer='zeros'))
model.add(Convolution2D(192, 3, 3,border_mode='same', kernel_regularizer=regularizers.l2(0.002)))
model.add(Activation('relu'))
model.add(Dropout(0.5))
# model.add(PReLU(alpha_initializer='zeros'))
model.add(Convolution2D(192, 3, 3,border_mode='same', subsample = (2,2)))
# model.add(Dropout(0.5))

model.add(Convolution2D(192, 3, 3, border_mode = 'same', kernel_regularizer=regularizers.l2(0.002)))
model.add(Activation('relu'))
model.add(Dropout(0.5))
# model.add(PReLU(alpha_initializer='zeros'))
model.add(Convolution2D(192, 1, 1,border_mode='valid', kernel_regularizer=regularizers.l2(0.002)))
model.add(Activation('relu'))
# model.add(PReLU(alpha_initializer='zeros'))
model.add(Convolution2D(100, 1, 1, border_mode='valid'))

model.add(GlobalAveragePooling2D())
model.add(Activation('softmax'))
# model = make_parallel(model, 4)

# model.load_weights("weights1.hdf5")

sgd = SGD(lr=0.015, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])


print (model.summary())

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

datagen1 = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
        width_shift_range=0.10,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.10,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False) 


datagen1.fit(X_train)

filepath="weights3.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=False, mode='max')

callbacks_list = [checkpoint]
    # Fit the model on the batches generated by datagen.flow().
history_callback = model.fit_generator(datagen1.flow(X_train, Y_train,
                                     batch_size=batch_size),
                        samples_per_epoch=X_train.shape[0],
                        nb_epoch=nb_epoch, validation_data= (X_test, Y_test),
                                       callbacks=callbacks_list, verbose=2)



pandas.DataFrame(history_callback.history).to_csv("history100.csv")


