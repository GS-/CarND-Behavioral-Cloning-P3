import sys
import csv
import cv2
import numpy as np
from sklearn.utils import shuffle

images = []
angles = []


source_images = []

def load_source_data(base_dir):
    with open("{}/driving_log.csv".format(base_dir)) as csv_file:
        reader = csv.reader(csv_file)
        for line in reader:
            data = []
            path = line[0]
            file_name = path.split('/')[-1]
            new_path = "{}/IMG/{}".format(base_dir, file_name)
            angle = float(line[3])
            data.append(new_path)
            data.append(angle)
            source_images.append(data)


def load_data(base_dir):
    lines = []
    with open("{}/driving_log.csv".format(base_dir)) as csv_file:
#    with open(f'{base_dir}/driving_log.csv') as csv_file:
        reader = csv.reader(csv_file)
        for line in reader:
            lines.append(line)
    for line in lines:
        path = line[0]
        file_name = path.split('/')[-1]
        new_path = "{}/IMG/{}".format(base_dir, file_name)
        image = cv2.imread(new_path)
        angle = float(line[3])
        images.append(image)
        angles.append(angle)
        image_flipped = np.fliplr(image)
        images.append(image_flipped)
        angles.append(-angle)


load_source_data('gs-data/tricky')
load_source_data('gs-data/track2x1')
print("Loading data...")
load_source_data('gs-data/track2x2')
load_source_data('gs-data/recovery1')
load_source_data('gs-data/recovery2')
print(len(source_images))
source_images = shuffle(source_images)




#load_data('gs-data/basic1')
#load_data('gs-data/starter')
#load_data('gs-data/tricky')
#load_data('gs-data/track2x1')
#print("Loading data...")
#load_data('gs-data/track2x2')
#load_data('gs-data/recovery1')
#load_data('gs-data/recovery2')
#print(len(images))
#sys.exit()

from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(source_images, test_size=0.2)
print(len(train_samples))
print(len(validation_samples))

BATCH_SIZE = 500
TRAIN_STEPS = (len(train_samples) // BATCH_SIZE) + 1
VALIDATE_STEPS = (len(validation_samples) // BATCH_SIZE) + 1
EPOCHS=5
DROPOUT_RATE=0.15
print(TRAIN_STEPS)
#sys.exit()

def generator(samples, batch_size=BATCH_SIZE):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            images = []
            angles = []
            for batch_sample in batch_samples:
                file_name = batch_sample[0]
                angle = float(batch_sample[1])
                image = cv2.imread(file_name)
                images.append(image)
                angles.append(angle)
                image_flipped = np.fliplr(image)
                images.append(image_flipped)
                angles.append(-angle)

            yield np.array(images), np.array(angles)


# compile and train the model using the generator function
train_generator = generator(train_samples)
validation_generator = generator(validation_samples)


from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Lambda, Dropout
from keras.layers import Cropping2D
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D

def first_model():
    model = Sequential()
    model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160, 320, 3)))
    model.add(Cropping2D(cropping=((70, 20), (0,0))))
    model.add(Conv2D(filters=24, kernel_size=(5,5), strides=(2,2), activation='relu'))
    model.add(Conv2D(filters=36, kernel_size=(5,5), strides=(2,2), activation='relu'))
    model.add(Conv2D(filters=46, kernel_size=(5,5), strides=(2,2), activation='relu'))
    model.add(Conv2D(filters=64, kernel_size=(3,3), activation='relu'))
    model.add(Conv2D(filters=64, kernel_size=(3,3), activation='relu'))
    model.add(Flatten())
    model.add(Dense(100))
    model.add(Dense(50))
    model.add(Dense(10))
    model.add(Dense(1))
    return model


def gs_model():
    model = Sequential()
    model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160, 320, 3)))
    model.add(Cropping2D(cropping=((70, 20), (0,0))))
    model.add(Conv2D(filters=24, kernel_size=(3,3), strides=(2,2), activation='relu'))
    model.add(Dropout(DROPOUT_RATE))
    model.add(Conv2D(filters=48, kernel_size=(4,4), strides=(1,1), activation='relu'))
    model.add(Conv2D(filters=96, kernel_size=(5,5), strides=(2,2), activation='relu'))
    model.add(Dropout(DROPOUT_RATE))
    model.add(MaxPooling2D())
    model.add(Dropout(DROPOUT_RATE))
    model.add(Conv2D(filters=128, kernel_size=(3,3), activation='relu'))
    model.add(Dropout(DROPOUT_RATE))
    model.add(Conv2D(filters=256, kernel_size=(4,4), activation='relu'))
    model.add(Dropout(DROPOUT_RATE))
    model.add(MaxPooling2D())
    model.add(Dropout(DROPOUT_RATE))
    model.add(Flatten())
    model.add(Dense(2000))
    model.add(Activation('relu'))
    model.add(Dense(1000))
    model.add(Activation('relu'))
    model.add(Dropout(DROPOUT_RATE))
    model.add(Dense(500))
    model.add(Dense(100))
    model.add(Activation('relu'))
    model.add(Dense(50))
    model.add(Activation('relu'))
    model.add(Dropout(DROPOUT_RATE))
    model.add(Dense(10))
    model.add(Activation('relu'))
    model.add(Dense(1))
    return model
       

# for layer in model.layers:
#     print(layer.get_output_at(0).get_shape().as_list())


def fit_model(model, model_save_name):
    model.summary()
    model.compile(
        loss='mse',
        optimizer='adam',
    )

    #history = model.fit(
    #    X_train,
    #    y_train,
    #    epochs=3,
    #    validation_split=0.25,
    #    shuffle=True,
    #)

    history = model.fit_generator(
        train_generator,
        steps_per_epoch=TRAIN_STEPS,
        validation_data=validation_generator,
        validation_steps=VALIDATE_STEPS,
        epochs=EPOCHS,
    )

    model.save(model_save_name)

model = gs_model()
fit_model(
    model=model,
    model_save_name='model.h5',
)
