import csv
import cv2
import numpy as np

lines = []
base_dir = 'gs-data/basic1'
with open('{}/driving_log.csv'.format(base_dir)) as csv_file:
    reader = csv.reader(csv_file)
    for line in reader:
        lines.append(line)

images = []
angles = []
for line in lines:
    path = line[0]
    file_name = path.split('/')[-1]
    new_path = '{}/IMG/{}'.format(base_dir, file_name)
    image = cv2.imread(new_path)
    angle = float(line[3])
    images.append(image)
    angles.append(angle)
    image_flipped = np.fliplr(image)
    images.append(image_flipped)
    angles.append(-angle)

X_train = np.array(images)
y_train = np.array(angles)

from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Lambda
from keras.layers import Cropping2D
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D

model = Sequential()
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160, 320, 3)))
model.add(Cropping2D(cropping=((70, 20), (0,0))))
model.add(Conv2D(filters=24, kernel_size=(5,5), strides=(2,2), activation='relu'))
model.add(Conv2D(filters=36, kernel_size=(5,5), strides=(2,2), activation='relu'))
model.add(Conv2D(filters=48, kernel_size=(5,5), strides=(2,2), activation='relu'))
model.add(Conv2D(filters=64, kernel_size=(3,3), activation='relu'))
model.add(Conv2D(filters=64, kernel_size=(3,3), activation='relu'))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))



model.compile(
    loss='mse',
    optimizer='adam',
)


history = model.fit(
    X_train,
    y_train,
    epochs=5,
    validation_split=0.2,
    shuffle=True,
)

model.save('model.h5')
