import csv
import cv2
import numpy as np

lines = []
base_dir = 'gs-data/basic1'
with open(f'{base_dir}/driving_log.csv') as csv_file:
    reader = csv.reader(csv_file)
    for line in reader:
        lines.append(line)

images = []
angles = []
for line in lines:
    path = line[0]
    file_name = path.split('/')[-1]
    new_path = f'{base_dir}/IMG/{file_name}'
    image = cv2.imread(new_path)
    angle = float(line[3])
    images.append(image)
    angles.append(angle)

X_train = np.array(images)
y_train = np.array(angles)

from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Lambda
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D

model = Sequential()
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160, 320, 3)))
model.add(Conv2D(filters=32, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D())
model.add(Conv2D(filters=32, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D())
model.add(Flatten())
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
