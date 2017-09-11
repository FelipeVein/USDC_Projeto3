import csv
import cv2
import numpy as np

lines = []
with open('data3/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

images = []
measurements = []
for line in lines:
    for i in range(0,3):
        source_path = line[i]
        filename = source_path.split('\\')[-1]
        current_path = 'data3/IMG/'+filename
        image = cv2.imread(current_path)
        images.append(image)
        measurement = float(line[3])
        if(i == 1):
            measurement = measurement + 0.25
        if(i == 2):
            measurement = measurement - 0.25
        measurements.append(measurement)
        images.append(np.fliplr(image))
        measurements.append(-measurement)
    

X_train = np.array(images)
y_train = np.array(measurements)


from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Activation, Dropout
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D

#model = Sequential()
#model.add(Cropping2D(cropping=((50,20), (0,0)), input_shape=(160,320,3)))
#model.add(Lambda(lambda x: x/ 255.0 - 0.5))
#model.add(Conv2D(6,(5,5),activation="relu"))
#model.add(MaxPooling2D())
#model.add(Conv2D(6,(5,5),activation="relu"))
#model.add(MaxPooling2D())
#model.add(Flatten())
#model.add(Dense(120))
#model.add(Activation('relu'))
#model.add(Dropout(0.5))
#model.add(Dense(84))
#model.add(Activation('relu'))
#model.add(Dropout(0.5))
#model.add(Dense(1))
model = Sequential()
model.add(Cropping2D(cropping=((70,25),(0,0)),input_shape = (160,320,3)))
model.add(Lambda(lambda x: x/255.0 - 0.5))
model.add(Conv2D(24,(5,5),strides=(2,2),activation="relu"))
model.add(Conv2D(36,(5,5),strides=(2,2),activation="relu"))
model.add(Conv2D(48,(5,5),strides=(2,2),activation="relu"))
model.add(Conv2D(64,(3,3),activation="relu"))
model.add(Conv2D(64,(3,3),activation="relu"))
model.add(Flatten())
model.add(Dense(100))
model.add(Activation("relu"))
model.add(Dropout(0.5))
model.add(Dense(50))
model.add(Activation("relu"))
model.add(Dropout(0.5))
model.add(Dense(10))
model.add(Dense(1))


model.compile(loss='mse', optimizer = 'adam')
hist = model.fit(X_train,y_train,validation_split=0.2,shuffle=True, epochs=10, verbose = 2)
print(hist.history)

model.save('model.h5')
