{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Using TensorFlow backend.\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch 1/3\n",
      "6336/6428 [============================>.] - ETA: 1s - loss: 0.0179"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/anaconda3/lib/python3.6/site-packages/keras/engine/training.py:1569: UserWarning: Epoch comprised more than `samples_per_epoch` samples, which might affect learning results. Set `samples_per_epoch` correctly to avoid this warning.\n",
      "  warnings.warn('Epoch comprised more than '\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "6432/6428 [==============================] - 159s - loss: 0.0179 - val_loss: 0.0151\n",
      "Epoch 2/3\n",
      "6432/6428 [==============================] - 147s - loss: 0.0123 - val_loss: 0.0118\n",
      "Epoch 3/3\n",
      "6420/6428 [============================>.] - ETA: 0s - loss: 0.0114"
     ]
    }
   ],
   "source": [
    "import os\n",
    "import csv\n",
    "from sklearn.utils import shuffle\n",
    "\n",
    "# Read the log data\n",
    "samples = []\n",
    "with open('./data/driving_log.csv') as csvfile:\n",
    "    reader = csv.reader(csvfile)\n",
    "    next(reader, None)\n",
    "    for line in reader:\n",
    "        samples.append(line)\n",
    "\n",
    "shuffle(samples)\n",
    "# Training and validation data split\n",
    "from sklearn.model_selection import train_test_split\n",
    "train_samples, validation_samples = train_test_split(samples, test_size=0.2)\n",
    "\n",
    "import cv2\n",
    "import numpy as np\n",
    "import sklearn\n",
    "\n",
    "# Making generator to load data into RAM in batches, instead of loading them all at once\n",
    "def generator(samples, batch_size=32):\n",
    "    num_samples = len(samples)\n",
    "    while 1: # Loop forever so the generator never terminates\n",
    "        shuffle(samples)\n",
    "        for offset in range(0, num_samples, batch_size):\n",
    "            batch_samples = samples[offset:offset+batch_size]\n",
    "\n",
    "            images = []\n",
    "            angles = []\n",
    "            # Load images from the central, left and right cameras, and the angles for turning\n",
    "            for batch_sample in batch_samples: \n",
    "                img_center = './data/IMG/'+batch_sample[0].split('/')[-1]\n",
    "                img_left = './data/IMG/'+batch_sample[1].split('/')[-1]\n",
    "                img_right = './data/IMG/'+batch_sample[2].split('/')[-1]\n",
    "                center_image = cv2.imread(img_center)\n",
    "                left_image = cv2.imread(img_left)\n",
    "                right_image = cv2.imread(img_right)\n",
    "                # Adding angles for left and right cameras\n",
    "                correction = 0.1\n",
    "                center_angle = float(batch_sample[3])\n",
    "                left_angle = float(batch_sample[3]) + correction\n",
    "                right_angle = float(batch_sample[3]) - correction\n",
    "                # Collecting images and angles\n",
    "                images.append(center_image)\n",
    "                images.append(left_image)\n",
    "                images.append(right_image)\n",
    "                angles.append(center_angle)\n",
    "                angles.append(left_angle)\n",
    "                angles.append(right_angle)\n",
    "            \n",
    "            X_train = np.array(images)\n",
    "            y_train = np.array(angles)\n",
    "            for image, angle in zip(X_train, y_train): # Data augment\n",
    "                image_flipped = np.fliplr(image)\n",
    "                angle_flipped = -angle\n",
    "                images.append(image_flipped)\n",
    "                angles.append(angle_flipped)\n",
    "            yield sklearn.utils.shuffle(X_train, y_train)\n",
    "\n",
    "# compile and train the model using the generator function\n",
    "\n",
    "# Get train and validation data into generator\n",
    "train_generator = generator(train_samples, batch_size=32)\n",
    "validation_generator = generator(validation_samples, batch_size=32)\n",
    "\n",
    "from keras.models import Sequential, Model\n",
    "from keras.layers import Flatten, Dense, MaxPooling2D, Dropout, Activation\n",
    "#from keras.layers.core import Lambda \n",
    "from keras.layers.convolutional import Cropping2D, Convolution2D\n",
    "from keras.layers.normalization import BatchNormalization\n",
    "\n",
    "# Define size of input\n",
    "row, col, ch = 160, 320, 3\n",
    "input_shape=(row, col, ch)\n",
    "\n",
    "# Define the neural network\n",
    "model = Sequential()\n",
    "# Preprocess incoming data, centered around zero with small standard deviation \n",
    "#model.add(Lambda(lambda x: x/127.5 - 1., input_shape=input_shape))\n",
    "\n",
    "# Crop and normalize data\n",
    "model.add(Cropping2D(cropping=((50,20), (0,0)), input_shape=input_shape))\n",
    "model.add(BatchNormalization(axis=-1))\n",
    "\n",
    "model.add(Convolution2D(24, 3, 3, subsample=(2,2)))\n",
    "model.add(Activation('relu'))\n",
    "model.add(Convolution2D(24, 3, 3, subsample=(2,2)))\n",
    "model.add(Activation('relu'))\n",
    "model.add(MaxPooling2D((2,2), strides=(2,2)))\n",
    "\n",
    "model.add(Convolution2D(36, 3, 3))\n",
    "model.add(Activation('relu'))\n",
    "model.add(Convolution2D(48, 3, 3))\n",
    "model.add(Activation('relu'))\n",
    "\n",
    "model.add(Convolution2D(64, 3, 3))\n",
    "model.add(Activation('relu'))\n",
    "model.add(Convolution2D(64, 3, 3))\n",
    "model.add(Activation('relu'))\n",
    "\n",
    "model.add(Flatten())\n",
    "model.add(Dense(800, activation='relu'))\n",
    "model.add(Dense(300, activation='relu'))\n",
    "model.add(Dense(50, activation='relu'))\n",
    "model.add(Dense(10, activation='relu'))\n",
    "model.add(Dense(1))\n",
    "\n",
    "# Complie and train the model\n",
    "model.compile(loss='mse', optimizer='adam')\n",
    "model.fit_generator(train_generator, \n",
    "                                      samples_per_epoch=len(train_samples), \n",
    "                                      validation_data=validation_generator,\n",
    "                                      nb_val_samples=len(validation_samples), \n",
    "                                      nb_epoch=3)\n",
    "model.save('model.h5')\n",
    "print ('model saved!')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.6.4"
  },
  "widgets": {
   "state": {},
   "version": "1.1.2"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
