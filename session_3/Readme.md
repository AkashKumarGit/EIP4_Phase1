1. Validation Accuracy for base network on test data is 83.17

2.# Define the model
model = Sequential()
model.add(SeparableConv2D(48, 3, 3, border_mode='same', input_shape=(32, 32, 3)))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(SeparableConv2D(48, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(SeparableConv2D(96, 3, 3, border_mode='same'))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(SeparableConv2D(96, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(SeparableConv2D(192, 3, 3, border_mode='same'))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(SeparableConv2D(192, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(256))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))
# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

3.
Epoch 1/50
390/390 [==============================] - 28s 72ms/step - loss: 1.6532 - acc: 0.3836 - val_loss: 1.2947 - val_acc: 0.5268
Epoch 2/50
390/390 [==============================] - 26s 67ms/step - loss: 1.2647 - acc: 0.5447 - val_loss: 1.1246 - val_acc: 0.5993
Epoch 3/50
390/390 [==============================] - 26s 66ms/step - loss: 1.1099 - acc: 0.6083 - val_loss: 1.0000 - val_acc: 0.6370
Epoch 4/50
390/390 [==============================] - 26s 66ms/step - loss: 1.0065 - acc: 0.6493 - val_loss: 1.1148 - val_acc: 0.6161
Epoch 5/50
390/390 [==============================] - 26s 66ms/step - loss: 0.9365 - acc: 0.6747 - val_loss: 0.8323 - val_acc: 0.7106
Epoch 6/50
390/390 [==============================] - 26s 66ms/step - loss: 0.8785 - acc: 0.6949 - val_loss: 0.8347 - val_acc: 0.7143
Epoch 7/50
390/390 [==============================] - 26s 67ms/step - loss: 0.8299 - acc: 0.7147 - val_loss: 0.8332 - val_acc: 0.7119
Epoch 8/50
390/390 [==============================] - 26s 66ms/step - loss: 0.7909 - acc: 0.7286 - val_loss: 0.7203 - val_acc: 0.7509
Epoch 9/50
390/390 [==============================] - 26s 66ms/step - loss: 0.7638 - acc: 0.7410 - val_loss: 0.6920 - val_acc: 0.7652
Epoch 10/50
390/390 [==============================] - 26s 66ms/step - loss: 0.7309 - acc: 0.7504 - val_loss: 0.6911 - val_acc: 0.7652
Epoch 11/50
390/390 [==============================] - 26s 66ms/step - loss: 0.7057 - acc: 0.7594 - val_loss: 0.6661 - val_acc: 0.7709
Epoch 12/50
390/390 [==============================] - 26s 66ms/step - loss: 0.6823 - acc: 0.7658 - val_loss: 0.6719 - val_acc: 0.7688
Epoch 13/50
390/390 [==============================] - 26s 66ms/step - loss: 0.6704 - acc: 0.7698 - val_loss: 0.6798 - val_acc: 0.7635
Epoch 14/50
390/390 [==============================] - 26s 66ms/step - loss: 0.6523 - acc: 0.7779 - val_loss: 0.6522 - val_acc: 0.7799
Epoch 15/50
390/390 [==============================] - 26s 66ms/step - loss: 0.6399 - acc: 0.7813 - val_loss: 0.6463 - val_acc: 0.7837
Epoch 16/50
390/390 [==============================] - 26s 66ms/step - loss: 0.6251 - acc: 0.7844 - val_loss: 0.6135 - val_acc: 0.7924
Epoch 17/50
390/390 [==============================] - 26s 66ms/step - loss: 0.6093 - acc: 0.7898 - val_loss: 0.6048 - val_acc: 0.7975
Epoch 18/50
390/390 [==============================] - 26s 66ms/step - loss: 0.5939 - acc: 0.7949 - val_loss: 0.6007 - val_acc: 0.7987
Epoch 19/50
390/390 [==============================] - 26s 65ms/step - loss: 0.5862 - acc: 0.7990 - val_loss: 0.6435 - val_acc: 0.7849
Epoch 20/50
390/390 [==============================] - 26s 66ms/step - loss: 0.5726 - acc: 0.8038 - val_loss: 0.6228 - val_acc: 0.7954
Epoch 21/50
390/390 [==============================] - 26s 65ms/step - loss: 0.5682 - acc: 0.8058 - val_loss: 0.5977 - val_acc: 0.7979
Epoch 22/50
390/390 [==============================] - 26s 65ms/step - loss: 0.5537 - acc: 0.8103 - val_loss: 0.5996 - val_acc: 0.7985
Epoch 23/50
390/390 [==============================] - 27s 68ms/step - loss: 0.5527 - acc: 0.8086 - val_loss: 0.6079 - val_acc: 0.7972
Epoch 24/50
390/390 [==============================] - 26s 67ms/step - loss: 0.5424 - acc: 0.8126 - val_loss: 0.5911 - val_acc: 0.8071
Epoch 25/50
390/390 [==============================] - 26s 67ms/step - loss: 0.5316 - acc: 0.8171 - val_loss: 0.5598 - val_acc: 0.8133
Epoch 26/50
390/390 [==============================] - 26s 67ms/step - loss: 0.5288 - acc: 0.8178 - val_loss: 0.5711 - val_acc: 0.8103
Epoch 27/50
390/390 [==============================] - 26s 68ms/step - loss: 0.5164 - acc: 0.8225 - val_loss: 0.5866 - val_acc: 0.8053
Epoch 28/50
390/390 [==============================] - 26s 68ms/step - loss: 0.5069 - acc: 0.8250 - val_loss: 0.5577 - val_acc: 0.8132
Epoch 29/50
390/390 [==============================] - 26s 67ms/step - loss: 0.5054 - acc: 0.8261 - val_loss: 0.5617 - val_acc: 0.8109
Epoch 30/50
390/390 [==============================] - 26s 68ms/step - loss: 0.5007 - acc: 0.8276 - val_loss: 0.5857 - val_acc: 0.8104
Epoch 31/50
390/390 [==============================] - 26s 67ms/step - loss: 0.4944 - acc: 0.8277 - val_loss: 0.5536 - val_acc: 0.8161
Epoch 32/50
390/390 [==============================] - 26s 67ms/step - loss: 0.4919 - acc: 0.8308 - val_loss: 0.5601 - val_acc: 0.8154
Epoch 33/50
390/390 [==============================] - 26s 67ms/step - loss: 0.4888 - acc: 0.8324 - val_loss: 0.5572 - val_acc: 0.8174
Epoch 34/50
390/390 [==============================] - 26s 67ms/step - loss: 0.4780 - acc: 0.8356 - val_loss: 0.5456 - val_acc: 0.8173
Epoch 35/50
390/390 [==============================] - 26s 67ms/step - loss: 0.4755 - acc: 0.8350 - val_loss: 0.5499 - val_acc: 0.8168
Epoch 36/50
390/390 [==============================] - 26s 67ms/step - loss: 0.4617 - acc: 0.8395 - val_loss: 0.6043 - val_acc: 0.8019
Epoch 37/50
390/390 [==============================] - 26s 67ms/step - loss: 0.4655 - acc: 0.8401 - val_loss: 0.5746 - val_acc: 0.8124
Epoch 38/50
390/390 [==============================] - 26s 67ms/step - loss: 0.4576 - acc: 0.8419 - val_loss: 0.5783 - val_acc: 0.8083
Epoch 39/50
390/390 [==============================] - 26s 68ms/step - loss: 0.4554 - acc: 0.8416 - val_loss: 0.5443 - val_acc: 0.8202
Epoch 40/50
390/390 [==============================] - 26s 67ms/step - loss: 0.4418 - acc: 0.8470 - val_loss: 0.5430 - val_acc: 0.8232
Epoch 41/50
390/390 [==============================] - 26s 68ms/step - loss: 0.4489 - acc: 0.8444 - val_loss: 0.5490 - val_acc: 0.8209
Epoch 42/50
390/390 [==============================] - 26s 67ms/step - loss: 0.4476 - acc: 0.8459 - val_loss: 0.5412 - val_acc: 0.8232
Epoch 43/50
390/390 [==============================] - 26s 67ms/step - loss: 0.4389 - acc: 0.8489 - val_loss: 0.5333 - val_acc: 0.8291
Epoch 44/50
390/390 [==============================] - 26s 67ms/step - loss: 0.4362 - acc: 0.8486 - val_loss: 0.5257 - val_acc: 0.8332
Epoch 45/50
390/390 [==============================] - 26s 67ms/step - loss: 0.4360 - acc: 0.8491 - val_loss: 0.5633 - val_acc: 0.8218
Epoch 46/50
390/390 [==============================] - 26s 67ms/step - loss: 0.4356 - acc: 0.8492 - val_loss: 0.5376 - val_acc: 0.8304
Epoch 47/50
390/390 [==============================] - 26s 67ms/step - loss: 0.4188 - acc: 0.8549 - val_loss: 0.5234 - val_acc: 0.8311
Epoch 48/50
390/390 [==============================] - 26s 68ms/step - loss: 0.4233 - acc: 0.8536 - val_loss: 0.5405 - val_acc: 0.8243
Epoch 49/50
390/390 [==============================] - 26s 67ms/step - loss: 0.4224 - acc: 0.8541 - val_loss: 0.5795 - val_acc: 0.8173
Epoch 50/50
390/390 [==============================] - 26s 68ms/step - loss: 0.4102 - acc: 0.8573 - val_loss: 0.5396 - val_acc: 0.8279
Model took 1305.76 seconds to train
## Model performed best in 44th Epoch , val_accuracy = 83.32 
