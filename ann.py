import tensorflow
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.models import load_model
import tensorflow.keras

import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

plt.figure(figsize=(10, 10))
for i in range(8):
    plt.subplot(2,4,i+1)
    plt.imshow(x_train[i])
    plt.title(f"index :{i}, label :{y_train[i][0]} ")
    plt.axis("off")
plt.show()

#normalizasyon
print(x_train.shape)
print(x_test.shape)
x_train=x_train.astype("float32")/255
x_test=x_test.astype("float32")/255

y_train=to_categorical(y_train,10)#10 sınıf olduğu için
y_test=to_categorical(y_test,10)

model=Sequential()

model.add(Flatten(input_shape=(32,32,3)))# 3 boyutlu olan veriyi 1 boyuta çeviriyoruz

model.add(Dense(512,activation="relu"))
model.add(Dense(256,activation="relu"))
model.add(Dropout(0.2))
model.add(Dense(10,activation="softmax"))
model.summary()

model.compile(optimizer="adam",loss="categorical_crossentropy",metrics=["accuracy"])

#early stopping= erken durdurma
#monitor= doğrulama setindeki val_los değerlerini izler
#patience= eğer 5 epoch boyunca val_loss değeri düşmezse eğitimi durdur
#restore_best_weights= en iyi ağırlıkları geri yükler
early_stopping=EarlyStopping(monitor="val_loss",patience=5, restore_best_weights=True)

#model_checkpoint= en iyi modeli kaydeder
checkpoint= ModelCheckpoint("ann_best_model.keras",monitor="val_loss",save_best_only=True)


history=model.fit(x_train,y_train,
                  epochs=10,
                  batch_size=60,
                  validation_split=0.2,
                 callbacks=[early_stopping,checkpoint])



test_loss, test_accuracy=model.evaluate(x_test,y_test)
print(f"Test accuracy: {test_accuracy} test loss: {test_loss}")

plt.figure()
plt.plot(history.history["accuracy"],marker="o", label="Training accuracy")
plt.plot(history.history["val_accuracy"],marker="o", label="Validation accuracy")
plt.title("Training and Validation Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)

plt.show()


plt.figure()    
plt.plot(history.history["loss"],marker="o", label="Training loss")
plt.plot(history.history[val_loss],marker="o",label="Validation loss")
plt.title("Training an Validation Loss")

plt.xlabel("Epochs")
plt.ylabel("loss")
plt.legend()
plt.grid(True)

plt.show()  

model.save("ann_model.keras")

loaded_model=load_model("ann_model.keras")

test_loss, test_accuracy=loaded_model.evaluate(x_test,y_test)
print(f"Test accuracy: {test_accuracy} test loss: {test_loss}")

