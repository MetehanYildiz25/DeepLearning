import tensorflow
from tensorflow.keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 
from tensorflow.keras.utils import to_categorical  # encoding işlemi için
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D  # özellik çıkarımı için
from tensorflow.keras.layers import Flatten, Dense, Dropout  # sınıflandırma için
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator  # veri artırımı için
from sklearn.metrics import classification_report
import warnings

warnings.filterwarnings("ignore")



# MNIST veri setini yükleme
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Sınıf etiketleri
class_labels = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
 
# Görüntüleri gösterme (DÜZELTİLDİ)
fig, axes = plt.subplots(1, 6, figsize=(20, 24))  
for i in range(6):
    axes[i].imshow(x_train[i], cmap="gray")  # Görüntüleri gösterme
    label = class_labels[y_train[i]]  # Etiketleri belirleme
    axes[i].set_title(label)  # Başlık ekleme
    axes[i].axis("off")  # Eksenleri kapatma

plt.show()

# **Veri Normalizasyonu**
x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255

# **Görüntüleri uygun formata getirme**
x_train = np.expand_dims(x_train, axis=-1)  # (28,28) -> (28,28,1)
x_test = np.expand_dims(x_test, axis=-1)    # (28,28) -> (28,28,1)

# **Etiketleri one-hot encoding formatına çevirme**
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)


datagen= ImageDataGenerator(
            rotation_range=20,  # 20 dereceye kadar döndürme sağlar 
            width_shift_range=0.2, # görüntüyü yatayda %20 döndür 
            height_shift_range=0.2, # görüntüyü dikeyde %20 döndürür
            shear_range=0.2, # görüntü üzerinde kaydırma
            zoom_range=0.2,  # görüntü üzerinde zoom yapar
            horizontal_flip= True, # görüntüyü yatayda ters çevirir
            fill_mode="nearest" # boş alanları doldurmak için en yakın pikselleri kullanır 
            
            
    )

datagen.fit(x_train)


#model oluşturma

model=Sequential()
#özellik çıkarma  conv-> relu -> conv-> relu -> pooling -> dropout 
model.add(Conv2D(32 ,(3,3), padding= "same", activation="relu", input_shape=x_train.shape[1:]))
model.add(Conv2D(32,(3,3), activation="relu"))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

#özellik çıkarma  conv-> relu -> conv-> relu -> pooling -> dropout 

model.add(Conv2D(64 ,(3,3), padding= "same", activation="relu"))
model.add(Conv2D(64,(3,3), activation="relu"))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.3))

#sınıflandırma 

model.add(Flatten())
model.add(Dense(512,activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(10, activation="softmax"))

model.summary()

model.compile(optimizer=RMSprop(learning_rate=1e-3),
              loss="categorical_crossentropy",
              metrics= ["accuracy"])

history=model.fit(datagen.flow(x_train,y_train,batch_size=512),
                  epochs=20,
                  validation_data=(x_test,y_test)
                  
    )



#model testi ve değerlendirmesi

y_pred=model.predict(x_test)
y_pred_class=np.argmax(y_pred,axis=1)
y_true=np.argmax(y_test,axis=1)

report=classification_report(y_true, y_pred_class, target_names=class_labels)


plt.figure()
#LOSS
plt.subplot(1,2,1)
plt.plot(history.history["loss"], label="Train loss")
plt.plot(history.history["val_loss"],label=" Validation loss")
plt.xlabel("Epochs")
plt.ylabel("loss")
plt.legend()
plt.grid(True)

plt.subplot(1,2,2)
plt.plot(history.history["accuracy"],label="Train accuracy")
plt.plot(history.history["val_accuracy"],label=" Validation accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)


plt.show()



















