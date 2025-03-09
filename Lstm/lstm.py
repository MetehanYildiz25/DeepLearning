import matplotlib.pyplot as plt 
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM,Embedding, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam


max_features=10000# en çok kullanılan 10000 kelimeyi almak istiyoruz 

(X_train,y_train),(X_test,y_test)=imdb.load_data(num_words = max_features)

maxlen=100# her yorumun uzunluğu 100 kelime olsun 
X_train=pad_sequences(X_train,maxlen=maxlen)

X_test=pad_sequences(X_test,maxlen=maxlen)


def build_lstm_model():
    model=Sequential()
    model.add(Embedding(input_dim=max_features,output_dim=64,input_length=maxlen))
    model.add(LSTM(units=8))
    model.add(Dropout(0.6))
    model.add(Dense(1,activation="sigmoid"))# 2 sınıf olduğu için sigmoid kullanılır7

    model.compile(optimizer= Adam(learning_rate=0.0001),
                  loss="binary_crossentropy",
                  metrics=["accuracy"])
    return model

model=build_lstm_model() 
model.summary()   
    
early_stopping=EarlyStopping(monitor="val_accuracy",patience=3,restore_best_weights=True)

history=model.fit(X_train,y_train,epochs=10,batch_size=16,validation_split=0.2,callbacks=[early_stopping])



loss,accuracy=model.evaluate(X_test,y_test)
print(f" test loss{loss} test accuracy {accuracy}")


plt.figure(figsize=(14,5))

plt.subplot(1,2,1)
plt.plot(history.history["loss"],label="training loss")
plt.plot(history.history["val_loss"],label="validation loss")
plt.title("LOSS")
plt.xlabel("Epochs")
plt.ylabel("loss")
plt.legend()
plt.grid(True)

plt.subplot(1,2,2)
plt.plot(history.history["accuracy"],label="training accuracy")
plt.plot(history.history["val_accuracy"],label="validation accuracy")
plt.title("ACCURACY")
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.legend()
plt.grid(True)


plt.show()
































    
    