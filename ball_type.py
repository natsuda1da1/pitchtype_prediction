#モジュールの読み込み
from __future__ import print_function

import csv
import statistics

import pandas as pd
from pandas import Series,DataFrame

from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import numpy as np
import matplotlib.pyplot as plt

import keras
from keras.datasets import fashion_mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop
from keras.optimizers import Adam


#CSVファイルの読み込み
ball_type_set = pd.read_csv("/Users/natsumi/documents/naist/type_all.csv",sep=",",header=0)
print(ball_type_set)

#説明変数
x = DataFrame(ball_type_set.drop("type",axis=1))

#ストレート(1), カーブ(2), シュート(3)での分類
y = DataFrame(ball_type_set["type"])

#説明変数・目的変数をそれぞれ訓練データ・テストデータに分割
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.05)


#データの整形
x_train = x_train.astype(np.float)
x_test = x_test.astype(np.float)

y_train = keras.utils.to_categorical(y_train,10)
y_test = keras.utils.to_categorical(y_test,10)


#ニューラルネットワークの実装①
model = Sequential()

model.add(Dense(50, activation='relu', input_shape=(9,)))
model.add(Dropout(0.2))

model.add(Dense(50, activation='relu', input_shape=(9,)))
model.add(Dropout(0.2))

model.add(Dense(50, activation='relu', input_shape=(9,)))
model.add(Dropout(0.2))

model.add(Dense(10, activation='softmax'))

model.summary()
print("\n")

#ニューラルネットワークの実装②
model.compile(loss='mean_squared_error',optimizer=RMSprop(),metrics=['accuracy'])
#勾配法には、Adam(lr=1e-3)という方法もある（らしい）。

#ニューラルネットワークの学習
history = model.fit(x_train, y_train,batch_size=250,epochs=5000,verbose=1,validation_data=(x_test, y_test))
model.save('model.h5')
#ニューラルネットワークの推論
score = model.evaluate(x_test,y_test,verbose=1)
print("\n")
print("Test loss:",score[0])
print("Test accuracy:",score[1])


#学習履歴のグラフ化に関する参考資料
#http://aidiary.hatenablog.com/entry/20161109/1478696865

def plot_history(history):
    #print(history.history.keys())

    # 精度の履歴をプロット
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend(['accuracy', 'val_accuracy'], loc='lower right')
    plt.show()

    # 損失の履歴をプロット
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(['loss', 'val_loss'], loc='lower right')
    plt.show()

# 学習履歴をプロット
plot_history(history)
