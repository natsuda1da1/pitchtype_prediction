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
from keras.models import load_model

#modelへ保存データを読み込み
model = load_model('model.h5')

import serial
import csv
import sys

model.summary()

ser = serial.Serial('/dev/tty.Nao',timeout=None)
idx=0
test_li = []

while len(test_li) <= 650:
    idx = idx+1
    line = ser.readline()
    line = list(map(lambda x:float(x),line.split()))
    
    del line[-1]
    line = np.array(line)
                
    predict = model.predict_classes(line.reshape(1, -1), batch_size=1, verbose=0)
    test_li.extend(predict)
test_li = np.array(test_li)
# sample = [315.78, -67.23, -350.12, 0.93, 0.81, 0.43, 25.4, 100.77, 9.1]
 # print("\n")
 # print("--サンプル球種のデータ--")
 #
 # print(sample)
 
# # sample = np.array(sample)
 # predict = model.predict_classes(sample.reshape(1,-1),batch_size=1,verbose=0)
print("\n")
print("--予測値--")
mode = statistics.mode(test_li)
if mode == 1:
	print('ストレート')
elif mode == 2:
    print('カーブ')
else:
    print('シュート')
print("\n")