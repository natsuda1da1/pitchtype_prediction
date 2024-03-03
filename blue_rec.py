import serial
import csv
import pandas as pd
import sys

ser = serial.Serial('/dev/tty.Nao',timeout=None)
idx=0
data = pd.DataFrame(columns=["gyroX","gyroY","gyroZ","accX","accY","accZ","pitch","roll","yaw","Temperature"])

try:
    while True:
        idx = idx+1
        line = ser.readline()
        if(line.decode()[:-1].startswith("Finish")):
            raise KeyboardInterrupt
        print(line)
        line = list(map(lambda x:float(x),line.split()))
        tmp_se=pd.Series(line,index=data.columns,name=idx)
        data = data.append(tmp_se,ignore_index = True)

except KeyboardInterrupt:
    print(data)
    data.to_csv('action.csv')
    ser.close()
    sys.exit(0)
