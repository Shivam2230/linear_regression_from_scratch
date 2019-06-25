import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset=pd.read_csv('/Weather.csv',low_memory=False)

dataset=dataset.head(1000)
m=0
c=0
L=0.0001
epochs=1000



x=dataset['MinTemp'].values.reshape(-1,1)
n=float(len(x))
print(x.shape)
y=dataset['MaxTemp'].values.reshape(-1,1)

plt.scatter(x, y)

for i in range(epochs):
    y_pred=m*x+c
    #print(y_pred)
    D_m=-(2/n)*sum(x*(y-y_pred))
    D_c=-(2/n)*sum(y-y_pred)
    m=m-L*D_m
    c=c-L*D_c

print(m,c)
y_pred = m*x + c


plt.plot([min(x), max(x)], [min(y_pred), max(y_pred)], color='red')  
plt.xlabel('MinTemp')
plt.ylabel('MaxTemp')
plt.show()

