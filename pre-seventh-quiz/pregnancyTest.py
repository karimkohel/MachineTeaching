# import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd

def unit_step_function(x):
    return 0 if x<0 else 1

df = pd.read_csv("data1.csv")

x = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

# print(x)
# print()
# print(y)

groups = df.groupby("Class")

for name, group in groups:
    plt.plot(group["Number of times pregnant"], group[" Age"], marker="x", linestyle="", label=name)
plt.xlabel('X1')
plt.ylabel('X2')
plt.title('Plotting the data')
plt.legend()
plt.show()