import numpy as np
import matplotlib.pyplot as plt
import math 
#pip install scipy
import scipy.io as scipy
from mpl_toolkits.mplot3d import Axes3D

# a
mat = scipy.loadmat('data/data_preg.mat')

data = mat['data']

x = data[:, 0]
y = data[:, 1]

# b
plt.scatter(x, y)
# plt.waitforbuttonpress()

def polyfit(n, color):
    b = np.polyfit(x, y, n)
    # prof: y2 = b2[2] + (b2[1] * x) + (b2[0] * x.^2)
    # meu 1: value = (b2[1] * x[i]) + (b2[0] * x[i] ** 2)
    totalResidue = 0
    y2 = []
    b0 = b[n]
    for i in range(len(x)):
        value = b0
        for k in range(n-1, -1, -1):
            value = value + (b[k] * x[i])
        value = value + (b[0] * x[i] ** 2)
        y2.append(value)
        residue = (x[i] - value) ** 2
        totalResidue = totalResidue + residue
    plt.plot(x, y2, color)    
    return totalResidue / len(x)

eqm = polyfit(1, 'r')
print(eqm)
eqm = polyfit(2, 'g')
print(eqm)
eqm = polyfit(3, 'k')
print(eqm)
eqm = polyfit(8, 'y')
print(eqm)

plt.show()