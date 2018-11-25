#José Wollinger e Michel Tank
import numpy as np
import matplotlib.pyplot as plt
import math 

x1 = [10,8,13,9,11,14,6,4,12,7,5]
y1 = [8.04,6.95,7.58,8.81,8.33,9.96,7.24,4.26,10.84,4.82,5.68]

x2 = [10,8,13,9,11,14,6,4,12,7,5]
y2 = [9.14,8.14,8.47,8.77,9.26,8.10,6.13,3.10,9.13,7.26,4.74]

x3 = [8,8,8,8,8,8,8,8,8,8,19]
y3 = [6.58,5.76,7.71,8.84,8.47,7.04,5.25,5.56,7.91,6.89,12.50]


def b1(x, y):
    total = 0
    divi = 0
    for i in range(len(x)):
        total = total + (x[i] - np.mean(x)) * (y[i] -  np.mean(y))
        divi = divi + np.power(x[i] - np.mean(x), 2)
    return total / divi

def b0(x , y):
    return np.mean(y) - b1(x,  y) * np.mean(x)

def correlation(xList, yList):
  meanX = np.mean(xList)
  meanY = np.mean(yList)
  totalUpCalc = 0
  totalDownCalc1 = 0
  totalDownCalc2 = 0 

  for idx in range(len(xList)):
    totalUpCalc = totalUpCalc + ((xList[idx] - meanX) * (yList[idx] - meanY))
    totalDownCalc1 = totalDownCalc1 + ((xList[idx] - meanX) ** 2)
    totalDownCalc2 = totalDownCalc2 + ((yList[idx] - meanY) ** 2)
  
  return totalUpCalc / math.sqrt(totalDownCalc1 * totalDownCalc2)

def regression(x, y):
  regressao = []
  for value in x:
    regressao.append(b0(x, y) + b1(x, y) * value)
  return regressao


plt.title('Correlation {:.5f}, b0 {:.5f}, b1 {:.5f}'.format(correlation(x1, y1), b0(x1, y1), b1(x1, y1)))
plt.scatter(x1, y1)
plt.plot(x1, regression(x1, y1), '-')
plt.show()

plt.title('Correlation {:.5f}, b0 {:.5f}, b1 {:.5f}'.format(correlation(x2, y2), b0(x2, y2), b1(x2, y2)))
plt.scatter(x2, y2)
plt.plot(x2, regression(x2, y2), '-')
plt.show()

plt.title('Correlation {:.5f}, b0 {:.5f}, b1 {:.5f}'.format(correlation(x3, y3), b0(x3, y3), b1(x3, y3)))
plt.scatter(x3, y3)
plt.plot(x3, regression(x3, y3), '-')
plt.show()

#3 - Qual dos datasets não é apropriado para regressão linear?

# O segundo pois a natureza da relação entre os dados não é linear, 
# para este caso uma regresão polinomial seria mais conveniente.
# O terceiro dataset também não é apropriado pois os dados são muito dispersos.


