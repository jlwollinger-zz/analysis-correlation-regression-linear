import numpy as np
import matplotlib.pyplot as plt
import math 
import scipy.io as scipy
from mpl_toolkits.mplot3d import Axes3D  
import random

# A
# Carrega a base de dados
mat = scipy.loadmat('data/data_preg.mat')

data = mat['data']

x = data[:, 0]
y = data[:, 1]

# B
# plt.scatter(x, y)

# Pega uma percentual de dados variados de um array passado
def getPieceOfData(data, percentage):
    k = len(data) * percentage // 100
    randomData = random.sample(list(data), k)
    return np.array(randomData)

# X e Y dos dados de treinamento
trainingDataX = getPieceOfData(x, 90)
trainingDataY = getPieceOfData(y, 90)

# X e Y dos dados de teste
testDataX = getPieceOfData(x, 10)
testDataY = getPieceOfData(y, 10)

# Calcula B, plota o gráfico e retorna o EQM
def polyfit(x, y, n, color):
    b = np.polyfit(x, y, n)
    xp = np.linspace(x.min(), x.max(), len(x))
    p = polynomial(b, xp, n)

    # Other way using a function from numpy
    # p = np.poly1d(b)
    # plt.plot(x, y, '.', xp, p(xp), '-')
    # eqm = np.square(y - p(xp)).mean()
    
    # calcular eqm
    eqm = np.square(y - p).mean()

    # plotar gráfico
    plt.plot(x, y, '.', xp, p, '-', color=color)
    return eqm

# Calcula polinômio
def polynomial(b, xp, n):
    p = []
    for i in range(len(xp)):
        x = xp[i]
        if (n == 1):
            r = b[0]*x + b[1]
        if (n == 2):
            r = b[0]*x**2 + b[1]*x + b[2]
        if (n == 3):
            r = b[0]*x**3 + b[1]*x**2 + b[2]*x + b[3]
        if (n == 8):
            r = b[0]*x**8 + b[1]*x**7 + b[2]*x**6 + b[3]*x**5 + b[4]*x**4 + b[5]*x**3 + b[6]*x**2 + b[7]*x + b[8]

        p.append(r)

    return p

# C - G
# eqm = polyfit(x, y, 1, 'r')
# print('EQM 1: ', eqm)
# eqm = polyfit(x, y, 2, 'g')
# print('EQM 2: ', eqm)
# eqm = polyfit(x, y, 3, 'k')
# print('EQM 3: ', eqm)
# eqm = polyfit(x, y, 8, 'y')
# print('EQM 8: ', eqm)

# I - J
# Executa o treinamento e teste dos dados
def trainAndTestData(n, color):
    b = np.polyfit(trainingDataX, trainingDataY, n)
    eqm = calcEqm(testDataX, testDataY, b, n, color)
    print(n, ' - ', eqm)

# Calcula o EQM
def calcEqm(x, y, b, n, color):    
    xp = np.linspace(x.min(), x.max(), len(x))
    p = polynomial(b, xp, n)
    eqm = np.square(y - p).mean()
    plt.plot(x, y, '.', xp, p, '-', color=color)
    return eqm

trainAndTestData(1, 'r')
trainAndTestData(2, 'g')
trainAndTestData(3, 'k')
trainAndTestData(8, 'y')

plt.show()

# K Qual método (n) é o melhor?
# Analisando o EQM em 5 rodadas de teste, o N = 8 é o melhor método na maioria das vezes

# Testes
# 1  -  0.4961227170478343
# 2  -  0.48464175646275437
# 3  -  0.5031436334750868
# 8  -  0.41338619477821004

# 1  -  1.0057956788024187
# 2  -  0.9865199520842296
# 3  -  0.996051731615205
# 8  -  1.228932639815032

# 1  -  0.8513473997312552
# 2  -  0.9171996987011118
# 3  -  0.8710100282611162
# 8  -  0.625558636062741

# 1  -  0.4604172734692457
# 2  -  0.41440994207827514
# 3  -  0.47710305174915346
# 8  -  0.4438981556974905

# 1  -  0.7140911218625761
# 2  -  0.7711042633338792
# 3  -  0.7624954479446768
# 8  -  0.5528278016116175