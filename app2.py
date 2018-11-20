import numpy as np
import matplotlib.pyplot as plt
import math 
#pip install scipy
import scipy.io as scipy
from mpl_toolkits.mplot3d import Axes3D

mat = scipy.loadmat('data/data.mat')
#Matriz 3x3. colunas:
#0 = tamanho, 1 = número de quartos, 2 = preço
data = mat['data']

#Tamanho e quartos
x = data[:, 0:3:2] #slice das colunas

y = data[:,1]


def calc_correlacao_e_regressao(x, y):
    corre = correlation(x, y)
    regre = regression(x, y)
    return corre, regre

def regression(x, y):
  regressao = []
  for value in x:
    regressao.append(b0(x, y) + b1(x, y) * value)
  return regressao

def scatter3d(x, y, z):
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(x, y, z)
    #plt.plot(y, regression(x, z), '-')
    plt.show()

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


def adicionarColuna(x, posicao, valor):
    return np.insert(x, posicao, values=valor, axis=1) #Adiciona uma coluna com 1s no índice 0

#TODO Implementar regressão múltipla
def regresseao_multipla(xTransposed, x, y):
    print('TODO')
    #x = ((xTransposed * x) * -1) * (xTransposed * y)

    #𝑦̂ = X*𝛽

    #𝛽= (Xt X)-1 Xty

#Plota gráfico 3d com coluna tamanho, preço e quartos.
#scatter3d(x[:,0], y, x[:,1])

#Correlação tamanho casa e preço
correlacao, regressao = calc_correlacao_e_regressao(x[:,0], y)
#print(correlacao)
#print(regressao)
#Correlação número de quartos e preço
correlacao, regressao = calc_correlacao_e_regressao(x[:,1], y)
#print(correlacao)
#print(regressao)
#x = np.insert(x, 0, values=1, axis=1) #Adiciona uma coluna com 1s no índice 0

x = adicionarColuna(x, 0, 1)
xTransposed = x.transpose()
print(x)
#regresseao_multipla(xTransposed, x, y)

#adicionar na primeira coluna o valor 1
#transpor a mastrix
#calcular regressão múltipla: LINEAR MULTIPLA ((Xtransp x X)-1) Xtransp x y
#calcular y, 𝑦̂ = X*𝛽