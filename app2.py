#José Wollinger e Michel Tank
import numpy as np
import matplotlib.pyplot as plt
import math 
#pip install scipy
import scipy.io as scipy
from mpl_toolkits.mplot3d import Axes3D
from numpy.linalg import inv

mat = scipy.loadmat('data/data.mat')
#Matriz 3x3. colunas:
#0 = tamanho, 1 = número de quartos, 2 = preço
data = mat['data']

#Tamanho e quartos
x = data[:, 0:2] #slice das colunas
x = np.insert(x, 0, values=1, axis=1) #Adiciona 1's na primeira coluna

#Preços
y = data[:,2]

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
    return  #Adiciona uma coluna com 1s no índice 0

def regresseao_multipla(x, y):
    beta = (inv(x.transpose().dot(x))).dot(x.transpose()).dot(y) #𝛽= (Xt X)-1 Xty
    #dot multiplica matriz
    #inv inverte a matriz
    return beta, x.dot(beta) #𝑦̂ = X*𝛽
    
#Correlação tamanho casa e preço
correlacaoTamanhoPreco = correlation(x[:,1], y)

#Correlação número de quartos e preço
correlacaoQuartosPreco = correlation(x[:,2], y)

#Calcula regressão múltipla e beta
beta, regressao  = regresseao_multipla(x, y)

#Plota gráfico 3d com coluna tamanho, preço e quartos.
fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(x[:,1], x[:,2], y)
plt.plot(x[:,1], x[:,2], regressao, '-')
plt.title('Correlacao TamXPreco: {:.5f}, Correlacao nQuartoXTamanho {:.5f}'.format(correlacaoTamanhoPreco, correlacaoQuartosPreco))
plt.show()


#Cria o array e calcula o preço da casa
z = np.array([1, 1650, 3])
resultado = z.dot(beta)
print('Preço de uma casa com tamanho de 1650 e 3 quartos {:.5f}'.format(resultado))