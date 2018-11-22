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
def regresseao_multipla(x, y):
    # b = (((datx' * datx) ^ -1) * datx') * daty;

    beta = (((x.transpose().dot(x)).dot(-1)).dot(x.transpose())).dot(y)

    return x.dot(beta)
    #x = ((xTransposed * x) * -1) * (xTransposed * y)

    #𝑦̂ = X*𝛽

    #𝛽= (Xt X)-1 Xty

#scatter3d(x[:,0], y, x[:,1])

#Correlação tamanho casa e preço
correlacaoTamanhoPreco = correlation(x[:,0], y)

#Correlação número de quartos e preço
correlacaoQuartosPreco = correlation(x[:,1], y)

#devem ser Correlacao TamXPreco: 0.85499, Correlacao nQuartoXTamanho 0.44226

x = adicionarColuna(x, 0, 1)
print(regresseao_multipla(x, y))

#Plota gráfico 3d com coluna tamanho, preço e quartos.
fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(x[:,1], y, x[:,2])
plt.plot(y, regresseao_multipla(x, y), '-')
plt.title('Correlacao TamXPreco: {:.5f}, Correlacao nQuartoXTamanho {:.5f}'.format(correlacaoTamanhoPreco, correlacaoQuartosPreco))
plt.show()


#calcular regressão múltipla: LINEAR MULTIPLA ((Xtransp x X)-1) Xtransp x y
#calcular y, 𝑦̂ = X*𝛽