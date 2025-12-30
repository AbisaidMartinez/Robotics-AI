# Graficación del metodo usando un solo cuadrante
"""
Created on Mon Dec 29 21:54:16 2025

@author: qbo28
"""

import matplotlib.pyplot as plt
import numpy as np 

r0 = 123
a = 7**5
c = 0
m = 2**31

r=r0
x=[r0/m]
iteraciones=range(7002)
x1=[]
y1=[]
x2=[]
y2=[]
i=0
pi=1
while (i<(10**5)) and (3.1415>pi or pi>3.1416):

    r1=(a*r0+c)%m

    x.append(r1/m)
    r0=r1
    if (x[i]**2+x[i+1]**2)<1:
        x1.append(x[i])
        y1.append(x[i+1])
    else:
        x2.append(x[i])
        y2.append(x[i+1])
    
    area=(len(x1)+1)/(i+1)
    pi=((area*4))
    i=i+1

theta=np.linspace(0,np.pi/2,100)
c1=np.cos(theta)
c2=np.sin(theta)

#plt.style.use('classic')
plt.figure(figsize=(10,12))
plt.xlim(0,1)
plt.ylim(0,1)
plt.plot(c1,c2)
plt.scatter(x1,y1,color='red',lw=0.1)
plt.scatter(x2,y2,color='green',lw=0.1)
plt.xlabel('x')
plt.ylabel('y')
plt.grid(ls='--', alpha=0.50)
plt.annotate(f"Área círculo={round(area,8)}",xy=(0.05,0.25)
             ,xycoords='data',horizontalalignment='left',verticalalignment='bottom',fontsize=20)
plt.annotate(r"$\pi$"+f"={round(pi,8)}",xy=(0.05,0.15)
             ,xycoords='data',horizontalalignment='left',verticalalignment='bottom',fontsize=20)
plt.annotate(f"Puntos={i}",xy=(0.05,0.05)
             ,xycoords='data',horizontalalignment='left',verticalalignment='bottom',fontsize=20)
plt.title(r'Monte Carlo, área y $\pi$', fontsize=28)

#%% Graficación del metodo usando los cuatro cuadrantes

import matplotlib.pyplot as plt
import numpy as np
import random as rnd

t=np.linspace(0,2*np.pi,100)
x=np.cos(t)
y=np.sin(t)

Out=0
In=0
i=0
pi=0
x_in=[]
y_in=[]
x_out=[]
y_out=[]
while (0.78539>pi or pi>0.7854):
    X=rnd.uniform(-1,1)
    Y=rnd.uniform(-1,1)
    r=np.sqrt(X**2+Y**2)
    if(r>1):
       Out+=1
       x_out.append(X)
       y_out.append(Y)
    else:
        In+=1
        x_in.append(X)
        y_in.append(Y)
    i+=1
    if(In>1):
        pi= 4* (In/i)
#plt.style.use('dark_background')
plt.figure(figsize=(10,10))
#plt.axis('equal')
plt.scatter(x_out,y_out,c='deepskyblue')
plt.scatter(x_in,y_in,c='darkorchid')
plt.plot(x,y,c="slategrey",lw=4)
plt.legend(["Values out","Values In","Radius"])

plt.xlim(-1,1)
plt.ylim(-1,1)
plt.annotate(f"Área círculo={round(pi,8)}",xy=(0.05,0.25)
             ,xycoords='data',horizontalalignment='left',verticalalignment='bottom',fontsize=20, color='black')
plt.annotate(r"$\pi$"+f"={round(pi,8)}",xy=(0.05,0.15)
             ,xycoords='data',horizontalalignment='left',verticalalignment='bottom',fontsize=20, color='black')
plt.annotate(f"No. Puntos={i}",xy=(0.05,0.05)
             ,xycoords='data',horizontalalignment='left',verticalalignment='bottom',fontsize=20, color='black')
plt.title("Monte Carlo, área y $\pi$", fontsize=28)

#%% Error de convergencia

import numpy as np
import random as rnd
import matplotlib.pyplot as plt

N_values = np.logspace(2, 5, 20, dtype=int)  # 20 valores entre 1e2 y 1e5
K=200
mse_error = []

for N in N_values:
    errors = []
    for k in range(K):
        In = 0
        for j in range(N):
            X = rnd.uniform(-1, 1)
            Y = rnd.uniform(-1, 1)
            r = np.sqrt(X**2 + Y**2)
            if r <= 1:
                In += 1
        pi_hat = 4 * (In / N)
        mse_value = (pi_hat - np.pi)**2
        errors.append(mse_value)
    mse_error.append(np.mean(errors))

# Graficar MSE vs N
plt.figure()
plt.loglog(N_values, mse_error, marker='o')
plt.xlabel("Número de puntos N")
plt.ylabel("Error cuadrático (MSE)")
plt.title("Decrecimiento del error de Monte Carlo para π")
plt.grid(True, which="both", ls="--")
plt.show()




