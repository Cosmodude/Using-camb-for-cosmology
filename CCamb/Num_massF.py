import sys, platform, os
import matplotlib
from matplotlib import pyplot as plt
import numpy as np
import csv
import pandas as pd
from astropy import units as u
from astropy import constants as c


csvf=pd.read_csv('Rockstar_for_CAMB.csv') 
mass=csvf["h.Mvir"] 
# mas=np.asarray(mass)
# mas10=np.log10(mas)
# mase=np.log(mas)
# print(mase[:3])
N= []
Ma=[]
j=0
step=10**12
M=10**12
while M <(10**15)+1 :
    Ma.append(M)
    N.append(0)
    N[j] = sum(map(lambda x : M+step>x>M, mass))
    j=j+1
    M=M+step
    print(j)
    print('M= ',M)
    if(j==9 or j==118):
        step=step*5
        print('step= ',step)
    if(j==18 or j==119):
        step=step*20
        print('step= ',step)

mas=np.asarray(Ma)
dN=[]
dmase=[]
dN_dmase=[]
for i in range(len(N)-1):
    dN.append(N[i+1]-N[i])
    dmase.append(np.log(mas)[i+1]-np.log(mas)[i])
    dN_dmase.append(dN[i]/dmase[i])
    
# dN=np.gradient(N)
# dmase=np.gradient(np.log(mas))

print('\n',dN)
print('\n',dmase)
logm=np.log10(mas)
# logm=np.delete(logm,-1)

WD = 'D:/SNU/Cosm'
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(np.delete(logm,-1) , np.abs(dN_dmase)/(4*(10**8)), c='k')
plt.title(f'Using MDPL2.Rockstar data')
ax.set_yscale('log')
ax.set_xlabel('$\log M (h^{-1}M_{\odot})$')
ax.set_ylabel('$dN/d\ln M$ [$(h^{-1}$Mpc$)^{-3}$]')
plt.tight_layout()
plt.savefig(WD+'/N3_Rockstar.png', dpi=300)
