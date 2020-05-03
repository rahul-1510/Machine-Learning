import numpy as np
import matplotlib.pyplot as plt
from numpy import *

#Reading Input
x = plt.imread('test1.jpeg')
y = x.reshape(x.shape[2],x.shape[0]*x.shape[1])

#For manual Input

# D = int(input("Enter the number of rows:")) 
# N = int(input("Enter the number of columns:")) 
# # #Initialize matrix 
# y = [] 
# print("Enter the entries rowwise:")   
# #For user input 
# for i in range(0,D):
#    y.append([])
#    for j in range(0,N):
#        y[i].append(int(input()))  


y=np.array(y,dtype='f')
# Subtracting Zero Mean
for i in range(y.shape[0]):
    (y[i,:])=((y[i,:]-(np.mean(y[i,:],axis=0))  ))                                

print(y)
#print(np.cov(y))
print('\n')
cxx=((np.dot(y,y.T))/(y.shape[1]))

#Printing Cxx
print(cxx)

eig_vals, eig_vecs = np.linalg.eig(cxx)
eig_vals=np.real(eig_vals)
eig_vecs=np.real(eig_vecs)

eig = np.diag(eig_vals)
print('\n')

#printing Eigen Value Matrix
print(eig)
#This should equal to Cyy

print('\n')
ym = np.dot(eig_vecs.T,y)
cyy=((np.dot(ym,ym.T))/(y.shape[1]))

#printing Cyy
print(cyy)

print('\n')

#Printing EDE.T 
print(np.dot(np.dot(eig_vecs,eig),eig_vecs.T))
#This should Equal to Cxx
