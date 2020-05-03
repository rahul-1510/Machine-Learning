import numpy as np
import random as random
import matplotlib.pyplot as plt
import math
#Function to calculate dot product of two vectors
def dotprod(A,B):
    product = 0 
    for i in range(0, y.shape[0]): 
        product = product + A[i] * B[i] 
    return product

#Function to calculate normal distribution for GMM
def norm(yn,muk,sigk):
	yn = yn - muk
	return ((1./(np.sqrt(((2*np.pi)**y.shape[0])*np.linalg.det(sigk)+0.00001))+1e-5)*np.exp(-(np.linalg.solve(sigk, yn).T.dot(yn)+0.00001)/2)) 

#Function to calculate Log likelihood 
def L(y,mu,sig,pi):
	return sum((np.log(sum((pi[j]*norm(y[:,i],mu[j,:],sig[j,:,:])) for j in range(k)))) for i in range (y.shape[1]))

#Expectation Maximization Algorithm
def EM(y,mu,sig,pi,max):
	cL = np.inf
	for _ in range(max):   
		pL = cL
		gamma = np.ones((y.shape[1],k),dtype='f')
		print(gamma)

		#E-Step
		
		for i in range(y.shape[1]):
			for j in range (k):
				gamma[i][j] = (pi[j]*norm(y[:,i],mu[j,:],sig[j,:,:])+1e-5)/(sum(pi[h]*norm(y[:,i],mu[h,:],sig[h,:,:])+0.00001 for h in range(j))+1e-5)

		print(gamma)
		# gamma = np.array(gamma,dtype='f')
		N = np.ones((k),dtype='f')
		for i in range(k):
			N[i] = (sum(gamma[j][i] for j in range(y.shape[1])))
		print(N)
		#M-Step

	# 	for i in range(k):
	# 		pi[i] = float(N[i]/y.shape[1])
	# 		mu[i] = sum(gamma[j][i]*y[:,j] for j in range(y.shape[1]))/N[i]
	# 		sig[i,:,:] = sum(gamma[j][i]*np.dot((y[:,j]-mu[i,:]),((y[:,j]-mu[i,:]).transpose()))  for j in range(y.shape[1]))/N[i]+0.00001*np.random.rand(y.shape[0],y.shape[0])

	# 	#Comparision of old and New Likelihood values 
	# 	#If it is less that epsilon return the parameters

	# 	cL = L(y,mu,sig,pi)
	# 	if abs(pL-cL)<eps:
	# 		break;

	# print(mu,sig,pi)


#Taking Input
x = plt.imread('test.jpeg')
y = x.reshape(x.shape[2],x.shape[0]*x.shape[1])
y = np.array(y,dtype='f') / 255
eps = 1e-5
k = int(input("Enter the Mixture Size k:")) 

#Initializing Parameters
mu = np.random.rand(k,y.shape[0])
sig=np.random.rand(k,y.shape[0],y.shape[0])
o=0;	
for i in range(k):
    for j in range(o,o-1+int(y.shape[1]/k)):
        sig[i,:,:] = np.cov(y[:,j])
    o=o+int(y.shape[1]/k)
for i in range (k):
	tem=sig[i,:,:]
	sig[i,:,:]=dotprod(tem,tem.transpose())+0.00001*np.random.rand(y.shape[0],y.shape[0])
pi=np.random.rand(k)
# print(pi)
# for i in range(k):
# 	pi[i]=(1./k)

# print(pi)
# print(pi.shape)
# print(mu)
# print(mu.shape)
# print(sig)
# print(sig.shape)

#Calling Function
EM(y,mu,sig,pi,1)
