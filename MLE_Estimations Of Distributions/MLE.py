import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import factorial
import scipy.stats

#Maximum Likelihood estimation for Binomial Distribution
def bino(n,p):
	#Function will print p
	
	b1 = np.random.binomial(n,p,1000)
	bins = np.linspace(-30,30,100)
	#plt.hist(b1, bins,label='Before Perfoming MLE')
	if(n==0):
		p = sum(np.random.binomial(n,p,1000)==0)/(1000)
		print(p)
	else:
		p = sum(np.random.binomial(n,p,1000))/(1000*n)
		print(p)
	b2 = np.random.binomial(n,p,1000)
	#plt.hist(b2, bins,label='After Perfoming MLE')
	#plt.legend(loc='best')
	#plt.savefig('binomial.pdf')
	print('\n')

#Maximum Likelihood estimation for Poisson's Distribution
def poisson(lamb):
	#Function will print lamda
	p1 = np.random.poisson(lamb,10000)
	bins = np.linspace(-30,30,100)
	#plt.hist(p1, bins,label='Before Perfoming MLE')
	lamb = sum(p1)/(10000)
	print(lamb)
	p2 = np.random.poisson(lamb,10000)
	#plt.hist(p2, bins,label='After Perfoming MLE')
	#plt.legend(loc='best')
	#plt.savefig('poisson.pdf')
	print('\n')

#Maximum Likelihood estimation for Exponential Distribution
def expon(lm):
	#Function will print lambda
	e1 = np.random.exponential((1/lm),10000)
	bins = np.linspace(-30,30,100)
	#plt.hist(e1,bins,label='Before Perfoming MLE')

	lm = (10000/sum(e1))
	print(lm)

	e2 = np.random.exponential((1/lm),10000)
	#plt.hist(e2,bins,label='After Perfoming MLE')
	#plt.legend(loc='best')
	#plt.savefig('Exponential.pdf')
	print('\n')

#Maximum Likelihood estimation for Gaussian Distribution
def gauss(u,sig):
	#Function will print u and sigma
	g1 = np.random.normal(u,sig,100000)
	bins = np.linspace(-30,30,100)
	#plt.hist(g1,bins,label='Before Perfoming MLE')

	u = sum(g1)/100000
	print(u)
	print('\n')
	sig = np.sqrt(sum(np.square(g1-u))/100000)
	print(sig)

	g2 = np.random.normal(u,sig,100000)
	#plt.hist(g2,bins,label='After Perfoming MLE')
	#plt.legend(loc='best')
	#plt.savefig('Gaussian.pdf')
	print('\n')
	
#Maximum Likelihood estimation for Laplacian Distribution

def sort(arr):
    for i in range(len(arr)):
        swap = i + np.argmin(arr[i:])
        (arr[i], arr[swap]) = (arr[swap], arr[i])
    return arr

def lapl(nu,l,N):
	#Function will print u and lambda
	lp1 = np.random.laplace(nu,l,N)
	bins = np.linspace(-30,30,100)
	#plt.hist(lp1,bins,label='Before Perfoming MLE')
	arr = sort(lp1)
	index = (N-1)//2
	if(N%2==0):
		nu = arr[index] 
		print(nu);
		print('\n')
		print(sum(abs(lp1-nu))/N)
	else:
		nu = (arr[index] + arr[index+1])/2.0 
		print(nu)
		l = (sum(abs(lp1-nu))/N)
		print(l)
		print('\n')

	lp2 = np.random.laplace(nu,l,N)	
	#plt.hist(lp2,bins,label='After Perfoming MLE')
	#plt.legend(loc='best')
	#plt.savefig('Laplacian.pdf')
	print('\n')

bino(5,0.9)
poisson(5)
expon(7)
gauss(3,7)
lapl(-5,4,1000) #For n = even case
#lapl(-5,4,999) #For N = odd case  
plt.show()
