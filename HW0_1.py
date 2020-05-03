import numpy as np
import random
import matplotlib.pyplot as plt

#Function Starts
def function(y,cn):
    f, r, p, sumt =[], [], [], 0
    #Taking null array p
    p = [0 for _ in range(0,k)]

    #Taking New null Cnew
    cnew=np.zeros((y.shape[0], k))

    #storing nearest centroid for each point in array r
    for i in range(0,y.shape[1]):
        f = [np.square(np.linalg.norm(y[:,i]-cn[:,j])) for j in range(0,k)]
        r.append(np.argmin(f))
    
    #changing each centroid to centroid of nearest points
        
    for i in range(0,y.shape[1]):
        cnew[:,r[i]]=cnew[:,r[i]] + y[:,i]
        p[r[i]]=p[r[i]]+1
    
    for i in range(0,k):
        if(p[i]!=0):
            cnew[:,i]=(cnew[:,i]/p[i])
   
    
    #checking for stopping condition
    for i in range(0,k):
        sumt=sumt+np.square(np.linalg.norm(cnew[:,i]-cn[:,i]))
    
        
    print(sumt-eps)
    print('\n')    
    
    if(sumt < eps):
        print("enter")
        print('\n')
        print(cnew)
        #if we want to print k klusters we can print r which will show for each
        #point which centroid is nearest
        #print(r)
        return
    else:
        function(y,cnew)

#Function Ends

# Reading Input image
x = plt.imread('test1.jpeg')

eps = 1e-5
#Taking No of Clusters as Input 
k = int(input("Enter the number of clusters:")) 

y = x.reshape(x.shape[2],x.shape[0]*x.shape[1])

#Initializing cn
cn = np.zeros((y.shape[0], k))
o = 0

#Dividing By 255 For standard input should be in between 0 to 1
y = np.array(y) / 255
print(y)

#Initializing first Centroid as centroid of first N/k points , second
#Centroid as centroid of second interval of N/K points and so on 
for i in range(0,k):
    for j in range(o,o-1+int(y.shape[1]/k)):
        cn[:,i] = cn[:,i] + y[:,j]
    cn[:,i] = cn[:,i]/(int(y.shape[1]/k))
    o=o+int(y.shape[1]/k)

#printing initialized centroids
print('\n')        
print(cn)

#calling Function
function(y,cn)
