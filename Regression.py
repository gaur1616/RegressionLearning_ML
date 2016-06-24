import numpy as np
from scipy.optimize import minimize
from scipy.io import loadmat
from numpy.linalg import det, inv
from math import sqrt, pi
import math
import scipy.linalg as linalg
import scipy.io
import matplotlib.pyplot as plt
import pickle
import sys

def ldaLearn(X,y):
    # Inputs
    # X - a N x d matrix with each row corresponding to a training example
    # y - a N x 1 column vector indicating the labels for each training example
    #
    # Outputs
    # means - A d x k matrix containing learnt means for each of the k classes
    # covmat - A single d x d learnt covariance matrix 
    
    # combining true label column with training examples matrix X
    X = np.concatenate((X,y), axis=1)
    
    # separating matrix X by level of diabates
    low = classCreator(X, 1.)
    lowmid = classCreator(X, 2.)
    lowhigh = classCreator(X, 3.)
    moderate = classCreator(X, 4.)
    high = classCreator(X, 5.)
    
    # calculating means
    means = np.vstack((low.mean(0), lowmid.mean(0), lowhigh.mean(0), moderate.mean(0), high.mean(0)));
    means = np.transpose(means);
    
    #calculating co-variance matrix
    covmat = np.cov(X[:,0:2], rowvar = 0);

    return means,covmat

'''
    ---------------- This function creates and returns class wise training matrix ---------------
'''
def classCreator(X, n):
    mat = np.where(X[:,2] == n);
    result = np.empty([0,2], dtype = float);
    for num in mat:
        result = np.vstack((result,X[num, 0:2]));
    return result
    
def qdaLearn(X,y):
    # Inputs
    # X - a N x d matrix with each row corresponding to a training example
    # y - a N x 1 column vector indicating the labels for each training example
    #
    # Outputs
    # means - A d x k matrix containing learnt means for each of the k classes
    # covmats - A list of k d x d learnt covariance matrices for each of the k classes
    # combining true label column with training examples matrix X
    X = np.concatenate((X,y), axis=1)
    
    # separating matrix X by level of diabates
    low = classCreator(X, 1.)
    lowmid = classCreator(X, 2.)
    lowhigh = classCreator(X, 3.)
    moderate = classCreator(X, 4.)
    high = classCreator(X, 5.)
    
    # calculating means
    means = np.vstack((low.mean(0), lowmid.mean(0), lowhigh.mean(0), moderate.mean(0), high.mean(0)));
    means = np.transpose(means);
    
    # finding the covariance classwise
    lowCov = np.cov(low[:,:], rowvar = 0)
    lowmidCov = np.cov(lowmid[:,:], rowvar = 0)
    lowhighCov = np.cov(lowhigh[:,:], rowvar = 0)
    moderateCov = np.cov(moderate[:,:], rowvar = 0)        
    highCov = np.cov(high[:,:], rowvar = 0)
    
    # calculating resulting covariance
    covmats = [lowCov, lowmidCov, lowhighCov, moderateCov, highCov];
    
    
    return means,covmats

def ldaTest(means,covmat,Xtest,ytest):
    # Inputs
    # means, covmat - parameters of the LDA model
    # Xtest - a N x d matrix with each row corresponding to a test example
    # ytest - a N x 1 column vector indicating the labels for each test example
    # Outputs
    # acc - A scalar accuracy value
    # ypred - N x 1 column vector indicating the predicted labels

    # defining resulting matrix
    result = np.empty(5)
    total = 0
    
    # initially keeping ypred vector of predicted labels to be equal to true
    # labels ahead in code will be modifying ypred predicted labels
    ypred = np.ndarray(shape=(ytest.shape[0],1))
    
    # looping all values
    for num in range(0, Xtest.shape[0]):    
        for j in range(0,5):
            initialValues = Xtest[num,:]
            totalMean = means[:,j]
            
            val = (1.0/2.0)* np.dot(np.dot(np.transpose((initialValues - totalMean)),linalg.inv(covmat)),(initialValues - totalMean));
            result[j] = (1.0/5.0) / (2*math.pi * math.sqrt(linalg.det(covmat))) * math.exp(-val);
        
        # if matched then adding to total
        if (ytest[num] == (np.argmax(result)+1)):
            total += 1;
            
        # filling predicted label
        ypred[num] = (np.argmax(result)+1)
        result = np.empty(5);
    
    acc = (total / float(Xtest.shape[0])) * 100;
    
    return acc,ypred

def qdaTest(means,covmats,Xtest,ytest):
    # Inputs
    # means, covmats - parameters of the QDA model
    # Xtest - a N x d matrix with each row corresponding to a test example
    # ytest - a N x 1 column vector indicating the labels for each test example
    # Outputs
    # acc - A scalar accuracy value
    # ypred - N x 1 column vector indicating the predicted labels

    # IMPLEMENT THIS METHOD
    # defining resulting matrix
    result = np.empty(5)
    total = 0
    
    # initially keeping ypred vector of predicted labels to be equal to true
    # labels ahead in code will be modifying ypred predicted labels
    ypred = np.ndarray(shape=(ytest.shape[0],1))
    
    # looping all values
    for num in range(0, Xtest.shape[0]):    
        for j in range(0,5):
            initialValues = Xtest[num,:]
            totalMean = means[:,j]
            
            val = (1.0/2.0)* np.dot(np.dot(np.transpose((initialValues - totalMean)),linalg.inv(covmats[j])),(initialValues - totalMean));
            result[j] = (1.0/5.0) / (2*math.pi * math.sqrt(linalg.det(covmats[j]))) * math.exp(-val);
       
        # if matched then adding to total
        if (ytest[num] == (np.argmax(result)+1)):
            total += 1;
            
        # filling predicted label
        ypred[num] = (np.argmax(result)+1)
        result = np.empty(5);
        
    acc = (total / float(Xtest.shape[0])) * 100;
    
    return acc,ypred

def learnOLERegression(X,y):
    # Inputs:                                                         
    # X = N x d 
    # y = N x 1                                                               
    # Output: 
    # w = d x 1                                                                

    w = np.dot(np.linalg.inv(np.dot(np.transpose(X), X)), np.dot(np.transpose(X),y));                                             
    return w

def learnRidgeRegression(X,y,lambd):
    # Inputs:
    # X = N x d                                                               
    # y = N x 1 
    # lambd = ridge parameter (scalar)
    # Output:                                                                  
    # w = d x 1  
    
    w = np.dot(np.linalg.inv(np.dot(np.transpose(X),X) + (lambd * X.shape[0] * np.identity(X.shape[1])) ), np.dot(np.transpose(X), y));
                                                
    return w

def testOLERegression(w,Xtest,ytest):
    # Inputs:
    # w = d x 1
    # Xtest = N x d
    # ytest = X x 1
    # Output:
    # rmse

    rmse = calculateRMSE(Xtest, ytest, w);
    
    return rmse
    
def calculateRMSE(Xtest, ytest, w):
    rmse = 0
    
    for num in range(0,Xtest.shape[0]):   
        rmse += (ytest[num] - np.dot(np.transpose(w), Xtest[num])) ** 2;
        
    rmse = float(sqrt(rmse) * 1/Xtest.shape[0]);
    
    return rmse

def regressionObjVal(w, X, y, lambd):

    # compute squared error (scalar) and gradient of squared error with respect
    # to w (vector) for the given data X and y and the regularization parameter
    # lambda                                                                  

    # reshaping w
    w1=np.array([w]).T
    
    # computing error    
    error = computeError(X, y, w1, lambd)

    # computing gradient
    error_grad = computeErrorGrad(X, y, w, lambd)                                        
    
    return error.flatten(), error_grad.flatten()
    
def computeError(X, y, w, lambd):
    computedError = ((np.dot(np.transpose(np.subtract(y,np.dot(X,w))),np.subtract(y,np.dot(X,w))))/(2.0*X.shape[0])) + ((np.dot(lambd,np.dot(np.transpose(w),w)))/2)
    
    return computedError
    
def computeErrorGrad(X, y, w, lambd):
    computedErrorGrad = ((-np.dot(np.transpose(y),X)+np.dot(np.transpose(w),np.dot(np.transpose(X),X)))/X.shape[0]) + np.dot(lambd,np.transpose(w))
    
    return computedErrorGrad
    
def mapNonLinear(x,p):
    # Inputs:                                                                  
    # x - a single column vector (N x 1)                                       
    # p - integer (>= 0)                                                       
    # Outputs:                                                                 
    # Xd - (N x (d+1))                                                         
    # IMPLEMENT THIS METHOD
    Xd= nonLinearRegression(x,p)
    return Xd
    
def nonLinearRegression(x,p):
    result = np.empty([x.shape[0],p+1],dtype=float)
    result[:,0] = 1
    for i in range (x.shape[0]):
        for n in range(1,p+1):
            result[i,n] = math.pow(x[i],n)
            
    return result

# Main script

# Problem 1

# load the sample data
                                                                 
if sys.version_info.major == 2:
    X,y,Xtest,ytest = pickle.load(open('C:\Users\Python\sample.pickle','rb'))
else:
    X,y,Xtest,ytest = pickle.load(open('C:\Users\Python\sample.pickle','rb'),encoding = 'latin1')

# LDA
 
means,covmat = ldaLearn(X,y)

ldaacc = ldaTest(means,covmat,Xtest,ytest)
print('LDA Accuracy = '+str(ldaacc))

# QDA

means,covmats = qdaLearn(X,y)
qdaacc = qdaTest(means,covmats,Xtest,ytest)
print('QDA Accuracy = '+str(qdaacc))

# plotting boundaries

x1 = np.linspace(-5,20,100)
x2 = np.linspace(-5,20,100)
xx1,xx2 = np.meshgrid(x1,x2)
xx = np.zeros((x1.shape[0]*x2.shape[0],2))
xx[:,0] = xx1.ravel()
xx[:,1] = xx2.ravel()

zacc,zldares = ldaTest(means,covmat,xx,np.zeros((xx.shape[0],1)))
plt.contourf(x1,x2,zldares.reshape((x1.shape[0],x2.shape[0])))
plt.scatter(Xtest[:,0],Xtest[:,1],c=ytest)

plt.show()

zacc,zqdares = qdaTest(means,covmats,xx,np.zeros((xx.shape[0],1)))
plt.contourf(x1,x2,zqdares.reshape((x1.shape[0],x2.shape[0])))
plt.scatter(Xtest[:,0],Xtest[:,1],c=ytest)

# Problem 2

if sys.version_info.major == 2:
    X,y,Xtest,ytest = pickle.load(open('C:\Users\Python\diabetes.pickle','rb'))
else:
    X,y,Xtest,ytest = pickle.load(open('C:\Users\Python\diabetes.pickle','rb'),encoding = 'latin1')

# add intercept
X_i = np.concatenate((np.ones((X.shape[0],1)), X), axis=1)
Xtest_i = np.concatenate((np.ones((Xtest.shape[0],1)), Xtest), axis=1)

w = learnOLERegression(X,y)
mle = testOLERegression(w,Xtest,ytest)

w_i = learnOLERegression(X_i,y)
mle_i = testOLERegression(w_i,Xtest_i,ytest)

print('RMSE without intercept '+str(mle))
print('RMSE with intercept '+str(mle_i))

# Problem 3

k = 101
lambdas = np.linspace(0, 1, num=k)
i = 0
rmses3 = np.zeros((k,1))
for lambd in lambdas:
    w_l = learnRidgeRegression(X_i,y,lambd)
    rmses3[i] = testOLERegression(w_l,Xtest_i,ytest)
    i = i + 1

fig=plt.figure()
plt.plot(lambdas,rmses3)
fig.suptitle('lambda vs error values plot(Problem 3)')
plt.xlabel('Lambda Values')
plt.ylabel('error values')

# Problem 4

k = 101
lambdas = np.linspace(0, 0.004, num=k)
i = 0
rmses4 = np.zeros((k,1))
opts = {'maxiter' : 100}    # Preferred value.                                                
w_init = np.ones((X_i.shape[1],1))
for lambd in lambdas:
    args = (X_i, y, lambd)
    w_l = minimize(regressionObjVal, w_init, jac=True, args=args,method='BFGS', options=opts)
    w_l = np.transpose(np.array(w_l.x))
    w_l = np.reshape(w_l,[len(w_l),1])
    rmses4[i] = testOLERegression(w_l,Xtest_i,ytest)
    i = i + 1

fig=plt.figure()
plt.plot(lambdas,rmses4)
fig.suptitle('Lambda vs Error values plot(Problem 4)')
plt.xlabel('Lambda Values')
plt.ylabel('error values')

# Problem 5
pmax = 7
lambda_opt = lambdas[np.argmin(rmses4)]
rmses5 = np.zeros((pmax,2))
for p in range(pmax):
    Xd = mapNonLinear(X[:,2],p)
    Xdtest = mapNonLinear(Xtest[:,2],p)
    w_d1 = learnRidgeRegression(Xd,y,0)
    rmses5[p,0] = testOLERegression(w_d1,Xdtest,ytest)
    w_d2 = learnRidgeRegression(Xd,y,lambda_opt)
    rmses5[p,1] = testOLERegression(w_d2,Xdtest,ytest)

fig=plt.figure()
plt.plot(range(pmax),rmses5)
plt.legend(('No Regularization','Regularization'))
plt.show()