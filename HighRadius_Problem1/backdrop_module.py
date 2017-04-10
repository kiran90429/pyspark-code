import pandas as pd;
import numpy as np;
import datetime


class Backdrop(object):

    
    def __init__(self,df={}):
        
        self.df = df;
        
    def setConditions(self,column):
        
        self.column = column;
        
        if self.column["final_delay"] > 14:
            return 1;
        
        else:
            return 0;
        
    def forward(self,X, W1, b1, W2, b2):
        
        self.X = X;
        self.W1 = W1;
        self.b1 = b1;
        self.W2 = W2;
        self.b2 = b2;
        
        
        self.Z = 1 / (1 + np.exp(-self.X.dot(self.W1) - self.b1))#sigmoid
        A = self.Z.dot(self.W2) + self.b2
        expA = np.exp(A)
        Y = expA / expA.sum(axis=1, keepdims=True)
        return Y, self.Z

    def classification_rate(self,Y, P):
        n_correct = 0
        n_total = 0
        self.Y = Y;
        for i in xrange(len(self.Y)):
            n_total += 1
            if self.Y[i] == P[i]:
                n_correct += 1
        return float(n_correct) / n_total

    def derivative_w2(self,Z, T, Y):
        self.Z = Z;
        self.T = T;
        self.Y = Y; 
        
        N, K = self.T.shape
        M = self.Z.shape[1] # H is (N, M)
    
        # fastest - let's not loop over anything
        ret4 = self.Z.self.T.dot(self.T - self.Y)
        # assert(np.abs(ret1 - ret4).sum() < 0.00001)
    
        return ret4
    

    def derivative_w1(self,X, Z, T, Y, W2):
        self.X = X;
        self.Z = Z;
        self.T = T;
        self.Y = Y;
        self.W2 = W2;
        N, D = self.X.shape
        M, K = self.W2.shape

    
        # fastest
        dZ = (self.T - self.Y).dot(self.W2.self.T) * self.Z * (1 - self.Z) #sigmoid
        #dZ = (T - Y).dot(W2.T) * (Z > 0) #relu
        ret2 = self.X.self.T.dot(dZ)
    
        # assert(np.abs(ret1 - ret2).sum() < 0.00001)
    
        return ret2


    def derivative_b2(self,T, Y):
        self.T = T;
        self.Y = Y;
        return (self.T - self.Y).sum(axis=0)


    def derivative_b1(self,T, Y, W2, Z):
        self.T = T;
        self.Y = Y;
        self.W2 = W2;
        self.Z = Z;
        return ((self.T - self.Y).dot(self.W2.self.T) * self.Z * (1 - self.Z)).sum(axis=0) #sigmoid
         #return ((T - Y).dot(W2.T) * (Z > 0)).sum(axis=0) #relu


    def cost(self,T, Y):
        self.T = T;
        self.Y = Y;
        tot = T * np.log(Y)
        return tot.sum()

    
    