import numpy as np
import matplotlib.pyplot as plt
import datetime;
import time;
import pandas as pd;
import math;

#from sklearn.utils import shuffle

#np.random.seed(1)

def setConditions(column):
    
    #previously 45
    
    if column["final_delay"] > 14:
        return 1
    
    
    else:
        return 0;



def forward(X, W1, b1, W2, b2):
    Z = 1 / (1 + np.exp(-X.dot(W1) - b1))#sigmoid
    
    # rectifier
#     Z = X.dot(W1) + b1
#     Z[Z < 0] = 0
    
    
    A = Z.dot(W2) + b2
    expA = np.exp(A)
    Y = expA / expA.sum(axis=1, keepdims=True)
    return Y, Z


# determine the classification rate
# num correct / num total
def classification_rate(Y, P):
    n_correct = 0
    n_total = 0
    for i in xrange(len(Y)):
        n_total += 1
        if Y[i] == P[i]:
            n_correct += 1
    return float(n_correct) / n_total


#def groupingAmount(amount):
    
    


def derivative_w2(Z, T, Y):
    N, K = T.shape
    M = Z.shape[1] # H is (N, M)

    # # slow
    # ret1 = np.zeros((M, K))
    # for n in xrange(N):
    #     for m in xrange(M):
    #         for k in xrange(K):
    #             ret1[m,k] += (T[n,k] - Y[n,k])*Z[n,m]

    # # a bit faster - let's not loop over m
    # ret2 = np.zeros((M, K))
    # for n in xrange(N):
    #     for k in xrange(K):
    #         ret2[:,k] += (T[n,k]* - Y[n,k])*Z[n,:]

    # assert(np.abs(ret1 - ret2).sum() < 0.00001)

    # # even faster  - let's not loop over k either
    # ret3 = np.zeros((M, K))
    # for n in xrange(N): # slow way first
    #     ret3 += np.outer( Z[n], T[n] - Y[n] )

    # assert(np.abs(ret1 - ret3).sum() < 0.00001)

    # fastest - let's not loop over anything
    ret4 = Z.T.dot(T - Y)
    # assert(np.abs(ret1 - ret4).sum() < 0.00001)

    return ret4

def giveMeMonth(date):
    year,month,date = (int(x) for x in date.split('-'))
    ans = datetime.date(year,month,date);
    month = ans.strftime("%m");
    return int(month);


def weekOfMonth(date):
    year,month,day = (int(x) for x in date.split('-'))
    first_week_month = datetime.datetime(year, month, 1).isocalendar()[1]
    if month == 1 and first_week_month > 10:
        first_week_month = 0
    user_date = datetime.datetime(year, month, day).isocalendar()[1]
    if month == 1 and user_date > 10:
        user_date = 0
    return user_date - first_week_month


def giveMeQuarter(date):
    year,month,date = (int(x) for x in date.split('-'))
    ans = datetime.date(year,month,date);
        
    Q = math.ceil(ans.month/3.0);
    return Q;





def derivative_w1(X, Z, T, Y, W2):
    N, D = X.shape
    M, K = W2.shape

    # slow way first
    # ret1 = np.zeros((X.shape[1], M))
    # for n in xrange(N):
    #     for k in xrange(K):
    #         for m in xrange(M):
    #             for d in xrange(D):
    #                 ret1[d,m] += (T[n,k] - Y[n,k])*W2[m,k]*Z[n,m]*(1 - Z[n,m])*X[n,d]

    # fastest
    dZ = (T - Y).dot(W2.T) * Z * (1 - Z) #sigmoid
    #dZ = (T - Y).dot(W2.T) * (Z > 0) #relu
    ret2 = X.T.dot(dZ)

    # assert(np.abs(ret1 - ret2).sum() < 0.00001)

    return ret2


def derivative_b2(T, Y):
    return (T - Y).sum(axis=0)


def derivative_b1(T, Y, W2, Z):
     return ((T - Y).dot(W2.T) * Z * (1 - Z)).sum(axis=0) #sigmoid
     #return ((T - Y).dot(W2.T) * (Z > 0)).sum(axis=0) #relu


def cost(T, Y):
    tot = T * np.log(Y)
    return tot.sum()


def main():
    
    data = pd.read_csv("D:/largefiles//fp_acctdocheader_fleetpride_open_last500kto100k.csv",index_col=None,usecols=['company_code','invoice_amount','doctype','branch','due_date','update_date','isOpen',"reference","ship_to","payment_terms","error_code_id"]);
    
    data.fillna(0,inplace=True);
    
    #selecting only the rows which are paid;
    df = data[data['isOpen'] == 0];
    
    #filtering the null dates
    df = df[(df["update_date"] != 0.0)&(df["due_date"]!= 0.0)];
    
    #changing the format of the date;
    df["due_date_final"] = df["due_date"].apply(lambda x:datetime.datetime.strptime(x,"%Y-%m-%d").strftime("%Y%m%d"));
    
    df["update_date_final"] = df["update_date"].apply(lambda x:datetime.datetime.strptime(x,"%Y-%m-%d").strftime("%Y%m%d"));
    
    #calculating the delay for each row or invoice
    df["final_delay"] = [(datetime.datetime.strptime(str(start),"%Y%m%d") - datetime.datetime.strptime(str(end),"%Y%m%d")).days for start,end in zip(df["update_date_final"],df["due_date_final"])]   
    
    #labeling the data with the conditions specified above;
    df["final_labels"] = 0;
    
    df = df.assign(final_labels = df.apply(setConditions,axis=1));
    
    #calculating the month of the due_date
    df["month"] = df["due_date"].apply(lambda x:giveMeMonth(x));
    
    #calculating the week in a month
    df["week"]  = df["due_date"].apply(lambda x:weekOfMonth(x));
    
    #calculating quarter of the payment;
    df["quarter"] = df["due_date"].apply(lambda x:giveMeQuarter(x));
    
    
    
    #class imbalance problem as of now not required;
#     label_0 = df[df["final_labels"]==0];
#     
#     label_1 = df[df["final_labels"]==1];
#     
#     #i = 0;
#     
#     
#     df_final_0 = df[df['final_labels']== 0]
#      
#     df_final_0 = df_final_0[-len(label_1):];
#     
    #print df_final_0;   
    
    #print len(df_final_0);
    
    #df_final_1 = label_1;
    
    #df_final_class = np.vstack([df_final_0,label_1]);
    
    
    #final_labels_df = [df_final_0,df_final_1];
    #print df_final_class;
    
    #df_final = pd.concat(final_labels_df);
   
    #df_final = shuffle(df_final);
   
    #binning Block
    df["invoice_amount_grouped"] = 0;
    
    #df["invoice_amount"] = df_final["invoice_amount"].astype(int);
    
    amount_labels = np.arange(10,1809);
    
    bin_values = np.arange(-150000,300000,250);
    
    df["invoice_amount_grouped"] = pd.cut(df["invoice_amount"],bins=bin_values,labels=amount_labels)
    
    #after grouping the amount
    df["invoice_amount_grouped"]= df["invoice_amount_grouped"].astype(int);
       
    #df_features = df_final[["company_code","invoice_amount_grouped","branch","doctype","reference","ship_to","payment_terms","error_code_id"]];
    
    
    #worked with deeplearing below features
    
    df_features = df[["reference","invoice_amount_grouped","ship_to","payment_terms","month","week","quarter"]];
    
    #df_final = shuffle(df_final);
    #df_features = df_final[["company_code","invoice_amount","branch","doctype","reference","ship_to","payment_terms","error_code_id"]];
    
    df_labels = df["final_labels"];
    
    
     #standardization or normalization block:
    
    df_features = (df_features - df_features.mean())/df_features.std();
    
    df_features.fillna(0,inplace=True);
    
    print df_features;
    
    print df_labels;
    
    X = np.array(df_features);
    
    Y = np.array(df_labels);
   
    
    
    K = 2;
    
    #changed M = 3;
    M = 3; # Same Accuracy for 4 neural networks also. for class > 7
    
    N,D = X.shape;
    
    
    T = np.zeros((N, K))
    for i in xrange(N):
        T[i, Y[i]] = 1

    # let's see what it looks like
    plt.scatter(X[:,0], X[:,1], c=Y, s=100, alpha=0.5)
    plt.show()
    
    
    #testing block;
    # testing with the dataset
    df_test = pd.read_csv("D:\largefiles\\fp_acctdocheader_fleetpride_open_last100k.csv",index_col=None,usecols=['company_code','invoice_amount','doctype','branch','due_date','update_date','isOpen',"reference","ship_to","payment_terms","error_code_id"]);
    
    df_test.fillna(0,inplace=True);
    
    len(df_test);
        
    df_test["due_date_final"] = df_test["due_date"].apply(lambda x:datetime.datetime.strptime(x,"%Y-%m-%d").strftime("%Y%m%d"));
    
    
    df_test["update_date_final"] = df_test["update_date"].apply(lambda x:datetime.datetime.strptime(x,"%Y-%m-%d").strftime("%Y%m%d"));
    
    
    df_test["month"] = df_test["due_date"].apply(lambda x:giveMeMonth(x));
    
    df_test["week"] = df_test["due_date"].apply(lambda x:weekOfMonth(x));
    
    df_test["quarter"] = map(giveMeQuarter,df_test["due_date"]);
    
    df_test["final_delay"] = [(datetime.datetime.strptime(str(start),"%Y%m%d") - datetime.datetime.strptime(str(end),"%Y%m%d")).days for start,end in zip(df_test["update_date_final"],df_test["due_date_final"])]   
     
    
    df_test["final_labels"] = 0;
    
    df_test = df_test.assign(final_labels = df_test.apply(setConditions,axis=1));
    
    #amount_labels = np.arange(10,22);
    
    df_test["invoice_amount_grouped"] = pd.cut(df_test["invoice_amount"],bins=bin_values,labels=amount_labels)
    
    df_test["invoice_amount_grouped"]= df_test["invoice_amount_grouped"].astype(int);
    
    #df_test_final = df_test[["company_code","invoice_amount","branch","numerical_date","doctype","reference","ship_to","payment_terms"]];
    
    

    df_test_final = df_test[["reference","invoice_amount_grouped","ship_to","payment_terms","month","week","quarter"]];
    
    df_test_std = (df_test_final - df_test_final.mean())/df_test_final.std();
    
    df_test_labels = df_test["final_labels"];
    
    df_test_std.fillna(0,inplace=True);
    
    Xt = np.array(df_test_std);
    
    Yt = np.array(df_test_labels);


    # randomly initialize weights
    W1 = np.random.randn(D, M)
    b1 = np.random.randn(M)
    W2 = np.random.randn(M, K)
    b2 = np.random.randn(K)

    #learning_rate = 10e-7
    learning_rate = 10e-6;#higher learning rate converges very fast;
    costs = []
    for epoch in xrange(100000):
        output, hidden = forward(X, W1, b1, W2, b2)
        if epoch % 100 == 0:
            c = cost(T, output)
            P = np.argmax(output, axis=1)
            r = classification_rate(Y, P)
            print "cost:", c, "classification_rate:", r
            costs.append(c)
                

        
        
            output2,hidden2 = forward(Xt, W1, b1, W2, b2)
            P2 = np.argmax(output2,axis=1)
            r2 = classification_rate(Yt, P2)
        
            print "Test Classification Rate",r2;

            

        # this is gradient ASCENT, not DESCENT
        # be comfortable with both!
        # oldW2 = W2.copy()
        W2 += learning_rate * derivative_w2(hidden, T, output)
        b2 += learning_rate * derivative_b2(T, output)
        W1 += learning_rate * derivative_w1(X, hidden, T, output, W2)
        b1 += learning_rate * derivative_b1(T, output, W2, hidden)
        
                
        
    plt.plot(costs)
    plt.show()


if __name__ == '__main__':
    main()

