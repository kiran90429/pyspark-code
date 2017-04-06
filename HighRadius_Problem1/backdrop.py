import numpy as np
import matplotlib.pyplot as plt
import datetime;
import time;
import pandas as pd;
from sklearn.utils import shuffle

np.random.seed(1)

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

    data = pd.read_csv("D:\largefiles\\fp_acctdocheader_fleetpride_closed1.csv",usecols=['customer_number','company_code','invoice_amount','doctype','branch','due_date','update_date','isOpen',"reference","ship_to","payment_terms","error_code_id"]);
    
    #print len(np.unique(data.customer_number));
    
    print len(data);
    
    data2 = pd.read_csv("D:\largefiles\\fp_acctdocheader_fleetpride_open_last500kto100k.csv",usecols=['customer_number','company_code','invoice_amount','doctype','branch','due_date','update_date','isOpen',"reference","ship_to","payment_terms","error_code_id"]);
     
    print len(data2);
    
    #df = pd.merge(data,data2,how='left',on=['customer_number','company_code','invoice_amount','doctype','branch','due_date','update_date','isOpen',"reference","ship_to","payment_terms","error_code_id"]);
    
    #print len(df);
    
    #df =  df.dropna(how='all');
    
    #print len(df);
    
    #df = data[(~data.customer_number.isin(data2.customer_number)) & (~data.invoice_amount.isin(data2.invoice_amount))];
    
    #print len(df);
    
    data = data[~data.isin(data2)].dropna();
    
    print len(data);
    #print len(data[~data.customer_number.isin(data2.customer_number)]);
    
    data3 = pd.read_csv("D:\largefiles\\fp_acctdocheader_fleetpride_open_last100k.csv",usecols=['customer_number','company_code','invoice_amount','doctype','branch','due_date','update_date','isOpen',"reference","ship_to","payment_terms","error_code_id"]);
    print len(data3);
#      
    #print len(data2);
     
    data = data[~data.isin(data3)].dropna();
    print len(data);
    
    data.fillna(0,inplace=True);
    
    df_paymentdone = data[data['isOpen'] == 0];
    
    df_paymentdone_clearingdatenotnull = df_paymentdone[(df_paymentdone["update_date"] != 0.0)&(df_paymentdone["due_date"]!= 0.0)];
    #print df_paymentdone_clearingdatenotnull;
    
    df_paymentdone_clearingdatenotnull["due_date_final"] = df_paymentdone_clearingdatenotnull["due_date"].apply(lambda x:datetime.datetime.strptime(x,"%Y-%m-%d").strftime("%Y%m%d"));
    
    
    df_paymentdone_clearingdatenotnull["month"] = df_paymentdone_clearingdatenotnull["due_date"].apply(lambda x:giveMeMonth(x));
    
    #df_paymentdone_clearingdatenotnull["year"] = df_paymentdone_clearingdatenotnull["due_date"].apply(lambda x:giveMeYear(x));
    
    df_paymentdone_clearingdatenotnull["update_date_final"] = df_paymentdone_clearingdatenotnull["update_date"].apply(lambda x:datetime.datetime.strptime(x,"%Y-%m-%d").strftime("%Y%m%d"));
    
    df_paymentdone_clearingdatenotnull["final_delay"] = [(datetime.datetime.strptime(str(start),"%Y%m%d") - datetime.datetime.strptime(str(end),"%Y%m%d")).days for start,end in zip(df_paymentdone_clearingdatenotnull["update_date_final"],df_paymentdone_clearingdatenotnull["due_date_final"])]   
        
    df_paymentdone_clearingdatenotnull["final_labels"] = 0;
    
    df_paymentdone_clearingdatenotnull = df_paymentdone_clearingdatenotnull.assign(final_labels = df_paymentdone_clearingdatenotnull.apply(setConditions,axis=1));
    
    #np.random.shuffle(df_paymentdone_clearingdatenotnull,inplace=True);
    
    #df_paymentdone_clearingdatenotnull = shuffle(df_paymentdone_clearingdatenotnull);
    
    label_0 = df_paymentdone_clearingdatenotnull[df_paymentdone_clearingdatenotnull["final_labels"]==0];
    
    label_1 = df_paymentdone_clearingdatenotnull[df_paymentdone_clearingdatenotnull["final_labels"]==1];
    
    #i = 0;
    
    
    df_final_0 = df_paymentdone_clearingdatenotnull[df_paymentdone_clearingdatenotnull['final_labels']== 0]
     
    df_final_0 = df_final_0[-len(label_1):];
    
    #print df_final_0;   
    
    #print len(df_final_0);
    
    df_final_1 = label_1;
    
    #df_final_class = np.vstack([df_final_0,label_1]);
    
    
    final_labels_df = [df_final_0,df_final_1];
    #print df_final_class;
    
    df_final = pd.concat(final_labels_df);
   
    #df_final = shuffle(df_final);
   
    df_final["feature_0"] = 0;
    
    df_final["feature_1"] = 0;
    
    df_final["feature_2"]=0;
    
    df_final[["feature_0"]] = (df_final["company_code"])*(df_final["reference"])
    
    df_final[["feature_1"]] = (df_final["branch"]*df_final["doctype"])
    
    #df_final[["feature_2"]] = (df_final["month"]*df_final["year"]);
    
    print df_final;
 
    df_final["invoice_amount_grouped"] = 0;
    
    #df_final["invoice_amount"] = df_final["invoice_amount"].astype(int);
    
    amount_labels = np.arange(10,1809);
    
    bin_values = np.arange(-150000,300000,250);
    
    df_final["invoice_amount_grouped"] = pd.cut(df_final["invoice_amount"],bins=bin_values,labels=amount_labels)
    
    #df_final["invoice_amount_grouped"] = pd.cut(df_final["invoice_amount"],bins=bin_values)
    
    
    #amount_labels = np.array(1,2,3,4,5,6,7,8);
    
    #df_final["invoice_amount_grouped"] = amount_labels[df_final["invoice_amount_grouped"]];
    
    df_final["invoice_amount_grouped"]= df_final["invoice_amount_grouped"].astype(int);
       
    #df_features = df_final[["company_code","invoice_amount_grouped","branch","doctype","reference","ship_to","payment_terms","error_code_id"]];
    
    #df_features = df_final[["feature_0","feature_1","invoice_amount","ship_to","payment_terms","month"]];
    
    #worked with deeplearing below features
    
    df_features = df_final[["reference","invoice_amount_grouped","ship_to","payment_terms","month"]];
    
    #df_final = shuffle(df_final);
    #df_features = df_final[["company_code","invoice_amount","branch","doctype","reference","ship_to","payment_terms","error_code_id"]];
    
    df_labels = df_final["final_labels"];
    
    
    labels = np.unique(df_labels);
    

    #shuffle:
    
    #df_features = shuffle(df_features);
 
    #standardization:
    
    df_features = (df_features - df_features.mean())/df_features.std();
    
    df_features.fillna(0,inplace=True);
    
    #df_features = shuffle(df_features);
    
    print df_features;
    
    print df_labels;
    
    X = np.array(df_features);
    
    Y = np.array(df_labels);
   
    
    
    K = 2;
    
    #changed M = 3;
    M = 2; # Same Accuracy for 4 neural networks also. for class > 7
    
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
    df_test["final_delay"] = [(datetime.datetime.strptime(str(start),"%Y%m%d") - datetime.datetime.strptime(str(end),"%Y%m%d")).days for start,end in zip(df_test["update_date_final"],df_test["due_date_final"])]   
     
    
    df_test["final_labels"] = 0;
    
    df_test = df_test.assign(final_labels = df_test.apply(setConditions,axis=1));
    
    #amount_labels = np.arange(10,22);
    
    df_test["invoice_amount_grouped"] = pd.cut(df_test["invoice_amount"],bins=bin_values,labels=amount_labels)
    
    df_test["invoice_amount_grouped"]= df_test["invoice_amount_grouped"].astype(int);
    
    #df_test_final = df_test[["company_code","invoice_amount","branch","numerical_date","doctype","reference","ship_to","payment_terms"]];
    
    df_test["feature_0"] = 0;
    
    df_test["feature_1"] = 0;
    
    df_test["feature_2"]= 0;
    
    df_test["feature_0"] = (df_test["company_code"])*(df_test["reference"]);
    
    df_test["feature_1"] = (df_test["branch"])*(df_test["doctype"]);
    
    #df_test["feature_2"] = (df_test["month"])*(np.sqrt(df_test["year"]));
    
    
    #df_test_final = df_test[["feature_0","feature_1","ship_to","payment_terms","month"]];#change
    
    #df_test_final = df_test[["company_code","invoice_amount_grouped","branch","doctype","reference","ship_to","payment_terms","error_code_id"]];
    
    df_test_final = df_test[["reference","invoice_amount_grouped","ship_to","payment_terms","month"]];
    
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

    
    

