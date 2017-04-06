import numpy as np;

import matplotlib.pyplot as plt;

import pandas as pd;

import datetime

import time

#from util3 import plot_decision_boundary

from sklearn.tree import DecisionTreeClassifier;
from sklearn.utils import shuffle;
from sklearn.svm import SVC;

class AdaBoost:
  def __init__(self, M):
    self.M = M

  def fit(self, X, Y):
    self.models = []
    self.alphas = []

    N, _ = X.shape
    W = np.ones(N) / N

    for m in xrange(self.M):
      tree = DecisionTreeClassifier(max_depth=1)
      tree.fit(X, Y, sample_weight=W)
      P = tree.predict(X)

      err = W.dot(P != Y)
      alpha = 0.5*(np.log(1 - err) - np.log(err))

      W = W*np.exp(-alpha*Y*P) # vectorized form
      W = W / W.sum() # normalize so it sums to 1

      self.models.append(tree)
      self.alphas.append(alpha)

  def predict(self, X):
    # NOT like SKLearn API
    # we want accuracy and exponential loss for plotting purposes
    N, _ = X.shape
    FX = np.zeros(N)
    for alpha, tree in zip(self.alphas, self.models):
      FX += alpha*tree.predict(X)
    return np.sign(FX), FX

  def score(self, X, Y):
    # NOT like SKLearn API
    # we want accuracy and exponential loss for plotting purposes
    P, FX = self.predict(X)
    L = np.exp(-Y*FX).mean()
    return np.mean(P == Y), L





def setConditions(column):
    
    #previously 45
    
    if column["final_delay"] > 14:
        return 1
    
#     elif column["delay"] > 30 and column["delay"] <=45:
#         
#         return 2;
#     
#     elif column["delay"] <= 30 and column["delay"] >=0:
#         return 3;
#     
#     elif column["delay"] < 0 and column ["delay"] >= -30:
#         
#         return 4;
    
    else:
        return -1;
    
    
def giveMeMonth(date):
    year,month,date = (int(x) for x in date.split('-'))
    ans = datetime.date(year,month,date);
    month = ans.strftime("%m");
    
    
    
    return int(month);

def giveMeYear(date):
    year,month,date = (int(x) for x in date.split('-'))
    ans = datetime.date(year,month,date);
    year = ans.strftime("%y");
    
    return int(year);

def getData(file):
    
    data = pd.read_csv(file,index_col=None,usecols=['company_code','invoice_amount','doctype','branch','due_date','update_date','isOpen',"reference","ship_to","payment_terms","error_code_id"]);
    
    data.fillna(0,inplace=True);
    
    df_paymentdone = data[data['isOpen'] == 0];
    
    df_paymentdone_clearingdatenotnull = df_paymentdone[(df_paymentdone["update_date"] != 0.0)&(df_paymentdone["due_date"]!= 0.0)];
    #print df_paymentdone_clearingdatenotnull;
    
    df_paymentdone_clearingdatenotnull["due_date_final"] = df_paymentdone_clearingdatenotnull["due_date"].apply(lambda x:datetime.datetime.strptime(x,"%Y-%m-%d").strftime("%Y%m%d"));
    
    
    df_paymentdone_clearingdatenotnull["month"] = df_paymentdone_clearingdatenotnull["due_date"].apply(lambda x:giveMeMonth(x));
    
    df_paymentdone_clearingdatenotnull["year"] = df_paymentdone_clearingdatenotnull["due_date"].apply(lambda x:giveMeYear(x));
    
    df_paymentdone_clearingdatenotnull["update_date_final"] = df_paymentdone_clearingdatenotnull["update_date"].apply(lambda x:datetime.datetime.strptime(x,"%Y-%m-%d").strftime("%Y%m%d"));
    
    df_paymentdone_clearingdatenotnull["final_delay"] = [(datetime.datetime.strptime(str(start),"%Y%m%d") - datetime.datetime.strptime(str(end),"%Y%m%d")).days for start,end in zip(df_paymentdone_clearingdatenotnull["update_date_final"],df_paymentdone_clearingdatenotnull["due_date_final"])]   
        
    df_paymentdone_clearingdatenotnull["final_labels"] = 0;
    
    df_paymentdone_clearingdatenotnull = df_paymentdone_clearingdatenotnull.assign(final_labels = df_paymentdone_clearingdatenotnull.apply(setConditions,axis=1));
    
    #np.random.shuffle(df_paymentdone_clearingdatenotnull,inplace=True);
    
    #df_paymentdone_clearingdatenotnull = shuffle(df_paymentdone_clearingdatenotnull);
    
    label_0 = df_paymentdone_clearingdatenotnull[df_paymentdone_clearingdatenotnull["final_labels"]==-1];
    
    label_1 = df_paymentdone_clearingdatenotnull[df_paymentdone_clearingdatenotnull["final_labels"]==1];
    
    #i = 0;
    
    
    df_final_0 = df_paymentdone_clearingdatenotnull[df_paymentdone_clearingdatenotnull['final_labels']== -1]
     
    df_final_0 = df_final_0[-len(label_1):];
    
    #print df_final_0;   
    
    #print len(df_final_0);
    
    df_final_1 = label_1;
    
    #df_final_class = np.vstack([df_final_0,label_1]);
    
    
    final_labels_df = [df_final_0,df_final_1];
    #print df_final_class;
    
    df_final = pd.concat(final_labels_df);
   
    #df_final = shuffle(df_final);
    
    label_0 = df_final[df_final["final_labels"]==-1];
    
    label_1 = df_final[df_final["final_labels"]==1];
    
    print len(label_0);
    
    print len(label_1);
    
    #print df_final;
    
    #print df_paymentdone_clearingdatenotnull;
    
    df_final["feature_0"] = 0;
    
    df_final["feature_1"] = 0;
    
    df_final["feature_2"]=0;
    
    df_final[["feature_0"]] = (df_final["company_code"])*(df_final["reference"])
    
    df_final[["feature_1"]] = (df_final["branch"]*df_final["doctype"])
    
    df_final[["feature_2"]] = (df_final["month"]*df_final["year"]);
    
    print df_final;
    #print df_paymentdone_clearingdatenotnull;
    
    #df_features = df_final[["company_code","invoice_amount","branch","doctype","reference","ship_to","payment_terms","error_code_id"]];
    
    #np.random.shuffle(df_final);
    
    df_final = shuffle(df_final);
    
    df_features = df_final[["feature_0","feature_1","ship_to","payment_terms","month"]];
    
    
    #df_features = df_final[["company_code","invoice_amount","branch","doctype","reference","ship_to","payment_terms","error_code_id"]];
    
    df_labels = df_final["final_labels"];
    
    labels = np.unique(df_labels);
    
#     print df_features;
#     
#     print df_labels;
#     
#     print labels;
#     
#     print len(label_0);
#     
#     print len(label_1);
#     
    #standardization:
    
    df_features = (df_features - df_features.mean())/df_features.std();
    
    df_features.fillna(0,inplace=True);
    
    print df_features;
    
    print df_labels;
    
    X = np.array(df_features);
    
    Y = np.array(df_labels);
    
   
    #modelling test:
    df_test = pd.read_csv("D:\largefiles\\fp_acctdocheader_fleetpride_open_last100k.csv",index_col=None,usecols=['company_code','invoice_amount','doctype','branch','due_date','update_date','isOpen',"reference","ship_to","payment_terms","error_code_id"]);
    
    df_test.fillna(0,inplace=True);
    
    #shuffle test data;
    
    #df_test = shuffle(df_test);
    
    #df_test = df_test.head()
    
    len(df_test);
        
    df_test["due_date_final"] = df_test["due_date"].apply(lambda x:datetime.datetime.strptime(x,"%Y-%m-%d").strftime("%Y%m%d"));
    
    
    df_test["update_date_final"] = df_test["update_date"].apply(lambda x:datetime.datetime.strptime(x,"%Y-%m-%d").strftime("%Y%m%d"));
    
    df_test["month"] = df_test["due_date"].apply(lambda x:giveMeMonth(x));
    
    df_test["year"] = df_test["due_date"].apply(lambda x:giveMeYear(x));
    
    #df_test = df_test[(df_test["update_date"] != 0.0)&(df_test["due_date"]!= 0.0)];
    #print df_test;
    
   
    
    #df_test.sort_values(by="numerical_date",ascending=0, inplace=True);
    
    df_test["final_delay"] = [(datetime.datetime.strptime(str(start),"%Y%m%d") - datetime.datetime.strptime(str(end),"%Y%m%d")).days for start,end in zip(df_test["update_date_final"],df_test["due_date_final"])]   
     
    #print df_test["final_delay"];
     
    #print len(df_test["final_delay"]);
     
    #print df_test["final_delay"].mean();
    #print df_test.head(10);
    #plt.hist(df_paymentdone_clearingdatenotnull["delay"]);
    #plt.show();
    
    df_test["final_labels"] = 0;
    
    df_test = df_test.assign(final_labels = df_test.apply(setConditions,axis=1));
    
    #df_test_final = df_test[["company_code","invoice_amount","branch","numerical_date","doctype","reference","ship_to","payment_terms"]];
    
    df_test["feature_0"] = 0;
    
    df_test["feature_1"] = 0;
    
    df_test["feature_2"]= 0;
    
    df_test["feature_0"] = (df_test["company_code"])*(df_test["reference"]);
    
    df_test["feature_1"] = (df_test["branch"])*(df_test["doctype"]);
    
    df_test["feature_2"] = (df_test["month"])*(np.sqrt(df_test["year"]));
    
    #df_test_final = df_test[["company_code","invoice_amount","branch","doctype","reference","ship_to","payment_terms","error_code_id"]];
    
    #df_test_final = df_test[["feature_0","feature_1","ship_to","payment_terms","month","year"]];
    
    
    df_test_final = df_test[["feature_0","feature_1","ship_to","payment_terms","month"]];
    
    #df_test_final = shuffle(df_test_final)
    #df_test_final = df_test[["company_code","invoice_amount","branch","doctype","reference","ship_to","payment_terms","numerical_day","WeekNumber"]];
    
    df_test_std = (df_test_final - df_test_final.mean())/df_test_final.std();
    
    df_test_labels = df_test["final_labels"];
    
    df_test_std.fillna(0,inplace=True);
    
    #df_test_labels[df_test_labels == 0] = -1;
    
    
    Xtest = np.array(df_test_std);
    
    #Xtest = np.array(df_test_std);
    
    Ytest = np.array(df_test_labels);
    


    T = 100
    train_errors = np.empty(T)
    test_losses = np.empty(T)
    test_errors = np.empty(T)
    for num_trees in xrange(T):
        if num_trees == 0:
            train_errors[num_trees] = None
            test_errors[num_trees] = None
            test_losses[num_trees] = None
            continue
        if num_trees % 20 == 0:
            print num_trees

        model = AdaBoost(num_trees)
        
        model.fit(X, Y)
        #acc = model.score(Xtest,Ytest);
        acc, loss = model.score(Xtest, Ytest)
        acc_train, _ = model.score(X, Y)
        #acc_train = model.score(X,Y);
        train_errors[num_trees] = 1 - acc_train
        test_errors[num_trees] = 1 - acc
        test_losses[num_trees] = loss
        #plt.scatter(X[:,2],X[:,1], s=100, c=Y, alpha=0.5)
        #plot_decision_boundary(X, model)
        #plt.show()

        if num_trees == T - 1:
            print "final train error:", 1 - acc_train
            print "final test error:", 1 - acc

    plt.plot(test_errors, label='test errors')
    plt.plot(test_losses, label='test losses')
    plt.legend()
    plt.show()

    plt.plot(train_errors, label='train errors')
    plt.plot(test_errors, label='test errors')
    plt.legend()
    plt.show()

    
    
def accuracy(y_true,y_pred):
    return np.mean(y_true == y_pred);

    
    
    
    #plt.scatter()
    
    
if __name__ == '__main__':
    
    getData("D:\largefiles\\fp_acctdocheader_fleetpride_open_last500kto100k.csv");