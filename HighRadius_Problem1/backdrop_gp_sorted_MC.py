import pandas as pd;
import numpy as np;
import datetime;
from math import ceil
import math;
import BackDropClass;
import pickle;
import collections;

from BackDropClass import BackDropClass;
from numpy import append
def setConditions(column):
    
    #previously 45
    
    if column["delay"] > 15 and column["delay"] <= 30:
        return 1
    elif column["delay"] > 30 and column["delay"] <= 45:
        return 2
    elif column["delay"] > 45:
        return 3
    
    else:
        return 0;


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
        

def giveMeMonth(date):
        year,month,date = (int(x) for x in date.split('-'))
        ans = datetime.date(year,month,date);
        month = ans.strftime("%m");
        #week = ans.strftime("%U");
        return int(month);


data = pd.read_csv("D:\largefiles\\fp_acctdocheader_fleetpride_open_last500kto100k.csv",usecols=['customer_number','doctype','due_date','update_date','open_amount']);

print data;

#list_u_customer_numbers = [];



#list_customer_numbers = [];

list_customer_numbers = data['customer_number'];

print list_customer_numbers;

counter = collections.Counter(list_customer_numbers);

list_customer_numbers_and_count = counter.most_common();

#print list_customer_numbers_and_count;

#list_customer_numbers = data['customer_number'];

print len(list_customer_numbers_and_count);

sorted_customer_list = [];

for i in range(len(list_customer_numbers_and_count)):
    sorted_customer_list.append(list_customer_numbers_and_count[i][0]);
    
print sorted_customer_list;



#list_u_customer_numbers = np.unique(data['customer_number']);

list_u_customer_numbers =sorted_customer_list;

print type(list_u_customer_numbers);

print list_u_customer_numbers[0];

#list of weights;
weights_1 = [];
weights_2 = [];
bias_1 = [];
bias_2 = [];

customer_number = 0;


#len(list_u_customer_numbers)
for i in range(2):
    #customer_number = list_u_customer_numbers[i];
    df = data.groupby(['customer_number']).get_group(list_u_customer_numbers[i]);
    print df;
    df["month"] = df["due_date"].apply(lambda x:giveMeMonth(x))
    df["quarter"]= df["due_date"].apply(lambda x:giveMeQuarter(x));
    print df["quarter"];
    df["week_in_month"] = df["due_date"].apply(lambda x:weekOfMonth(x));
    
    print df["week_in_month"];
    print df["month"]
    
    #df["due_date"] = pd.to_datetime(df["due_date"],infer_datetime_format="%Y%m%d");
    #df["update_date"]= pd.to_datetime(df["update_date"],infer_datetime_format="%Y%m%d");
    df["due_date"] = df["due_date"].apply(lambda x:datetime.datetime.strptime(x,"%Y-%m-%d").strftime("%Y%m%d"));
    
    df["update_date"] = df["update_date"].apply(lambda x:datetime.datetime.strptime(x,"%Y-%m-%d").strftime("%Y%m%d"));
    
    print df["due_date"];
    
    df["day_in_week"] = df["due_date"].apply(lambda x:datetime.datetime.strptime(x,"%Y%m%d").weekday());
    
    print df["day_in_week"];
    
    
    df["delay"] = [(datetime.datetime.strptime(str(x),"%Y%m%d") - datetime.datetime.strptime(str(y),"%Y%m%d")).days for x,y in zip(df["update_date"],df["due_date"])]
    
    print df["delay"];
    
    df["labels"] = 0;
    
    df = df.assign(labels = df.apply(setConditions,axis=1));    
    
    print df["labels"];

#     model = backdrop_module();
#     
#     print type(model);

    #classimbalance      
#     label_0 = df[df["labels"] == 0];
#     
#     label_1 = df[df["labels"] == 1];
#     
#     label_2 = df[df["labels"] == 2];
#     
#     label_3 = df[df["labels"] == 3];
#     
#     df_0 = df[df["labels"] == 0];
#     
#     df_0 = df_0[-len(label_1):]
#     
#     df_1 = label_1;
#     
#     final_df = [df_0,df_1];
#     
#     df = pd.concat(final_df);
#     
#     print df;
#     
#     print df["open_amount"];
#     
    df_features = df[["open_amount","month","quarter","week_in_month","day_in_week"]];
    
    df_labels = df["labels"];
    
    mod = BackDropClass(df);
    
    print type(mod);
    
    X = np.array(df_features);
    
    Y = np.array(df_labels);
    
    K = 4;
    
    #M = 2;
    M = 4;
    N,D = X.shape;
    
    T = np.zeros((N,K));
    
    for i in xrange(N):
        T[i,Y[i]] = 1;
    
    
    # randomly initialize weights
    W1 = np.random.randn(D, M)
    b1 = np.random.randn(M)
    W2 = np.random.randn(M, K)
    b2 = np.random.randn(K)

    #learning_rate = 10e-7
    learning_rate = 10e-6;#higher learning rate converges very fast;
    costs = []
    print "Number of rows of customer data",len(X);
    for epoch in xrange(1000000): #1000000
        output, hidden = mod.forward(X, W1, b1, W2, b2)
        if epoch % 100 == 0:
            c = mod.cost(T, output)
            P = np.argmax(output, axis=1)
            r = mod.classification_rate(Y, P)
            print "cost:", c, "classification_rate:", r
            costs.append(c)
            if(c >= -1.0):
                break;

        
        
            output2,hidden2 = mod.forward(X, W1, b1, W2, b2)
            P2 = np.argmax(output2,axis=1)
            r2 = mod.classification_rate(Y, P2)
        
            print "Train Classification Rate",r2;

            

        # this is gradient ASCENT, not DESCENT
        # be comfortable with both!
        # oldW2 = W2.copy()
        W2 += learning_rate * mod.derivative_w2(hidden, T, output)
        b2 += learning_rate * mod.derivative_b2(T, output)
        W1 += learning_rate * mod.derivative_w1(X, hidden, T, output, W2)
        b1 += learning_rate * mod.derivative_b1(T, output, W2, hidden)

        #print W2,b2,W1,b1;
    
    print W2,b2,W1,b1;
    
    print W2.shape;
    
    #serialization
    
#     W1 = pickle.dumps(W1,protocol=0);
#     W2 = pickle.dumps(W2,protocol=0);
#     b1= pickle.dumps(b1,protocol=0);
#     b2 = pickle.dumps(b2,protocol=0);
#     
    
    
    
    
    
    weights_2.append(W2);
    weights_1.append(W1);
    bias_1.append(b1);
    bias_2.append(b2);
    
#shift+tab for methods shorcut;
print "weights_2",weights_2;
print "weights_1",weights_1;
print "bias_1",bias_1;
print "bias_2",bias_2

final_df = pd.DataFrame()



final_df['weights_2'] = weights_2

final_df['weights_1'] = weights_1;

final_df['bias_1'] = bias_1;

final_df['bias_2'] = bias_2;


final_df["customer_number"] = list_u_customer_numbers[:2];


# for i in range(2):
#     final_df['customer_number'][i] = list_u_customer_numbers[i];
# print final_df;

print final_df;

#final_df = pickle.dumps(final_df.p,protocol=0);

final_df = pickle.dump(final_df,open("final_multiclass_df.p","wb"));

print list_u_customer_numbers[:2];
#final_df.to_csv("weights.csv",columns=['customer_number','weights_1','weights_2','bias_1','bias_2']);





# weights = np.vstack((np.array(weights_2),np.array(weights_1)));    
# 
# print weights;

# weighted_df = pd.DataFrame();
# 
# weighted_df['weights_2'] = weights_2;
# 
# weighted_df['weights_1'] = weights_1;
# 
# weighted_df['bias_1'] = bias_1;
# 
# weighted_df['bias_2'] = bias_2;
# 
# print weighted_df;

    






