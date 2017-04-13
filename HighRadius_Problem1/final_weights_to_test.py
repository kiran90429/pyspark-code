import pandas as pd;
import numpy as np;
import datetime;
import math;
from BackDropClass import BackDropClass;
import re;
import pickle;
import collections;
#from backdrop_gp import giveMeMonth,giveMeQuarter,weekOfMonth;

def setConditions(column):
    
    #previously 45
    
    if column["delay"] > 7:
        return 1
    
    
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


#weights_data = pd.read_csv("weights.csv",sep=",",dtype=None);

#deserialization:
weights_data = pickle.load(open("final_df.p","rb")); 
    

data = pd.read_csv("D:\largefiles\\fp_acctdocheader_fleetpride_open_last100k.csv",usecols=['customer_number','doctype','due_date','update_date','open_amount']);

print data;

list_customer_numbers = data["customer_number"];

counter = collections.Counter(list_customer_numbers);

list_customer_numbers_and_count = counter.most_common();

#print list_customer_numbers_and_count;

#list_customer_numbers = data['customer_number'];

#print len(list_customer_numbers_and_count);



sorted_customer_list = [];

for i in range(len(list_customer_numbers_and_count)):
    sorted_customer_list.append(list_customer_numbers_and_count[i][0]);
    
print sorted_customer_list;

#list_u_customer_numbers = np.unique(data['customer_number']);

list_u_customer_numbers =sorted_customer_list;

# list_u_customer_numbers = [];
#  
# list_u_customer_numbers = np.unique(data['customer_number']);
 #len(list_u_customer_numbers)
for i in range(10):
    df = data.groupby(['customer_number']).get_group(list_u_customer_numbers[i]);
    #print df;
    df["month"] = df["due_date"].apply(lambda x:giveMeMonth(x))
    df["quarter"]= df["due_date"].apply(lambda x:giveMeQuarter(x));
    #print df["quarter"];
    df["week_in_month"] = df["due_date"].apply(lambda x:weekOfMonth(x));
    
#     print df["week_in_month"];
#     print df["month"]
#     
    #df["due_date"] = pd.to_datetime(df["due_date"],infer_datetime_format="%Y%m%d");
    #df["update_date"]= pd.to_datetime(df["update_date"],infer_datetime_format="%Y%m%d");
    df["due_date"] = df["due_date"].apply(lambda x:datetime.datetime.strptime(x,"%Y-%m-%d").strftime("%Y%m%d"));
    
    df["update_date"] = df["update_date"].apply(lambda x:datetime.datetime.strptime(x,"%Y-%m-%d").strftime("%Y%m%d"));
    
    #print df["due_date"];
    
    df["day_in_week"] = df["due_date"].apply(lambda x:datetime.datetime.strptime(x,"%Y%m%d").weekday());
    
    #print df["day_in_week"];

    df_features = df[["open_amount","month","quarter","week_in_month","day_in_week"]];
    
    df["delay"] = [(datetime.datetime.strptime(str(x),"%Y%m%d") - datetime.datetime.strptime(str(y),"%Y%m%d")).days for x,y in zip(df["update_date"],df["due_date"])]
    
    #print df["delay"];
    
    df["labels"] = 0;
    
    df = df.assign(labels = df.apply(setConditions,axis=1));    
    
    #print df["labels"];
    
    Yt = np.array(df["labels"]);

    #print list_u_customer_numbers[i];
    
    #weights_data[weights_data['customer_number']== list_u_customer_numbers[i]]
    #print weights_data;
    Xt = np.array(df_features);
    
    #print weights_data["weights_1"];
#     W1 = np.matrix(weights_data['weights_1'])
#     W2 = np.matrix(weights_data['weights_2'])
#     b1 = np.matrix(weights_data["bias_1"])
#     b2 = np.matrix(weights_data["bias_2"])
#     
    
#     W1 = weights_data['weights_1'].apply(lambda x:pickle.loads(x)).as_matrix();
#     W2 = weights_data['weights_2'].map(lambda x:pickle.loads(x)).as_matrix();
#     b1 = weights_data["bias_1"].apply(lambda x:pickle.loads(x)).as_matrix();
#     b2 = weights_data["bias_2"].apply(lambda x:pickle.loads(x)).as_matrix();
#     print W1,W2,b1,b2
#     print W1.shape
#     print W2.shape;
#     
#     W1 = weights_data['weights_1'].replace(to_replace='\r\n',value=',',regex=True);
#     #W1 = weights_data['weights_1'].replace(to_replace='(?<=\d)\s+(-?=\d|\-)',regex=True);
#     W2 = weights_data['weights_2'].replace(to_replace='\r\n',value=',',regex=True);
#     #W2 = weights_data['weights_2'].replace(to_replace='(?<=\d)\s+(?=\d|\-)',regex=True);
#     print W1,W2;
#     print np.array(W1.shape),np.array(W2.shape);
#     b1 = weights_data['bias_1'].replace(to_replace='\r\n',value=',',regex=True);
#     b2 = weights_data['bias_2'].replace(to_replace='\r\n',value=',',regex=True);
# #     b1 = np.array(weights_data['bias_1'].iloc[[i]]);
#     b2 = np.array(weights_data['bias_2'].iloc[[i]]);
# #     
#     print Xt,W1,W2,b1,b2;
#     print W1;
#     print W1.shape;
#     print W2;
#     print W2.shape;
# #   
#     W1 = W1.strip();
#     print W1;
# 
#     W1 = W1.replace("\r\n","");
#     
#     print W1;
    
    
    
    #print W2.shape();
    
    #if(weights_data['customer_number'][i] == list_u_customer_numbers[i]):

    

    #df.filter(df['customer_number'] == list_u_customer_numbers[i])
    
    
#     print df['customer_number'];
#     
#     df = df[["open_amount","month","quarter","week_in_month","day_in_week"]];
#    
# 
#     
#     Xt = np.array(df);
#     
#     print Xt.shape


        
    model = BackDropClass();
    
    index_weights = (weights_data['customer_number'] == list_u_customer_numbers[i]).argmax();
    
    #print index_weights;
    
    if(weights_data['customer_number'][index_weights] == df['customer_number'].iloc[0]): 
    
        output2,hidden2 = model.forward(Xt, weights_data["weights_1"][index_weights], weights_data["bias_1"][index_weights], weights_data["weights_2"][index_weights], weights_data["bias_2"][index_weights])
        print output2;
        P2 = np.argmax(output2,axis=1)
        r2 = model.classification_rate(Yt, P2)
        #print P2; 
        print "For a customer number",df['customer_number'].iloc[0];     
        print "Test Classification Rate",r2;
        
    else:
        continue
        
# #---weights data-----------#
# 
# 
# print weights_data[['customer_number','weights_1','weights_2','bias_1','bias_2']];
# 
# #----------calculating classification rate-----------#
# 
# #model = BackDropClass();
# 
# # output2,hidden2 = model.forward(Xt, W1, b1, W2, b2)
# # P2 = np.argmax(output2,axis=1)
# # r2 = model.classification_rate(Yt, P2)
# #         
# # print "Test Classification Rate",r2;



