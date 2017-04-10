import pandas as pd;
import numpy as np;
import datetime;


def giveMeMonth(date):
        year,month,date = (int(x) for x in date.split('-'))
        ans = datetime.date(year,month,date);
        month = ans.strftime("%m");
        return int(month);


data = pd.read_csv("D:\largefiles\\fp_acctdocheader_fleetpride_closed1.csv",usecols=['customer_number','doctype','due_date','update_date']);

print data;

#list_u_customer_numbers = [];

list_u_customer_numbers = np.unique(data['customer_number']);

print type(list_u_customer_numbers);

print list_u_customer_numbers[0];

#len(list_u_customer_numbers)
for i in range(1):
    df = data.groupby(['customer_number']).get_group(list_u_customer_numbers[i]);
    print df;
    df["month"] = df["due_date"].apply(lambda x:giveMeMonth(x))
    #df["due_date"] = pd.to_datetime(df["due_date"],infer_datetime_format="%Y%m%d");
    #df["update_date"]= pd.to_datetime(df["update_date"],infer_datetime_format="%Y%m%d");
    df["due_date"] = df["due_date"].apply(lambda x:datetime.datetime.strptime(x,"%Y-%m-%d").strftime("%Y%m%d"));
    
    df["update_date"] = df["update_date"].apply(lambda x:datetime.datetime.strptime(x,"%Y-%m-%d").strftime("%Y%m%d"));
    
    print df["due_date"];
    
    df["delay"] = [(datetime.datetime.strptime(str(x),"%Y%m%d") - datetime.datetime.strptime(str(y),"%Y%m%d")).days for x,y in zip(df["update_date"],df["due_date"])]
    
    print df["delay"];
    
#df = data.groupby(['customer_number']).get_group(606940);

#print df;


#print df[1];







