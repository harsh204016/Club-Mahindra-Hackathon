from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor
from csv import DictReader
from sklearn.metrics import mean_squared_error
from math import sqrt

data=pd.read_csv("train.csv")
#number of days
data['checkin_days'] = pd.to_datetime(data['checkin_date'],dayfirst=True)
data['checkout_days'] = pd.to_datetime(data['checkout_date'],dayfirst=True)
data["number_of_days_stay"]=(data['checkout_days']-data["checkin_days"]).dt.days
data['booking_days'] = pd.to_datetime(data['booking_date'],dayfirst=True)
data["no_of_days_beforeBooking"]=(data['checkin_days']-data["booking_days"]).dt.days

#week days 
data["checkin_days"]=data.checkin_days.dt.day_name()
data["checkout_days"]=data.checkout_days.dt.day_name()

#label encoding
X1=LabelEncoder()
data["weekday_checkout_number"]=X1.fit_transform(data["checkout_days"])
X2=LabelEncoder()
data["weekday_checkin_number"]=X2.fit_transform(data["checkin_days"])
X3=LabelEncoder()
data["member_age_buckets"]=X3.fit_transform(data["member_age_buckets"])
X4=LabelEncoder()
data["cluster_code"]=X4.fit_transform(data["cluster_code"])
X5=LabelEncoder()
data['reservationstatusid_code']=X5.fit_transform(data['reservationstatusid_code'])
X6=LabelEncoder()
data['resort_id']=X6.fit_transform(data['resort_id'])

#encoding=data.groupby('resort_id').size()
#encoding=encoding/len(data)
#data['resort_id']=data.resort_id.map(encoding)


data.drop(['reservation_id', 'booking_date', 'checkin_date', 'checkout_date','booking_days'
           ,'checkin_days', 'checkout_days','memberid'],
axis=1,inplace=True)

data.columns
data.shape

data.fillna(0,inplace=True)

X=data.iloc[:,[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,19,20,21,22]].values
Y=data.iloc[:,18].values

'''#missing values
from sklearn.preprocessing import Imputer
imputer=Imputer(missing_values='NaN', strategy=0,axis=0)
imputer=imputer.fit(X[:,:])
X[:,:]=imputer.transform(X[:,:])
'''

with open('test.csv') as f:
    reserve_id=[row["reservation_id"] for row in DictReader(f)]




#splitting of train and test set
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.1)


xgb = XGBRegressor(n_estimators=100, learning_rate=0.1, gamma=1, subsample=0.7,
                           colsample_bytree=1, max_depth=10 , min_child_weight=4)

xgb.fit(X_train,y_train)

# Predicting the Test set results
y_pred = xgb.predict(X_test)

#numpy.savetxt("ypred1.csv", y_pred1, delimiter=",",header="amount_spent_per_room_night_scaled")

rms=sqrt(mean_squared_error(y_test,y_pred))
print("rms",rms)
######################## TEST DATA ###################################################


data2=pd.read_csv("test.csv")

data2['checkin_days'] = pd.to_datetime(data2['checkin_date'],dayfirst=True)
data2['checkout_days'] = pd.to_datetime(data2['checkout_date'],dayfirst=True)
data2["number_of_days_stay"]=(data2['checkout_days']-data2["checkin_days"]).dt.days
data2['booking_days'] = pd.to_datetime(data2['booking_date'],dayfirst=True)
data2["no_of_days_beforeBooking"]=(data2['checkin_days']-data2["booking_days"]).dt.days

#week days 
data2["checkin_days"]=data2.checkin_days.dt.day_name()
data2["checkout_days"]=data2.checkout_days.dt.day_name()

#label encoding
X1=LabelEncoder()
data2["weekday_checkout_number"]=X1.fit_transform(data2["checkout_days"])
X2=LabelEncoder()
data2["weekday_checkin_number"]=X2.fit_transform(data2["checkin_days"])
X3=LabelEncoder()
data2["member_age_buckets"]=X3.fit_transform(data2["member_age_buckets"])
X4=LabelEncoder()
data2["cluster_code"]=X4.fit_transform(data2["cluster_code"])
X5=LabelEncoder()
data2['reservationstatusid_code']=X5.fit_transform(data2['reservationstatusid_code'])
X6=LabelEncoder()
data2['resort_id']=X6.fit_transform(data2['resort_id'])


#encoding=data2.groupby('resort_id').size()
#encoding=encoding/len(data2)
#data2['resort_id']=data2.resort_id.map(encoding)


data2.drop(['reservation_id','booking_date','checkin_date','checkout_date','booking_days'
           ,'checkin_days', 'checkout_days','memberid'],axis=1,inplace=True)

    
data2=data2.fillna(0)
data2=data2.iloc[:,:].values

#missing values
#data2[:,:]=imputer.transform(data2[:,:])

data2.shape



#prediction
y_pred1= xgb.predict(data2)
y_pred1.shape

with open('test.csv') as f:
    reserve_id=[row["reservation_id"] for row in DictReader(f)]
y_pred_list=list(y_pred1)
df=pd.DataFrame(data={"reservation_id":reserve_id,
                      "amount_spent_per_room_night_scaled":y_pred_list})
df.to_csv("submit5.csv",sep=',',index=False)


