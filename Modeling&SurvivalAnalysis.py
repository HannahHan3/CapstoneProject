#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 19 22:11:01 2020

@author: huiminhan
"""


import pandas as pd
pd.set_option('max_columns',1000)
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import decomposition
from sklearn.metrics import average_precision_score
import pickle
from sklearn.metrics import mean_absolute_error as MAE
import datetime
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression,Lasso
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import f1_score
from sklearn.metrics import auc
from matplotlib import pyplot
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import make_scorer
import xgboost as xgb
from xgboost import XGBClassifier
from tqdm import tqdm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn import linear_model
from sklearn import preprocessing
from scipy import stats
from sklearn.linear_model import LogisticRegression


listing = pd.read_csv('roofstock_marketplace_listing_historical_full.csv')
transaction = pd.read_csv('roofstock_marketplace_transactions_full.csv')
offer = pd.read_csv('roofstock_marketplace_offers_full.csv')
buyer = pd.read_csv('roofstock_marketplace_buyers_full.csv')
newdf=pd.read_csv('without_estimate.csv')
newdf['DISCOUNT']=newdf['OFFER_PRICE']/newdf['LIST_PRICE']
zesti=pd.read_csv('zestimate_data.csv')
zesti_dic=zesti.set_index('LISTING_ID')['zestimate'].to_dict()
newdf['ZESTIMATE']=newdf['LISTING_ID'].map(lambda x: zesti_dic[x])
#newdf['ZESTIMATE_DISCOUNT']=newdf['LIST_PRICE']/newdf['ZESTIMATE']
newdf['ZESTIMATE_DIFFERENCE']=newdf['LIST_PRICE']-newdf['ZESTIMATE']

newdf['OFFER_OR_NOT']=newdf['diff_days'].map(lambda x: 1 if x<180 else 0)
newdf['OFFER_OR_NOT']=newdf['OFFER_OR_NOT'].astype('category')
INSPECTION_TYPE_ID=pd.get_dummies(newdf['INSPECTION_TYPE_ID'],dummy_na=True,prefix='INSPECTION_TYPE_ID')
newdf=pd.concat([newdf,INSPECTION_TYPE_ID],axis=1)
newdf=newdf.drop('INSPECTION_TYPE_ID',axis=1)

ddrop=[]
for i in newdf.columns:
    if 'LEASING' in i:
        ddrop.append(i)

for i in newdf.columns:
    if 'PAYMENTSTATUS' in i:
        ddrop.append(i)

newdf=newdf.drop(ddrop,axis=1)
newdf=newdf.drop(['GMAPPOVHEADING','GMAPPOVPITCH','ISALLOWOFFER','PRICEVISIBILITY','HASAUDIO','ISEXCLUSIVE'],axis=1)

df_has_offer=newdf[newdf['diff_days']<=180]
df_has_offer=df_has_offer[df_has_offer['LISTING_STATUS_For Sale']==1]
ddrop1=[]
for i in newdf.columns:
    if 'LISTING_STATUS' in i:
        ddrop1.append(i)
df_has_offer=df_has_offer.drop(ddrop1[1:],axis=1)


df_has_offer=df_has_offer.groupby('LISTING_ID').apply(lambda t: t[t.EVENT_UTC==t.EVENT_UTC.min()]).drop_duplicates()
df_has_offer=df_has_offer.drop_duplicates(subset=['LISTING_ID','EVENT_UTC'],keep='first')
print(len(df_has_offer))
df_has_offer=df_has_offer.drop('LISTING_ID',axis=1).reset_index().drop('level_1',axis=1)

# Survival analysis
listing['REC_END_TS']=listing['REC_END_TS'].map(lambda x: x[:10])
listing=listing[listing['LIST_PRICE'].isnull()==False]
df_list=[]
for i in tqdm(df_has_offer.index):
    try:
        offer_date=df_has_offer['EVENT_UTC'][i][:10]
        initial_publish_date=df_has_offer['LISTING_INITIAL_PUBLISH_TS'][i][:10]
        listing_id=df_has_offer['LISTING_ID'][i]
        listing1=listing[(listing['LISTING_ID']==listing_id)&(listing['REC_END_TS']<=offer_date)&(listing['REC_END_TS']>=initial_publish_date)]
        listing1['REC_END_TS']=listing1['REC_END_TS'].map(lambda x:x[:10])
        xx=listing1[['LISTING_ID','REC_END_TS','LISTING_INITIAL_PUBLISH_TS','LIST_PRICE']].sort_values('REC_END_TS').drop_duplicates(subset=['REC_END_TS'],keep='first')
        xx1=xx.reset_index()
        initial_list_price=xx1['LIST_PRICE'][0]
        xx=xx.dropna(subset=['LISTING_INITIAL_PUBLISH_TS'])
        xx['LISTING_INITIAL_PUBLISH_TS']=xx['LISTING_INITIAL_PUBLISH_TS'].map(lambda x:x[:10])
        xx['REC_END_TS']=pd.to_datetime(xx['REC_END_TS'])
        xx['LISTING_INITIAL_PUBLISH_TS']=pd.to_datetime(xx['LISTING_INITIAL_PUBLISH_TS'])
        xx['DAYS_ON_MARKET']=xx['REC_END_TS']-xx['LISTING_INITIAL_PUBLISH_TS']
        xx['DAYS_ON_MARKET']=xx['DAYS_ON_MARKET'].map(lambda x: x.days)
        diff_days=max(2,(df_has_offer.iloc[i:i+1,]['diff_days'][i]))
        s_df=pd.concat([(df_has_offer.iloc[i:i+1,])]*int(diff_days-1))
        s_df=s_df.reset_index().drop('index',axis=1)
        s_df['DAY_ON_MARKET']=list(range(int(diff_days-1)))
        s_df['INITIAL_LIST_PRICE']=initial_list_price
        today_price_list=[]
        for x in s_df.index:
            try:
                mm=xx[xx['DAYS_ON_MARKET']<=s_df['DAY_ON_MARKET'][x]]
                today_price_list.append(mm[mm['DAYS_ON_MARKET']==mm['DAYS_ON_MARKET'].max()]['LIST_PRICE'].iloc[0,])
            except:
                today_price_list.append(initial_list_price)
        s_df['CURRENT_LIST_PRICE']=today_price_list
        s_df['OFFER_TOMORROW']=0
        s_df=s_df.set_value(s_df.index[-1],'OFFER_TOMORROW',1)
        df_list.append(s_df)
    except:
        print(i)
df_survival=pd.concat(df_list)
df_survival=df_survival.reset_index().drop('index',axis=1)

dd_list=[]
for i in tqdm(df_survival['LISTING_ID'].unique()):
    aa=df_survival[df_survival['LISTING_ID']==i]
    bb=aa.tail(1)
    cc=bb.copy()
    cc['OFFER_TOMORROW']=0
    dd_list.append(pd.concat([aa]+[cc]*(179-len(aa))))

df_survival2=pd.concat(dd_list)

train_id=random.sample(list(df_survival['LISTING_ID'].unique()),int(len(list(df_survival['LISTING_ID'].unique()))*0.6))
test_id=list(set(df_survival['LISTING_ID'].unique())-set(train_id))
df_survival1=df_survival2.drop(['EVENT_UTC','OFFER_OR_NOT',
                                                                 'LISTING_INITIAL_PUBLISH_TS','DISCOUNT','OFFER_OR_NOT','ZESTIMATE',
                                                                 'PREVIOUSYEARLYPROPERTYTAXES'],axis=1)  
df_survival1['OFFER_TOMORROW']=df_survival1['OFFER_TOMORROW'].astype('int')                     
                      
X_train=df_survival1[df_survival1['LISTING_ID'].isin(train_id)].drop(['diff_days','OFFER_TOMORROW','LISTING_ID'],axis=1)
X_test=df_survival1[df_survival1['LISTING_ID'].isin(test_id)].drop(['diff_days','OFFER_TOMORROW','LISTING_ID'],axis=1) 
y_train=df_survival1[df_survival1['LISTING_ID'].isin(train_id)]['OFFER_TOMORROW']
y_test=df_survival1[df_survival1['LISTING_ID'].isin(test_id)]['OFFER_TOMORROW']

abcc=[]
for i in range(len(df_survival1.columns)):
    if 'COMPUTED' in df_survival1.columns[i]:
        abcc.append(i)

PCA_computed = PCA(n_components=0.99995)
PCA_com = PCA_computed.fit(df_survival1.iloc[:,27:45])
#principal_com_Df = pd.DataFrame(data = PCA_com
             #, columns = ['COM_PC1', 'COM_PC2','COM_PC3','COM_PC4','COM_PC5','COM_PC6','COM_PC7','COM_PC8'])
com_df=pd.DataFrame(PCA_computed.transform(df_survival1.iloc[:,27:45]))
com_df.columns=['COM_PC1', 'COM_PC2','COM_PC3','COM_PC4','COM_PC5','COM_PC6','COM_PC7','COM_PC8']


df_survival1=df_survival1.drop(df_survival.iloc[:,27:45], axis=1)
df_survival1=df_survival1.join(com_df)


# # Random forest


rf=RandomForestClassifier()
rf_model=rf.fit(X_train,y_train)
rf_pred=rf.predict(X_test)
rf_acc=accuracy_score(rf_pred,y_test)
print('random forest accuracy for predicting offer_or_not is '+str(rf_acc))
#print('random forest auc: '+str(roc_auc_score(rf_pred,y_test)))
print('random forest average precision score: '+str(average_precision_score(rf_pred,y_test)))

importances = rf.feature_importances_
rf_importance_df=pd.DataFrame({'feature':list(df_survival.drop(['diff_days','EVENT_UTC','LISTING_ID','OFFER_TOMORROW','OFFER_OR_NOT',
                                                                 'LISTING_INITIAL_PUBLISH_TS','DISCOUNT','OFFER_OR_NOT','ZESTIMATE',
                                                                 'PREVIOUSYEARLYPROPERTYTAXES','OFFER_TOMORROW'],
                                                         axis=1).columns),'importance':importances})
rf_importance_df_order=rf_importance_df[rf_importance_df['importance']>0].sort_values('importance',ascending=False)
rf_importance_df_order[:20].plot.bar(x='feature')
rf_proba=rf.predict_proba(X_test)[:,1]
tt=X_test.copy()
tt['PROBA']=rf_proba
tt['LISTING_ID']=list(df_survival1[df_survival1['LISTING_ID'].isin(test_id)]['LISTING_ID'])
tt['DIFF_DAYS']=list(df_survival1[df_survival1['LISTING_ID'].isin(test_id)]['diff_days'])
tt=tt.groupby('LISTING_ID').apply(lambda t: t[t.PROBA==t.PROBA.max()]).drop_duplicates()
tt=tt.drop('LISTING_ID',axis=1).reset_index().drop('level_1',axis=1)
tt=tt.drop_duplicates('LISTING_ID',keep='last')
print('mae for survival analysis for random forest: '+str(np.mean(abs(tt['DIFF_DAYS']-tt['DAY_ON_MARKET']))))
print('RMSE for survival analysis for random forest: '+str(np.mean((tt['DIFF_DAYS'])**2-(tt['DAY_ON_MARKET'])**2)))


# # logistic regression

rf=LogisticRegression()
rf_model=rf.fit(X_train,y_train)
rf_pred=rf.predict(X_test)
rf_acc=accuracy_score(rf_pred,y_test)
print('random forest accuracy for predicting offer_or_not is '+str(rf_acc))
#print('random forest auc: '+str(roc_auc_score(rf_pred,y_test)))
print('random forest average precision score: '+str(average_precision_score(rf_pred,y_test)))


tt=X_test.copy()
tt['PROBA']=rf_proba
tt['LISTING_ID']=list(df_survival1[df_survival1['LISTING_ID'].isin(test_id)]['LISTING_ID'])
tt['DIFF_DAYS']=list(df_survival1[df_survival1['LISTING_ID'].isin(test_id)]['diff_days'])
tt=tt.groupby('LISTING_ID').apply(lambda t: t[t.PROBA==t.PROBA.max()]).drop_duplicates()
tt=tt.drop('LISTING_ID',axis=1).reset_index().drop('level_1',axis=1)
tt=tt.drop_duplicates('LISTING_ID',keep='last')
print('mae for survival analysis for random forest: '+str(np.mean(abs(tt['DIFF_DAYS']-tt['DAY_ON_MARKET']))))
print('RMSE for survival analysis for random forest: '+str((np.mean((tt['DIFF_DAYS'])**2-(tt['DAY_ON_MARKET'])**2)))**0.5

