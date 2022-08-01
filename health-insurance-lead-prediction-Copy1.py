#!/usr/bin/env python
# coding: utf-8

# In[3]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

#import os
#for dirname, _, filenames in os.walk('/kaggle/input'):
 #   for filename in filenames:
  #      print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[4]:


test = pd.read_csv("test_data.csv")
train = pd.read_csv("train_data.csv")


# In[5]:


train.head()


# In[6]:


train = train.drop(columns = ['ID'])


# ### Univariate analysis

# In[7]:


df_target = train[['Response']]
df_target['Response'] = np.where(train['Response']==1,'Res','Not Res')
print(df_target['Response'].value_counts())
print('-'*40)
print(df_target['Response'].value_counts()/len(train)*100)
print('-'*40)
 
fig = plt.figure(figsize=(8,6))
sns.countplot(x='Response',data=df_target, palette='hls')
plt.xlabel('Target')
plt.ylabel('Frequency')
plt.grid(False)
plt.show();


# In[8]:


from sklearn.utils import resample
#create two different dataframe of majority and minority class 
df_majority = train[(train['Response']==0)] 
df_minority = train[(train['Response']==1)] 
# upsample minority class
df_minority_upsampled = resample(df_minority, 
                                 replace=True,    # sample with replacement
                                 n_samples=  28932, # to match majority class
                                 random_state=42)  # reproducible results
# Combine majority class with upsampled minority class
df_upsampled = pd.concat([df_minority_upsampled, df_majority])


# In[9]:


train = df_upsampled.copy()


# In[10]:


df_target = train[['Response']]
df_target['Response'] = np.where(train['Response']==1,'Res','Not Res')
print(df_target['Response'].value_counts())
print('-'*40)
print(df_target['Response'].value_counts()/len(train)*100)
print('-'*40)
 
fig = plt.figure(figsize=(8,6))
sns.countplot(x='Response',data=df_target, palette='hls')
plt.xlabel('Target')
plt.ylabel('Frequency')
plt.grid(False)
plt.show();


# In[9]:


fig, (ax1, ax2) = plt.subplots(1, 2 ,figsize=(10,5))

individual = train[train['Reco_Insurance_Type'] == 'Individual'].copy()

individual.boxplot(column=['Upper_Age'], ax=ax1)
sns.distplot(individual['Upper_Age'], kde=False, bins=30, ax=ax2)

plt.tight_layout()
ax1.grid(False)
ax2.grid(False)
plt.grid(False)
plt.show();


# In[10]:


fig, [[ax1, ax2] , [ax3 , ax4]] = plt.subplots(2, 2 ,figsize=(10,5))

Joint = train[train['Reco_Insurance_Type'] == 'Joint'].copy()

Joint.boxplot(column=['Upper_Age'], ax=ax1)
sns.distplot(Joint['Upper_Age'], kde=False, bins=30, ax=ax2)
Joint.boxplot(column=['Lower_Age'], ax=ax3)
sns.distplot(Joint['Lower_Age'], kde=False, bins=30, ax=ax4)

plt.tight_layout()
ax1.grid(False)
ax2.grid(False)
ax3.grid(False)
ax4.grid(False)
plt.grid(False)
plt.show();


# In[11]:


plt.subplots(figsize=(8,5))
sns.distplot(train['Reco_Policy_Premium'], kde=False)
plt.grid(False)
plt.ylabel('Count')
plt.show()


# In[12]:


plt.subplots(figsize=(10,10))
sns.countplot(y="City_Code", data=train, order = train['City_Code'].value_counts().index)
plt.grid(False)
plt.xlabel('Count')
plt.show()


# In[13]:


train['Region_Code'].value_counts()


# In[14]:


plt.subplots(figsize=(10,10))
sns.countplot(y="Reco_Insurance_Type", data=train, order = train['Reco_Insurance_Type'].value_counts().index)
plt.grid(False)
plt.xlabel('Count')
plt.show()


# In[15]:


plt.subplots(figsize=(10,10))
sns.countplot(y="Accomodation_Type", data=train, order = train['Accomodation_Type'].value_counts().index)
plt.grid(False)
plt.xlabel('Count')
plt.show()


# In[16]:


plt.subplots(figsize=(10,10))
sns.countplot(y="Health Indicator", data=train, order = train['Health Indicator'].value_counts().index)
plt.grid(False)
plt.xlabel('Count')
plt.show()


# In[17]:


plt.subplots(figsize=(10,10))
sns.countplot(y="Holding_Policy_Duration", data=train, order = train['Holding_Policy_Duration'].value_counts().index)
plt.grid(False)
plt.xlabel('Count')
plt.show()


# In[18]:


plt.subplots(figsize=(10,10))
sns.countplot(y="Holding_Policy_Type", data=train, order = train['Holding_Policy_Type'].value_counts().index)
plt.grid(False)
plt.xlabel('Count')
plt.show()


# In[19]:


plt.subplots(figsize=(10,10))
sns.countplot(y="Reco_Policy_Cat", data=train, order = train['Reco_Policy_Cat'].value_counts().index)
plt.grid(False)
plt.xlabel('Count')
plt.show()


# ## **EDA**

# In[20]:


df_age = train[['Response','Upper_Age']].copy()
df_age['bin_person_age'] = pd.qcut(df_age['Upper_Age'].astype(float), q = 5, duplicates='drop', precision=0)
df_age = df_age.groupby('bin_person_age').agg({'Response': ['count', 'sum']})
df_age.columns = df_age.columns.map('_'.join)
df_age['dist'] = df_age['Response_count']/df_age['Response_count'].sum()
df_age['response_rate'] = df_age['Response_sum']/df_age['Response_count']
df_age = df_age.reset_index().copy()
df_age


# In[21]:


sns.set(font_scale=1.1)
sns.set_style("whitegrid")
fig, ax = plt.subplots(1, 2, figsize=(18,5))

### First figure is distribution plot ###
train_individual = train.copy()

sns.distplot(train_individual.loc[train_individual['Response'] == 0, 'Upper_Age'], kde=False, bins=40,
            label = 'Not Res', color = 'green', ax = ax[0])
sns.distplot(train_individual.loc[train_individual['Response'] == 1, 'Upper_Age'], kde=False, bins=20,
             label = 'Response', color = 'red', ax = ax[0])

### Second figure is bar plot + line chart ###

### y axis (left) is bar plot ###
sns.barplot(df_age['bin_person_age'], df_age['dist']*100, data = df_age, color='navajowhite', alpha=.6,ax = ax[1])

### y axis (right) is line chart ###
ax2 = ax[1].twinx()
ax2 = sns.lineplot(data=df_age, x=df_age.index, y=df_age['response_rate']*100                   , marker='o',                  markerfacecolor='darkorange', markersize=6, color='lightslategray', 
                  linewidth=2, label='% response_rate')
for x,y in zip(df_age.index, df_age['response_rate']*100):
          label = "{:.2f}".format(y)
          plt.annotate(label, (x,y), textcoords="offset points", xytext=(0,5), ha='center', color='black') 

ax[0].legend()
ax[0].grid(False)
ax[1].grid(False)
ax2.grid(False)
ax[0].set_ylabel('Count')
ax[1].set_ylabel('% Distribution')
ax2.set_ylabel('% response_rate')
plt.tight_layout() 
plt.show();


# In[22]:


df_age = train[['Response','Lower_Age']].copy()
df_age['bin_person_age'] = pd.qcut(df_age['Lower_Age'].astype(float), q = 5, duplicates='drop', precision=0)
df_age = df_age.groupby('bin_person_age').agg({'Response': ['count', 'sum']})
df_age.columns = df_age.columns.map('_'.join)
df_age['dist'] = df_age['Response_count']/df_age['Response_count'].sum()
df_age['response_rate'] = df_age['Response_sum']/df_age['Response_count']
df_age_lower = df_age.reset_index().copy()
df_age_lower


# In[23]:


sns.set(font_scale=1.1)
sns.set_style("whitegrid")
fig, ax = plt.subplots(1, 2, figsize=(18,5))

### First figure is distribution plot ###
train_individual = train.copy()

sns.distplot(train_individual.loc[train_individual['Response'] == 0, 'Lower_Age'], kde=False, bins=40,
            label = 'Not Res', color = 'green', ax = ax[0])
sns.distplot(train_individual.loc[train_individual['Response'] == 1, 'Lower_Age'], kde=False, bins=20,
             label = 'Response', color = 'red', ax = ax[0])

### Second figure is bar plot + line chart ###

### y axis (left) is bar plot ###
sns.barplot(df_age_lower['bin_person_age'], df_age_lower['dist']*100, data = df_age_lower, color='navajowhite', alpha=.6,
            ax = ax[1])

### y axis (right) is line chart ###
ax2 = ax[1].twinx()
ax2 = sns.lineplot(data=df_age_lower, x=df_age_lower.index, y=df_age_lower['response_rate']*100                   , marker='o',                  markerfacecolor='darkorange', markersize=6, color='lightslategray', 
                  linewidth=2, label='% response_rate')
for x,y in zip(df_age_lower.index, df_age['response_rate']*100):
          label = "{:.2f}".format(y)
          plt.annotate(label, (x,y), textcoords="offset points", xytext=(0,5), ha='center', color='black') 

ax[0].legend()
ax[0].grid(False)
ax[1].grid(False)
ax2.grid(False)
ax[0].set_ylabel('Count')
ax[1].set_ylabel('% Distribution')
ax2.set_ylabel('% response_rate')
plt.tight_layout() 
plt.show();


# In[24]:


df_plot = train.copy()

for i in df_plot._get_numeric_data().columns:
  if i in ['City_Code','Region_Code','Holding_Policy_Type','Reco_Policy_Cat','Response']:
    pass
  else:

    ### Generate dataframe for visualization ###

    df_group = df_plot[['Response', i]]
    df_group['bin_'+i] = pd.qcut(df_group[i].astype(float), q=5, duplicates='drop', precision=0)
    df_group = df_group.groupby('bin_'+i).agg({'Response': ['count', 'sum']})
    df_group.columns = df_group.columns.map('_'.join)
    df_group['dist'] = df_group['Response_count']/df_group['Response_count'].sum()
    df_group['Response_rate'] = df_group['Response_sum']/df_group['Response_count']
    df_group = df_group.reset_index()
 
    x1 = df_group['bin_'+i].astype(str)
    x2 = df_group.index
    y1 = df_group['dist']*100
    y2 = df_group['Response_rate']*100

    sns.set(font_scale=1.1)
    sns.set_style("whitegrid")
    fig, ax1  = plt.subplots(figsize=(9,5))
    plt.tick_params(axis='x', rotation = 45)

    ### y axis (left) is bar plot ###

    ax1.bar(x1,y1, data = df_group, color='navajowhite', alpha=.6)

    ### y axis (right) is line chart ###
    ax2 = ax1.twinx()
    ax2 = sns.lineplot(x2, y2, data = df_group, marker='o',                    markerfacecolor='darkorange', markersize=6, color='lightslategray', 
                    linewidth=2, label='% Response Rate')
    ax1.set_xlabel(i.replace("_"," ").title())
    ax1.set_ylabel('% Distribution')
    ax2.set_ylabel('% Response Rate')
    ax1.grid(False)
    ax2.grid(False)

    for x,y in zip(x2,y2):
            label = "{:.2f}".format(y)
            plt.annotate(label, (x,y), textcoords="offset points", xytext=(0,5), ha='center', color='black') 


    plt.show();  


# In[26]:


def mean_respone_per_category(df, var):
     
    temp_df = pd.Series(df[var].value_counts() / len(df)).reset_index()
    temp_df.columns = [var, '%Dist']

    temp_df = temp_df.merge(df.groupby([var])['Response'].mean().reset_index(),
                            on=var,
                            how='left').sort_values(by='Response').reset_index(drop=True)
    
    fig, ax = plt.subplots(figsize=(8, 4))
    plt.xticks(temp_df.index, temp_df[var], rotation=45)
    ax2 = ax.twinx()
    ax.bar(temp_df.index, temp_df["%Dist"]*100, color='navajowhite', 
           alpha=.6)
    
    ax2.plot(temp_df.index, temp_df["Response"]*100, color='lightslategray', label='Seconds',
            marker='o', markerfacecolor='darkorange', linewidth=2)
    ax.set_ylabel('% Distribution')
    ax.set_xlabel(var)
    ax2.set_ylabel('% Response Rate')
    ax.grid(False)
    ax2.grid(False)
    for x,y in zip(temp_df.index, temp_df["Response"]*100):
          label = "{:.2f}".format(y)
          plt.annotate(label, (x,y), textcoords="offset points", xytext=(0,5), ha='center', color='black') 
    plt.show()

    return temp_df


# In[27]:


train.head()


# In[28]:


for col in train.drop(columns = ["Region_Code","Upper_Age","Lower_Age","Reco_Policy_Premium","Response"]):
    df_cat = mean_respone_per_category(train, col)
    print('*'*70)


# ### Weight of Evidence (WOE) & Information Value (IV)

# In[29]:


train.info()


# In[30]:


df_copy = train.copy()


# In[31]:


train['Holding_Policy_Type'] = train['Holding_Policy_Type'].astype('str')
train['Reco_Policy_Cat'] = train['Reco_Policy_Cat'].astype('str')
train['Region_Code'] = train['Region_Code'].astype('str')


# In[32]:


def fill_missing(df):

  for col in df.columns: 
    if  df[col].dtypes == np.object : 
      df[col] = df[col].fillna(value='Undefined')
    elif df[col].dtypes == np.number:
      df[col] = df[col].fillna(df[col].median()) 
    df[col].replace('nan','Undefined',inplace=True)

  return df


# In[33]:


df_clean = fill_missing(train)
df_clean.info()


# In[34]:


def binning_numeric(df):
  
  for i in df._get_numeric_data().columns:
    if i == 'Response':
      continue
    else:
      df[i] = pd.qcut(df[i].astype(float), q=5, duplicates='drop', precision=0).astype('object')

  return df_clean


# In[35]:


df_bin = binning_numeric(df_clean)
df_bin.info()


# In[36]:


df_woe = df_bin.copy()
df_woe.head()


# In[37]:


d = pd.DataFrame(df_woe.groupby(['Upper_Age']).size(), columns=['total'])

d['count_good'] = df_woe.groupby(['Upper_Age'])['Response'].sum()
d['count_bad'] = d['total']-d['count_good']

d['dist_bad'] = d['count_bad']/d['count_bad'].sum()
d['dist_good'] = d['count_good']/d['count_good'].sum()

d['woe'] = np.log(d.dist_bad/d.dist_good)
d["iv"] = (d.dist_bad-d.dist_good)*np.log(d.dist_bad/d.dist_good)

d = d.replace([np.inf, -np.inf], 0)
d = d.reset_index()

d


# In[38]:


### Loop calculate WOE for each varibale in dataframe ###

def woe_iv(df_woe):

  iv_dict = {}
  final_iv = {}

  for i in df_woe.select_dtypes(object).columns:

    ### Calculate WOE and IV ###
    d = pd.DataFrame(df_woe.groupby([i]).size(), columns=['total'])

    d['count_res'] = df_woe.groupby([i])['Response'].sum()
    d['count_not_res'] = d['total']-d['count_res']
    
    d['dist_not_res'] = d['count_not_res']/d['count_not_res'].sum()
    d['dist_res'] = d['count_res']/d['count_res'].sum()
    
    d['woe'] = np.log(d.dist_res/d.dist_not_res)
    d["iv"] = (d.dist_res-d.dist_not_res)*np.log(d.dist_res/d.dist_not_res)

    d = d.replace([np.inf, -np.inf], 0)
    d = d.reset_index()
    
    ### Append dataframe in dictionary ###
    if i not in final_iv:
      final_iv[i] = []
    final_iv[i].append(d)
    
    ### Map WOE value ###
    woe_dict = d.groupby([i])['woe'].mean().to_dict()
    df_woe['woe_'+i] = df_woe[i].map(woe_dict)
    
    ### Calculate final IV of each feature and append in dictionary
    if i not in iv_dict:
      iv_dict[i] = []
    iv_dict[i].append(d['iv'].sum())
  
  ### Generate IV dataframe
  iv_df = pd.DataFrame.from_dict(iv_dict, orient='index', columns=['IV'])
  iv_df.index.name = 'Feature'

  return final_iv, iv_df, df_woe


# In[39]:


final_iv, IV, df_woe = woe_iv(df_woe)


# In[40]:


IV.sort_values(by='IV', ascending=False)


# In[41]:


df_woe.info()


# In[42]:


def iv_group(df):
  
    if df['IV'] > 0.5:
        val = 'Suspicious'
    elif df['IV'] > 0.3 and df['IV'] <= 0.5 :
        val = 'Strong'
    elif df['IV'] > 0.1 and df['IV'] <= 0.3 :
        val = 'Medium'
    elif df['IV'] > 0.02 and df['IV'] <= 0.1 :
        val = 'Weak'
    else:
        val = 'Not useful'

    return val

IV['Predictive_Power'] = IV.apply(iv_group, axis=1)
IV.sort_values('IV',ascending=False)


# In[43]:


X = df_woe.loc[:, df_woe.columns.str.startswith('woe_')]
y = df_woe['Response']


# In[45]:


y.head()


# In[44]:


X = X[['woe_City_Code', 'woe_Region_Code', 'woe_Accomodation_Type', 'woe_Upper_Age', 'woe_Lower_Age',
       'woe_Health Indicator', 'woe_Holding_Policy_Duration',
       'woe_Holding_Policy_Type', 'woe_Reco_Policy_Cat',
       'woe_Reco_Policy_Premium']]


# In[45]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=99, stratify=y)
    
print(y_train.value_counts(normalize=True))
print(y_test.value_counts(normalize=True))


# ### Train logistic regression

# In[46]:


import statsmodels.api as sm


# In[47]:


model = sm.Logit(y_train, sm.add_constant(X_train)).fit(disp=True, 
                                                        maxiter=100)

model.summary()

results_as_html = model.summary().tables[1].as_html()
summary = pd.read_html(results_as_html, header=0, index_col=0)[0]
summary['Feature'] = summary.index
summary = summary.reset_index(drop=True)
summary['Feature'] = summary['Feature'].str.replace('woe_', '')
summary = summary.set_index(['Feature'], drop=True)
summary


# In[48]:


def train_model(X_train, y_train):

    ### Fit logistic regression ###
    model = sm.Logit(y_train, sm.add_constant(X_train)).fit(disp=True, 
                                                            maxiter=100,
                                                            intercept=True)
    
    ### Generate model summary ###
    results_as_html = model.summary().tables[1].as_html()
    summary = pd.read_html(results_as_html, header=0, index_col=0)[0]
    summary['Feature'] = summary.index
    summary = summary.reset_index(drop=True)
    summary['Feature'] = summary['Feature'].str.replace('woe_', '')
    summary = summary.set_index(['Feature'], drop=True)

    return model, summary


# In[49]:


lr, lr_summary = train_model(X_train, y_train)


# In[50]:


lr_summary


# # Train

# In[51]:


y_pred_train = lr.predict(sm.add_constant(X_train))
y_pred_test = lr.predict(sm.add_constant(X_test))

y_pred_test


# # Variance Inflation Factor (VIF)

# In[52]:


from statsmodels.stats.outliers_influence import variance_inflation_factor


# In[53]:


features = X.copy()
features.columns = features.columns.str.replace('woe_', '')
features = features.assign(const=1)
features

vif = pd.DataFrame()
vif["Feature"] = features.columns
vif["VIF Factor"] = [variance_inflation_factor(features.values, i) for i in range(features.shape[1])]
vif


# ### Model evaluation

# In[54]:


from sklearn import metrics
from sklearn.metrics import roc_curve, auc, classification_report


# In[55]:


def roc_gini(y, y_pred_proba):

  assert y.shape == y_pred_proba.shape
  fpr, tpr, _ = metrics.roc_curve(y, y_pred_proba)
  auc = metrics.roc_auc_score(y, y_pred_proba)
  gini = (2 * auc - 1)*100

  fig = plt.figure(figsize=(8,6))
  plt.plot(fpr, tpr, color='darkorange', label='%s AUC = %0.4f, Gini = %0.2f' % ('Model: ', auc,  gini), 
           linewidth=2.5)
  plt.plot([0, 1], [0, 1], color='navy', linestyle='--', linewidth=2)
  plt.xlim([-0.01, 1.01])
  plt.ylim([-0.01, 1.01])
  plt.xlabel('False Positive Rate')
  plt.ylabel('True Positive Rate')
  plt.legend(loc='lower right',fontsize='small')
  plt.grid(False)
  plt.show()


# In[56]:


get_ipython().system('pip install colorama')


# In[57]:


def ks(target=None, prob=None):
    
    ### Calculate gain table ###
    calculate_ks = {'target': target, 'prob': prob}
    data = pd.DataFrame(calculate_ks, columns = ['target', 'prob'])
    data['target0'] = 1 - data['target']
    data['bucket'] = pd.qcut(data['prob'], 10, duplicates='drop')
    grouped = data.groupby('bucket', as_index = False)
    kstable = pd.DataFrame()
    kstable['min_prob'] = grouped.min()['prob']
    kstable['max_prob'] = grouped.max()['prob']
    kstable['Res']   = grouped.sum()['target']
    kstable['Not_res'] = grouped.sum()['target0']
    kstable['total'] = kstable['Res'] + kstable['Not_res']
    kstable = kstable.sort_values(by="min_prob", ascending=False).reset_index(drop = True)
    kstable['Res_rate'] = (kstable['Res'] / data['target'].sum()).apply('{0:.4%}'.format)
    kstable['Not_res_rate'] = (kstable['Not_res'] / data['target0'].sum()).apply('{0:.4%}'.format)
    kstable['cum_Res_rate']=(kstable['Res'] / data['target'].sum()).cumsum()
    kstable['cum_Not_res_rate']=(kstable['Not_res'] / data['target0'].sum()).cumsum()
    kstable['KS'] = np.round(kstable['cum_Res_rate']-kstable['cum_Not_res_rate'], 4) * 100
    kstable['Resrate_decile'] = (kstable['Res']/kstable['total']).apply('{0:.4%}'.format)
    kstable['cum_Res_rate']= kstable['cum_Res_rate'].apply('{0:.4%}'.format)
    kstable['cum_Not_res_rate']= kstable['cum_Not_res_rate'].apply('{0:.4%}'.format)
    kstable.index = range(1,11)
    kstable.index.rename('Decile', inplace=True)
    kstable = kstable.sort_values(by="min_prob", ascending=True).reset_index(drop = True)
    kstable['cum_total'] = kstable['total'].cumsum()
    kstable['cum_Res'] = kstable['Res'].cumsum()
    kstable['actual_Res'] = (kstable['cum_Res']/kstable['cum_total'])
    kstable = kstable.sort_values(by="actual_Res", ascending=False)
    kstable['actual_Res'] = kstable['actual_Res'].apply('{0:.2%}'.format)
    kstable.index = range(1,11)
    kstable.index.rename('Decile', inplace=True)
    kstable = kstable[['min_prob', 'max_prob', 'Res', 'Not_res', 'total', 'cum_Res', 'cum_total', 'Res_rate', 'Not_res_rate',
           'cum_Res_rate', 'cum_Not_res_rate', 'KS', 'Resrate_decile', 'actual_Res']]
    
    pd.set_option('display.max_columns', 20)
    from colorama import Fore
    print(Fore.RED + "KS is " + str(kstable['KS'].max())+"%"+ " at decile " + str((kstable.index[kstable['KS']==kstable['KS'].max()][0])))
    
    ### Display KS ###
    ks_plot = kstable.copy()
    ks_plot['cum_Res_rate'] = ks_plot['cum_Res_rate'].str.replace('%','').astype(float)
    ks_plot['cum_Not_res_rate'] = ks_plot['cum_Not_res_rate'].str.replace('%','').astype(float)
    fig = plt.figure(figsize=(8,6))
    plt.plot(ks_plot['cum_Res_rate'], color='red',marker = 'o', label='Res')
    plt.plot(ks_plot['cum_Not_res_rate'], color='blue',marker = 's', label = 'Not Res')
    plt.xlim([0.9, 10.05])
    plt.ylim([-0.1, 100.8])
    plt.xlabel('Decile')
    plt.ylabel('Cumulative Probability (%)')
    plt.vlines(x = ks_plot.index[ks_plot['KS']==ks_plot['KS'].max()]
               ,color='black', ymin= ks_plot[['cum_Res_rate','cum_Not_res_rate']].loc[ks_plot['KS'].idxmax()][0], 
               ymax = ks_plot[['cum_Res_rate','cum_Not_res_rate']].loc[ks_plot['KS'].idxmax()][1],
               linestyles='--',
               label='KS = %0.2f' % (ks_plot['KS'].max()))
    plt.legend(loc='lower right',fontsize='small')
    plt.grid(False)
    plt.show()
    
    return(kstable)


# In[58]:


print('Training set')
roc_gini(y_train.values, y_pred_train)
gain_table_train = ks(target=y_train, prob=y_pred_train) 
    
print('Test set')
roc_gini(y_test.values, y_pred_test)
gain_table_test = ks(target=y_test, prob=y_pred_test) 


# ### Summary overall of model

# In[59]:


def summary_model(IV, vif, summary):
    
    summary_model = IV.merge(vif, left_on= 'Feature', right_on='Feature').merge(lr_summary ,on='Feature')
    summary_model = summary_model.rename(columns={'P>|z|': "p-value", "VIF Factor": "vif", "IV":"iv"})
    ### Calculate feature importance ###
    summary_model['feature_importance'] = (summary_model['coef'].abs()/summary_model['coef'].abs().sum())*100
    
    return summary_model


# In[61]:


summary = summary_model(IV, vif, lr_summary)
summary


# ### Model tuning
# 
# Trick for tuning the logistic regression model
# 
# * Remove some features (p-value > 0.05)
# * Remove some features (not risk ranking e.g. loan_amnt)
# * Remove some features (low predictive power from IV e.g. not-useful)

# In[62]:


summary['Feature'].loc[summary['p-value']>0.05].to_list()


# In[63]:


X_2 = X[['woe_Region_Code', 'woe_Reco_Policy_Cat']].copy()
y_2 = df_woe['Response']


# In[64]:


X_2


# In[65]:


X_train_2, X_test_2, y_train_2, y_test_2 = train_test_split(X_2, y_2, test_size=0.2, random_state=99, stratify=y)


# In[66]:


lr_2, lr_2_summary = train_model(X_train_2, y_train_2)


# In[67]:


y_pred_train_2 = lr_2.predict(sm.add_constant(X_train_2))
y_pred_test_2 = lr_2.predict(sm.add_constant(X_test_2))


# In[68]:


print('Training set')
roc_gini(y_train_2.values, y_pred_train_2)
gain_table_train_2 = ks(target=y_train_2, prob=y_pred_train_2)

print('Test set')
roc_gini(y_test_2.values, y_pred_test_2)
gain_table_test_2 = ks(target=y_test_2, prob=y_pred_test_2)


# In[69]:


X = df_woe.loc[:, df_woe.columns.str.startswith('woe_')]
y = df_woe['Response']

X_final = X[['woe_Region_Code', 'woe_Reco_Policy_Cat']].copy()
y_final = y

df_grade = df_woe.copy()
threshold = 0.5

final_model = sm.Logit(y_final, sm.add_constant(X_final)).fit(disp=True, maxiter=100, intercept=True)

df_grade['pd'] = final_model.predict(sm.add_constant(X_final))
df_grade['prediction'] = np.where(df_grade['pd'] >= threshold, 1, 0)

sns.set(font_scale=1.2)
sns.set_style("whitegrid")

roc_gini(y_final.values, df_grade['pd'].values)
gain_table_final = ks(target=y_final, prob=df_grade['pd'])
print('*'*60)
target_names = ['Not res', 'Res']
print(classification_report(y_final.values, df_grade['prediction'].values, target_names=target_names))


# In[82]:


y


# In[83]:


test.drop_duplicates()
y_test.drop_duplicates()
#submission.drop_duplicates()
submission = pd.DataFrame({'ID': test['ID'], 'Response': y})
submission.to_csv('Insurance.csv', index=False)

