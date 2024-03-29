#!/usr/bin/env python
# coding: utf-8

# In[2]:


get_ipython().run_line_magic('pylab', 'inline')
import pandas as pd
import matplotlib.pyplot as plt
import time
import geohash as g
from keras.models import Sequential
from keras.layers import Dense


# In[3]:


def deriveDay(x):
    value=x%7
    return value

def deriveHour(x):
	return x.hour
	
def derivePartOfDay(x):
    x=x.hour
    if (x > 4) and (x <= 8):
        return "Early Morning"
    elif (x > 8) and (x <= 11 ):
        return "Morning"
    elif (x >= 12) and (x <= 16):
        return "Noon"
    elif (x > 16) and (x <= 20) :
        return "Evening"
    elif (x > 20) and (x <= 24):
        return "Night"
    elif (x <= 4):
        return "Late Night"
		
def deriveX(strlatlon):
    
    lat=strlatlon[0]
    lon=strlatlon[1]
    
    x=cos(lat)*cos(lon)
    return x

def deriveY(strlatlon):
    
    lat=strlatlon[0]
    lon=strlatlon[1]
    
    y=cos(lat)*sin(lon)
    return y
	
def deriveZ(strlatlon):
    
    lat=strlatlon[0]
    lon=strlatlon[1]
    
    z=sin(lat)
    return z
	
def getHourFloat(x):
    hour=x.hour
    minute= x.minute    
    minute=minute/60   
    total=hour+minute
    
    return total

def getHourFloatT1 (x):
    hour=x.hour
    minute= x.minute
    minute=minute+15
    minute=minute/60   
    total=hour+minute
    
    return total 
	
def getHourFloatT2 (x):
    hour=x.hour
    minute= x.minute
    minute=minute+30
    minute=minute/60   
    total=hour+minute
    
    return total 
	
def getHourFloatT3 (x):
    hour=x.hour
    minute= x.minute
    minute=minute+45
    minute=minute/60   
    total=hour+minute
    
    return total
	
def getHourFloatT4 (x):
    hour=x.hour
    minute= x.minute
    minute=minute+60
    minute=minute/60   
    total=hour+minute
    
    return total
	
def getHourFloatT5 (x):
    hour=x.hour
    minute= x.minute
    minute=minute+75
    minute=minute/60   
    total=hour+minute
    
    return total


# In[7]:


df=pd.read_csv("training.csv")


# In[8]:


df['dayOfWeek']=df['day'].apply(deriveDay)
df['timestamp'] = pd.to_datetime(df['timestamp'],format= '%H:%M' ).dt.time
df['hour']=df['timestamp'].apply(deriveHour)
df['PartOfDay']=df['timestamp'].apply(derivePartOfDay)


# In[9]:


# convert geohash to x,y,z
df['latlong']=df['geohash6'].apply(g.decode)
df['x']=df['latlong'].apply(deriveX)
df['y']=df['latlong'].apply(deriveY)
df['z']=df['latlong'].apply(deriveZ)


# In[10]:


# transform timestamp into two variables that swing back and forth out of sink
df['hourfloat']=df['timestamp'].apply(getHourFloat)
df['x_hour']=np.sin(2.*np.pi*df.hourfloat/24.)
df['y_hour']=np.cos(2.*np.pi*df.hourfloat/24.)

df['hourfloatT1']=df['timestamp'].apply(getHourFloatT1)
df['x_hour_T1']=np.sin(2.*np.pi*df.hourfloatT1/24.)
df['y_hour_T1']=np.cos(2.*np.pi*df.hourfloatT1/24.)

df['hourfloatT2']=df['timestamp'].apply(getHourFloatT2)
df['x_hour_T2']=np.sin(2.*np.pi*df.hourfloatT2/24.)
df['y_hour_T2']=np.cos(2.*np.pi*df.hourfloatT2/24.)

df['hourfloatT3']=df['timestamp'].apply(getHourFloatT3)
df['x_hour_T3']=np.sin(2.*np.pi*df.hourfloatT3/24.)
df['y_hour_T3']=np.cos(2.*np.pi*df.hourfloatT3/24.)

df['hourfloatT4']=df['timestamp'].apply(getHourFloatT4)
df['x_hour_T4']=np.sin(2.*np.pi*df.hourfloatT4/24.)
df['y_hour_T4']=np.cos(2.*np.pi*df.hourfloatT4/24.)

df['hourfloatT5']=df['timestamp'].apply(getHourFloatT5)
df['x_hour_T5']=np.sin(2.*np.pi*df.hourfloatT5/24.)
df['y_hour_T5']=np.cos(2.*np.pi*df.hourfloatT5/24.)


# In[13]:


gen_onehot_features1 = pd.get_dummies(df['dayOfWeek'])


# In[15]:


trainingData=pd.concat([df[['x', 'y','z','x_hour','y_hour']],gen_onehot_features1,df[['demand']]], axis=1)
trainingDataT1=pd.concat([df[['x', 'y','z','x_hour_T1','y_hour_T1']],gen_onehot_features1,df[['demand']]], axis=1)
trainingDataT2=pd.concat([df[['x', 'y','z','x_hour_T2','y_hour_T2']],gen_onehot_features1,df[['demand']]], axis=1)
trainingDataT3=pd.concat([df[['x', 'y','z','x_hour_T3','y_hour_T3']],gen_onehot_features1,df[['demand']]], axis=1)
trainingDataT4=pd.concat([df[['x', 'y','z','x_hour_T4','y_hour_T4']],gen_onehot_features1,df[['demand']]], axis=1)
trainingDataT5=pd.concat([df[['x', 'y','z','x_hour_T5','y_hour_T5']],gen_onehot_features1,df[['demand']]], axis=1)


# In[16]:


seed = 7
numpy.random.seed(seed)


# In[18]:


X = trainingData.iloc[:,0:12]
Y = trainingData.iloc[:,12]


# In[21]:


X_T1 = trainingDataT1.iloc[:,0:12]
X_T2 = trainingDataT2.iloc[:,0:12]
X_T3 = trainingDataT3.iloc[:,0:12]
X_T4 = trainingDataT4.iloc[:,0:12]
X_T5 = trainingDataT5.iloc[:,0:12]


# In[29]:


model = Sequential()
model.add(Dense(16, input_dim=12, kernel_initializer='uniform', activation='relu'))
model.add(Dense(12, kernel_initializer='uniform', activation='relu'))
model.add(Dense(1, kernel_initializer='uniform', activation='sigmoid'))


# In[30]:


model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


# In[31]:


model.fit(X, Y, epochs=150, batch_size=10,  verbose=0)


# In[32]:


predictionsT1= model.predict(X_T1)
predictionsT2 = model.predict(X_T2)
predictionsT3 = model.predict(X_T3)
predictionsT4 = model.predict(X_T4)
predictionsT5 = model.predict(X_T5)


# In[33]:


print('Demands for T1: '+ str(predictionsT1)) # return the demands for timestamp+15mins
print('Demands for T2: '+ str(predictionsT2)) # return the demands for timestamp+30mins
print('Demands for T3: '+ str(predictionsT3)) # return the demands for timestamp+45mins
print('Demands for T4: '+ str(predictionsT4)) # return the demands for timestamp+60mins
print('Demands for T5: '+ str(predictionsT5)) # return the demands for timestamp+75mins


# In[34]:


df['demand_T1']=predictionsT1
df['demand_T2']=predictionsT2
df['demand_T3']=predictionsT3
df['demand_T4']=predictionsT4
df['demand_T5']=predictionsT5


# In[35]:


Result=df[['geohash6', 'day','timestamp','demand','demand_T1','demand_T2','demand_T3','demand_T4','demand_T5']]
Result


# In[ ]:




