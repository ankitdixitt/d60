#!/usr/bin/env python
# coding: utf-8

# In[2]:


import fcsparser


# In[3]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# In[4]:


#Importing Libraries

import fcsparser as fcsp
import re
import os
import time
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.svm import SVC, SVR
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler, normalize, LabelEncoder, MinMaxScaler
from sklearn.model_selection import StratifiedKFold, GridSearchCV, ParameterGrid, train_test_split, RepeatedStratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression, Lasso
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.cluster import KMeans
from sklearn.metrics import f1_score, accuracy_score, mean_squared_error, confusion_matrix
from sklearn.decomposition import PCA


import random

# Ignore warnings
import warnings
warnings.filterwarnings('ignore')

get_ipython().run_line_magic('matplotlib', 'inline')


# In[5]:


# Importing data

def load_data(dir):
    print('---- loading data ---->')
    data = {}
    for dirname, _, filename in os.walk(dir):
        for file in filename:
            path = os.path.join(dirname, file)
            # Slicing in order to remove .fcs from filename
            f_name = ''.join(list(file)[:-4])
            data[f_name] = pd.DataFrame(fcsp.parse(path, meta_data_only=False, reformat_meta=True)[1])
    return data, fcsp.parse(path, meta_data_only=False, reformat_meta=True)[0]['_channels_']
 

'''
for dirname, _, filename in os.walk(HERE):
    for file in filename:
      path = os.path.join()
        meta = fcsparser.parse(, meta_data_only=True, reformat_meta=True)
'''

def prepData(x,scaler=None):
        """
        Normalize data
        """
        if not scaler:
                scaler = StandardScaler().fit(x)
        x_transformed= np.nan_to_num(np.array(scaler.transform(x)))
        return normalize(x_transformed),scaler

# Loading fcs file data into a dictionary
global path

dir = os.path.abspath(os.path.dirname("C:\\Users\\ankit\\OneDrive\\Desktop\\FCM data for Ankit\\NCD0_1.fcs"))
dir = os.path.join(dir, "C:\\Users\\ankit\\OneDrive\\Desktop\\FCM data for Ankit")
data, channels = load_data(dir)

# Function to return list of different taken samples
def sample_list(det = data):
    return set([sample[0:3] for sample in det.keys()])


# Function to check the dimensions of our fcs files
def checkDimensions(data, dimensions):
    for sample in data:
        if data[sample].shape != dimensions: print(sample, data[sample].shape)
    return 


# In[6]:


data


# In[7]:


checkDimensions(data,data[random.choice(list(data.keys()))].shape)


# In[8]:


#concatenating the dataframes of all triplicates -->
def triplicates_combined(data = data):
    deta = {}
    for key in data.keys():
        deta[key[:4]]= pd.concat([data[key[:4]+'_1'],data[key[:4]+'_2'],data[key[:4]+'_3']],axis=0)
    return deta

triplicates_combined()


# In[9]:


channels


# In[10]:


data.keys()


# In[11]:


Day_1 = dict([(key,data[key]) for key in data.keys() if key.endswith('_1')])
Day_1.keys()


# In[12]:


Day_1


# In[14]:


def plot_sample_data(data, key, day = 0):
    key = key[:3]
    plot = pd.concat([data[key+str(day)+'_'+str(i)] for i in range(1,4)])
    plt.scatter(plot['FSC-A'],plot['SSC-A'], color = 'red')
    plt.xlabel('FSC-A')
    plt.ylabel('SSC-A')
    plt.title(f'Data for {key}, Week: {day}')
    plt.show()
    

[plot_sample_data(data, 'OHD',i) for i in range(0,10)]


# In[1]:


def plot_sample_data(data, key, day = 0):
    key = key[:3]
    plot = pd.concat([data[key+str(day)+'_'+str(i)] for i in range(1,4)])
    plt.scatter(plot['FSC-A'],plot['SSC-A'], color = 'red')
    plt.xlabel('FSC-A')
    plt.ylabel('SSC-A')
    plt.title(f'Data for {key}, Week: {day}')
    plt.show()
    

[plot_sample_data(data, 'OMD',i) for i in range(0,10)]


# In[14]:


def plot_sample_data(data, key, day = 0):
    key = key[:3]
    plot = pd.concat([data[key+str(day)+'_'+str(i)] for i in range(1,4)])
    plt.scatter(plot['FSC-A'],plot['FSC-H'], color = 'aqua')
    plt.xlabel('FSC-A')
    plt.ylabel('FSC-H')
    plt.title(f'Data for {key}, week: {day}')
    plt.show()
    

[plot_sample_data(data,'NLD',i) for i in range(0,10)]


# In[21]:


# function to plot combined data of all weeks(0-9) for a particular sample

def plot_sample(data, key):
    key = key[:3]
    original = pd.DataFrame()
    for day in range(0,10):
        l = pd.concat([data[key+str(day)+'_'+str(i)] for i in range(1,4)], axis = 0)
        original = pd.concat([original, l])
    plot = original
    print(plot.shape)
    plt.scatter(plot['FSC-A'],plot['FSC-H'], color = 'red')
    plt.xlabel('FSC-A')
    plt.ylabel('FSC-H')
    plt.show()

plot_sample(data,'OHD')


# In[16]:


def plot_sample_data(data, key, day = 1):
    key = key[:3]
    plot = pd.concat([data[key+str(day)+'_'+str(i)] for i in range(1,4)])
    plt.scatter(plot['FSC-A'],plot['FSC-H'], color = 'red')
    plt.xlabel('FSC-A')
    plt.ylabel('SSC-A')
    plt.title(f'Data for {key}, Day: {day}')
    plt.show()
    

[plot_sample_data(data, 'NHD',i) for i in range(0,10)]


# In[17]:


# function to plot combined data of all WEEKS(0-9) for a particular sample

def plot_sample(data, key):
    key = key[:3]
    original = pd.DataFrame()
    for day in range(0,10):
        l = pd.concat([data[key+str(day)+'_'+str(i)] for i in range(1,4)], axis = 0)
        original = pd.concat([original, l])
    plot = original
    print(plot.shape)
    plt.scatter(plot['FSC-A'],plot['FSC-H'], color = 'red')
    plt.xlabel('FSC-A')
    plt.ylabel('SSC-A')
    plt.show()

plot_sample(data,'OLD')


# In[8]:


# Running Clustering with feature vector of size 3 in place of 3000 feature vector

num_clusters = 4
cols2 = random.choice(list(data.keys()))
sample = data[cols2][['FSC-A','SSC-A']]
kmean = KMeans(n_clusters = num_clusters).fit(sample)
kmean.predict(sample)
klabels = kmean.labels_
print(klabels)
#filter rows of original data

cluster_count1 = np.zeros(num_clusters)
for cluster in range(num_clusters):
    filter_lb = sample[klabels == cluster]


fig = plt.figure(figsize=(10,10))

ax = fig.add_subplot(111)
ax.scatter(sample['FSC-A'],sample['SSC-A'],c = klabels) 
plt.xlabel("FSC-A")
plt.ylabel("SSC-A")

plt.show()


# In[10]:


cluster_count1 = np.zeros(num_clusters)
for cluster in range(num_clusters):
    filter_lb = sample[klabels == cluster]


fig = plt.figure(figsize=(10,10))

ax = fig.add_subplot(111)
ax.scatter(sample['FSC-A'],sample['SSC-A'],c = klabels) 
plt.xlabel("FSC-A")
plt.ylabel("SSC-A")

plt.show()


# In[19]:


def elbow_curve(sample = sample):
    wcss = []
    for i in range(1,20):
      kmeans = KMeans(n_clusters=i,init='k-means++',max_iter=300,n_init=10,random_state=0)
      kmeans.fit(sample)
      wcss.append(kmeans.inertia_)
      print('Cluster', i, 'Inertia', kmeans.inertia_,)
    plt.plot(range(1,20),wcss)
 
    plt.title('The Elbow Curve')
    plt.xlabel('Number of clusters')
    plt.ylabel('WCSS') ##WCSS stands for total within-cluster sum of square
    plt.show()

elbow_curve()


# In[13]:


# Running Clustering with feature vector of size 3 in place of 3000 feature vector

num_clusters = 4
cols2 = random.choice(list(data.keys()))
sample = data[cols2][['FSC-A','SSC-A']]
kmean = KMeans(n_clusters = num_clusters).fit(sample)
kmean.predict(sample)
klabels = kmean.labels_

#filter rows of original data

cluster_count1 = np.zeros(num_clusters)
for cluster in range(num_clusters):
    filter_lb = sample[klabels == cluster]


fig = plt.figure(figsize=(10,10))

ax = fig.add_subplot(111)
ax.scatter(sample['FSC-A'],sample['SSC-A'],c = klabels) 
plt.xlabel("FSC-A")
plt.ylabel("SSC-A")

plt.show()


# In[73]:


num_clusters = 8
cols2 = random.choice(list(data.keys()))
sample = data[cols2][['FSC-A','SSC-A']]
kmean = KMeans(n_clusters = num_clusters).fit(sample)
kmean.predict(sample)
klabels = kmean.labels_

#filter rows of original data

cluster_count1 = np.zeros(num_clusters)
for cluster in range(num_clusters):
    filter_lb = sample[klabels == cluster]


fig = plt.figure(figsize=(10,10))

ax = fig.add_subplot(111, )
ax.scatter(sample['FSC-A'],sample['SSC-A'],c = klabels) 

plt.show()


# In[14]:


num_clusters = 8
cols2 = random.choice(list(data.keys()))
sample = data[cols2][['FSC-A','SSC-A']]
kmean = KMeans(n_clusters = num_clusters).fit(sample)
kmean.predict(sample)
klabels = kmean.labels_

#filter rows of original data


cluster_count1 = np.zeros(num_clusters)
for cluster in range(num_clusters):
    filter_lb = sample[klabels == cluster]


fig = plt.figure(figsize=(10,10))

ax = fig.add_subplot(111, projection='3d' )
ax.scatter(sample['FSC-A'],sample['SSC-A'],c = klabels) 
plt.xlabel("FSC-A", )
plt.ylabel("SSC-A",)

plt.show()


# In[70]:


data.keys()
channels


# In[24]:


from sklearn.metrics import silhouette_score


# In[25]:


range_n_clusters = list (range(2,8))
print ("Number of clusters from 2 to 8: \n", range_n_clusters)

cols2 = random.choice(list(data.keys()))
sample = data[cols2][['FSC-A','SSC-A']]
kmean = KMeans(n_clusters = num_clusters).fit(sample)
kmean.predict(sample)
klabels = kmean.labels_
for n_clusters in range_n_clusters:
    clusterer = KMeans (n_clusters=num_clusters).fit(sample)
    preds = clusterer.predict(sample)
    centers = clusterer.cluster_centers_
    

    score = silhouette_score (sample, preds, metric='euclidean')
    print(score)
    #print ("For n_clusters = {}, silhouette score is {})".format(n_clusters, score)
plt.show()    


# In[29]:


from sklearn.metrics import silhouette_samples,silhouette_score


# In[37]:


get_ipython().system('pip install FlowCytometryTools')


# In[18]:


data=sample
print("FSC-A Skewness: {:.3f}".format(data["FSC-A"].skew()))
print("FSC-A Kurtosis: {:.3f}". format(data["FSC-A"].kurt()))
print("SSC-A Skewness: {:.3f}".format(data["SSC-A"].skew()))
print("SSC-A Kurtosis: {:.3f}". format(data["SSC-A"].kurt()))


# In[38]:


import FlowCytometryTools


# In[48]:


data


# In[76]:


num_clusters = 3
cols2 = random.choice(list(data.keys()))
sample = data[cols2][['FSC-A','SSC-A']]
kmean = KMeans(n_clusters = num_clusters).fit(sample)
kmean.predict(sample)
klabels = kmean.labels_

#filter rows of original data


cluster_count1 = np.zeros(num_clusters)
for cluster in range(num_clusters):
    filter_lb = sample[klabels == cluster]


fig = plt.figure(figsize=(10,10))

ax = fig.add_subplot(111, projection='3d' )
ax.scatter(sample['FSC-A'],sample['SSC-A'],c = klabels) 
plt.xlabel("FSC-A", )
plt.ylabel("SSC-A",)

plt.show()


# In[78]:


num_clusters = 3
cols2 = random.choice(list(data.keys()))
sample = data[cols2][['FSC-A','SSC-A']]
kmean = KMeans(n_clusters = num_clusters).fit(sample)
kmean.predict(sample)
klabels = kmean.labels_

#filter rows of original data

cluster_count1 = np.zeros(num_clusters)
for cluster in range(num_clusters):
    filter_lb = sample[klabels == cluster]


fig = plt.figure(figsize=(10,10))

ax = fig.add_subplot(111, )
ax.scatter(sample['FSC-A'],sample['SSC-A'],c = klabels) 

plt.show()


# In[15]:


from scipy import stats
data=sample
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))
sns.distplot(data['FSC-A'], fit= stats.norm, ax=ax[0])
stats.probplot(data['FSC-A'], plot=plt)
ax[0].set_xlabel('FSC-A Distribution', fontsize = 13)
ax[1].set_xlabel('FSC-A Probability', fontsize = 13)
ax[1].yaxis.tick_right() # where the y axis marks will be


# In[21]:


data=sample
plt.title("FSC-A")
ax = sns.distplot(data["FSC-A"])


# In[19]:


data=sample
print("FSC-A Skewness: {:.3f}".format(data["FSC-A"].skew()))
print("FSC-A Kurtosis: {:.3f}". format(data["FSC-A"].kurt()))
print("SSC-A Skewness: {:.3f}".format(data["SSC-A"].skew()))
print("SSC-A Kurtosis: {:.3f}". format(data["SSC-A"].kurt()))


# In[21]:


from scipy import stats
data=data
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))
sns.distplot(data['SSC-A'], fit= stats.norm, ax=ax[0])
stats.probplot(data['SSC-A'], plot=plt)
ax[0].set_xlabel('SSC-A Distribution', fontsize = 13)
ax[1].set_xlabel('SSC-A Probability', fontsize = 13)
ax[1].yaxis.tick_right() # where the y axis marks will be


# In[30]:


data.keys()


# In[31]:


channels


# In[34]:


# Importing data

def load_data(dir):
    print('---- loading data ---->')
    data = {}
    for dirname, _, filename in os.walk(dir):
        for file in filename:
            path = os.path.join(dirname, file)
            # Slicing in order to remove .fcs from filename
            f_name = ''.join(list(file)[:-4])
            data[f_name] = pd.DataFrame(fcsp.parse(path, meta_data_only=False, reformat_meta=True)[1])
    return data, fcsp.parse(path, meta_data_only=False, reformat_meta=True)[0]['_channels_']
 

'''
for dirname, _, filename in os.walk(HERE):
    for file in filename:
      path = os.path.join()
        meta = fcsparser.parse(, meta_data_only=True, reformat_meta=True)
'''

def prepData(x,scaler=None):
        """
        Normalize data
        """
        if not scaler:
                scaler = StandardScaler().fit(x)
        x_transformed= np.nan_to_num(np.array(scaler.transform(x)))
        return normalize(x_transformed),scaler

# Loading fcs file data into a dictionary
global path

dir = os.path.abspath(os.path.dirname('C:\\Users\\ankit\\OneDrive\\Desktop\\FCM data for Ankit\\NCD0_1.fcs'))
dir = os.path.join(dir, 'C:\\Users\\ankit\\OneDrive\\Desktop\\FCM data for Ankit')
data, channels = load_data(dir)

# Function to return list of different taken samples
def sample_list(det = data):
    return set([sample[0:3] for sample in det.keys()])


# Function to check the dimensions of our fcs files
def checkDimensions(data, dimensions):
    for sample in data:
        if data[sample].shape != dimensions: print(sample, data[sample].shape)
    return 


# In[38]:


print('First five rows of sample fcs file -->')
random_sample = random.choice(list(data.keys()))
random_fcs_sample = data[random_sample]
random_fcs_sample.head()


# In[39]:


# Printing Channels present in each fcs file
channels


# In[40]:


features = random_fcs_sample.columns
print('Features in each fcs file : {}'.format(', '.join(features)))


# In[41]:


random_fcs_sample[features].std().plot(kind='bar', figsize=(8,6), title=f"Features Standard Deviation for {random_sample}")


# In[42]:



features_highest_variance = random_fcs_sample[features].std().sort_values(ascending=False)
features_highest_variance.plot(kind = 'bar')


# In[43]:


cm = np.corrcoef(random_fcs_sample[features].values.T)
sns.set(font_scale=1.0)
fig = plt.figure(figsize=(10, 8))
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 8}, yticklabels=features, xticklabels=features)
plt.title('Features Correlation Heatmap')
plt.show()


# In[47]:


deta = triplicates_combined()

def k_means_(num_clusters = 5, data = deta, samp = random.choice(list(deta.keys()))):
    sample = data[samp][['FSC-A','SSC-A','AmCyan-A']]
    sample_vectors = np.array([[]])
    for i in range(300):
        vector = sample.sample(n = 1000).to_numpy().flatten()
        sample_vectors = np.append(sample_vectors, [vector])
    sample_vectors = np.reshape(sample_vectors, (300,3000))
    kmean = KMeans(n_clusters = num_clusters).fit(sample_vectors)
    kmean.predict(sample_vectors)
    labels = kmean.labels_

    lb = [samp + '-' + str(label) for label in labels]
    return pd.DataFrame(sample_vectors), pd.DataFrame(lb)


# In[56]:


deta2 = {}
for key in deta.keys():
    try:
      deta2[key[:-2]] = pd.concat([deta[key[:-2] + 'D' + str(i)] for i in range(0,9)])
    except KeyError:
      deta2[key] = deta[key]

for key in deta2.keys():
  print(key, deta2[key].shape)


# In[86]:


import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Generate a random dataset with positive skewness
data=sample

# Plot the histogram and the estimated probability density function
sns.distplot(data, kde=True, hist=True)

# Add labels and title
plt.xlabel('Data')
plt.ylabel('Probability Density')
plt.title('Skewness Plot')

# Show the plot
plt.show()


# In[109]:


import numpy as np

# Load the CSV file into a numpy array
data = np.loadtxt("C:\\Users\\ankit\\OneDrive\\Desktop\\FCM Data For Kartik\\CHD1_1.fcs.csv", delimiter=",", skiprows=1)

# Calculate the RMSD
rmsd = np.sqrt(np.mean(np.square(data - data.mean(axis=0))))

# Print the RMSD value
print("RMSD:", rmsd)


# In[118]:


pip install flowio


# In[121]:


import flowio
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error

# Set directory containing FCS files
dir = "C:\\Users\\ankit\\OneDrive\\Desktop\\FCM data for Ankit"

# Get list of FCS files in directory
fcs_files = glob.glob(os.path.join(dir, '*.fcs'))

# Loop over FCS files and calculate autocorrelation and RMSD
for f in fcs_files:
    # Load FCS file
    fcs = flowio.FlowData(f)

    # Get data from FCS file
    data = fcs.events

    # Calculate autocorrelation
    corr = np.correlate(data[:,0], data[:,0], mode='full')
    autocorr = corr[len(corr)//2:]
    autocorr = autocorr / autocorr[0]
    lag = np.arange(0, len(autocorr))

    # Calculate RMSD
    actual_values = data[:,0]
    predicted_values = ... # insert your predicted values here
    rmsd = mean_squared_error(actual_values, predicted_values, squared=False)

    # Print results
    print(f'Autocorrelation for file {f}:')
    print(pd.DataFrame({'Lag': lag, 'Autocorrelation': autocorr}))
    print(f'RMSD for file {f}: {rmsd}')


# In[120]:


import flowio
import numpy as np
import glob
import os

dir = "C:\\Users\\ankit\\OneDrive\\Desktop\\FCM data for Ankit"

# Find all the FCS files in the directory
fcs_files = glob.glob(os.path.join(dir, '*.fcs'))

for file in fcs_files:
    # Load the FCS file
    f = flowio.FlowData(file)
    
    # Get the events from the FCS file
    data = f.events
    
    # Convert the data to a NumPy array
    data = np.array(data)
    
    # Calculate autocorrelation
    corr = np.correlate(data[:,0], data[:,0], mode='full')
    autocorr = corr[len(corr)//2:]
    autocorr = autocorr / autocorr[0]
    
    # Calculate RMSD
    rmsd = np.sqrt(np.mean(np.square(data[:,0] - np.mean(data[:,0]))))
    
    # Print the results
    print('File:', file)
    print('Autocorrelation:', autocorr)
    print('RMSD:', rmsd)


# In[122]:


import glob
import os
import flowio
import numpy as np

dir ="C:\\Users\\ankit\\OneDrive\\Desktop\\FCM data for Ankit"
fcs_files = glob.glob(os.path.join(dir, '*.fcs'))

fsc_ssc_values = []

for fcs_file in fcs_files:
    # Read FSC-A and SSC-A channels
    with flowio.FlowData(fcs_file) as f:
        data = f.events
        fsc_ssc = data[:,[0,1]]
        fsc_ssc_values.append(fsc_ssc)

# Calculate RMSD
fsc_ssc_values = np.array(fsc_ssc_values)
rmsd = np.sqrt(np.mean((fsc_ssc_values - np.mean(fsc_ssc_values, axis=0))**2, axis=0))

print('RMSD FSC-A: {:.2f}'.format(rmsd[0]))
print('RMSD SSC-A: {:.2f}'.format(rmsd[1]))


# In[123]:


import glob
import os
import numpy as np
import flowio

dir = "C:\\Users\\ankit\\OneDrive\\Desktop\\FCM data for Ankit"
fcs_files = glob.glob(os.path.join(dir, '*.fcs'))

rmsd_values = []
for fcs_file in fcs_files:
    # Read FSC-A and SSC-A channels
    f = flowio.FlowData(fcs_file)
    data = f.events
    fsc_ssc = data[:,[0,1]]
    
    # Calculate RMSD
    mean = np.mean(fsc_ssc, axis=0)
    diff = fsc_ssc - mean
    rmsd = np.sqrt(np.mean(diff**2))
    rmsd_values.append(rmsd)
    
print(rmsd_values)


# In[ ]:


def plot_sample_data(data, key, day = 1):
    key = key[:3]
    plot = pd.concat([data[key+str(day)+'_'+str(i)] for i in range(1,4)])
    plt.scatter(plot['FSC-A'],plot['FSC-H'], color = 'red')
    plt.xlabel('FSC-A')
    plt.ylabel('SSC-A')
    plt.title(f'Data for {key}, Day: {day}')
    plt.show()
    

[plot_sample_data(data, 'NHD',i) for i in range(0,10)]


# In[ ]:


range_n_clusters = list (range(2,8))
print ("Number of clusters from 2 to 8: \n", range_n_clusters)

cols2 = random.choice(list(data.keys()))
sample = data[cols2][['FSC-A','SSC-A']]
kmean = KMeans(n_clusters = num_clusters).fit(sample)
kmean.predict(sample)
klabels = kmean.labels_
for n_clusters in range_n_clusters:
    clusterer = KMeans (n_clusters=num_clusters).fit(sample)
    preds = clusterer.predict(sample)
    centers = clusterer.cluster_centers_
    

    score = silhouette_score (sample, preds, metric='euclidean')
    print(score)
    #print ("For n_clusters = {}, silhouette score is {})".format(n_clusters, score)
plt.show()    


# In[ ]:


num_clusters = 8
cols2 = random.choice(list(data.keys()))
sample = data[cols2][['FSC-A','SSC-A']]
kmean = KMeans(n_clusters = num_clusters).fit(sample)
kmean.predict(sample)
klabels = kmean.labels_

#filter rows of original data

cluster_count1 = np.zeros(num_clusters)
for cluster in range(num_clusters):
    filter_lb = sample[klabels == cluster]


fig = plt.figure(figsize=(10,10))

ax = fig.add_subplot(111, )
ax.scatter(sample['FSC-A'],sample['SSC-A'],c = klabels) 

plt.show()


# In[ ]:


def plot_sample_data(data, key, day=1):
    key = key[:3]
    plot = pd.concat([data[key+str(day)+'_'+str(i)] for i in range(1,4)])
    sample = data[cols2][['FSC-A','SSC-A']]
    kmean = KMeans(n_clusters=num_clusters).fit(sample)
    kmean.predict(sample)
    klabels = kmean.labels_

    # Filter rows of original data
    cluster_count1 = np.zeros(num_clusters)
    for cluster in range(num_clusters):
        filter_lb = sample[klabels == cluster]

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)
    ax.scatter(sample['FSC-A'], sample['SSC-A'], c=klabels)

    # Save plot as a PNG image file
    fig.savefig('scatter_plot.png')

    plt.close(fig)

