#!/usr/bin/env python
# coding: utf-8

# In[2]:


from sklearn.datasets import load_iris
import numpy as np
DATA= load_iris()
x=DATA.data[:,:2]
minimum=np.amin(x)
maximum=np.amax(x)
np.amin(x,axis=1)
np.average(x)
w=[1,2]
average=np.average(x,weights=w,axis=1)
mean=np.mean(x)
np.mean(x,axis=0)
median=np.median(x)
standard_div=np.std(x)
sum_=np.sum(x,axis=0)
variance=np.var(x)
quantile=np.quantile(x,0.75)
print(minimum,maximum,sum_,median,mean,standard_div,variance,quantile)


# In[27]:


from sklearn.datasets import load_iris
DATA= load_iris()
feature_name=DATA.feature_names
target_data=DATA.target
target_name=DATA.target_names
print(feature_name,target_data,target_name)
print(DATA)


# In[29]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.datasets import load_iris
import pandas as pd
DATA= load_iris()
X_train=DATA.data[:,:2]
data=X_train
scaler = MinMaxScaler()

# Initialize the MinMaxScaler


# Fit and transform the data using the scaler
normalized_data = scaler.fit_transform(data)

# Original data scatter plot
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.scatter(data[:, 0], data[:, 1], c='blue', label='Original Data')
plt.title('Original Data')
plt.xlabel('original Feature 1')
plt.ylabel('original Feature 2')
plt.legend()

# Normalized data scatter plot
plt.subplot(1, 2, 2)
plt.scatter(normalized_data[:, 0], normalized_data[:, 1], c='purple', label='Normalized Data')
plt.title('Min_max_Normalized Data')
plt.xlabel('Min_max_Normalized Feature 1')
plt.ylabel('Min_max_Normalized Feature 2')
plt.legend()

plt.tight_layout()
plt.show(),normalized_data


# In[30]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
DATA= load_iris()
X_train=DATA.data[:,:2]
data=X_train
scaler= StandardScaler()

# Initialize the Standard scaler
scaler = StandardScaler()

# Fit and transform the data using the scaler
normalized_data = scaler.fit_transform(data)

# Original data scatter plot
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.scatter(data[:, 0], data[:, 1], c='blue', label='Original Data')
plt.title('Original Data')
plt.xlabel('original Feature 1')
plt.ylabel('original Feature 2')
plt.legend()

# Normalized data scatter plot
plt.subplot(1, 2, 2)
plt.scatter(normalized_data[:, 0], normalized_data[:, 1], c='red', label='Normalized Data')
plt.title('standard_scaler_Normalized Data')
plt.xlabel('standard_scaler_Normalized Feature 1')
plt.ylabel('standard_scaler_Normalized Feature 2')
plt.legend()

plt.tight_layout()
plt.show(),normalized_data


# In[31]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import RobustScaler
DATA= load_iris()
X_train=DATA.data[:,:2]
data=X_train
scaler= RobustScaler()


# Initialize the Robust scaler
scaler = StandardScaler()

# Fit and transform the data using the scaler
normalized_data = scaler.fit_transform(data)

# Original data scatter plot
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.scatter(data[:, 0], data[:, 1], c='blue', label='Original Data')
plt.title('Original Data')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()

# Normalized data scatter plot
plt.subplot(1, 2, 2)
plt.scatter(normalized_data[:, 0], normalized_data[:, 1], c='brown', label='Normalized Data')
plt.title('Robust_scaled_Normalized Data')
plt.xlabel('Robust_scaled_Normalized Feature 1')
plt.ylabel('Robust_scaled_Normalized Feature 2')
plt.legend()

plt.tight_layout()
plt.show(),normalized_data


# In[32]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PowerTransformer

# Sample data (replace this with your actual data)
DATA= load_iris()
X_train=DATA.data[:,:2]
data=X_train

# Initialize the PowerTransformer
scaler = PowerTransformer()

# Fit and transform the data using the scaler
transformed_data = scaler.fit_transform(data)

# Original data scatter plot
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.scatter(data[:, 0], data[:, 1], c='blue', label='Original Data')
plt.title('Original Data')
plt.xlabel('original Feature 1')
plt.ylabel('original Feature 2')
plt.legend()

# Transformed data scatter plot
plt.subplot(1, 2, 2)
plt.scatter(transformed_data[:, 0], transformed_data[:, 1], c='green', label='Transformed Data')
plt.title('Transform normalized Data')
plt.xlabel('Transform normalized Feature 1')
plt.ylabel('Transform normalized Feature 2')
plt.legend()

plt.tight_layout()
plt.show(),transformed_data


# In[ ]:




