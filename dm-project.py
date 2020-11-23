import numpy as np
import scipy as sc
import pandas as pd
import matplotlib.pyplot as plt
from pandas_profiling import ProfileReport
from sklearn.preprocessing import MinMaxScaler, StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA

data = pd.read_csv("donors.csv", header=0, sep=',')


################# DATA PREPARATION ##############################
###############FILLING MISSING DATA #########################
data.replace("", np.nan, inplace=True)


## FILL MISSING VALUES
df_central = data.copy()

df_central.isna().sum()

df_central.median()

# Check the object columns = non-metric columns
non_metric_features = df_central.select_dtypes(include=['object']).columns

# Drop the non metric features
metric_features = df_central.columns.drop(non_metric_features).to_list()

modes = df_central[non_metric_features].mode().loc[0]

# Fill NaNs on df_central
df_central.fillna(df_central.median(), inplace=True)
df_central.fillna(modes, inplace=True)
df_central.isna().sum()  # checking how many NaNs we still have

# Check the number of missing values 
df_central.isna().values.sum()


##################################################################
####################### OUTLIERS #################################






##################################################################
####################### DATA NORMALIZATION #######################

data_standard = df_central.copy()

scaler = StandardScaler()
scaled_feat = scaler.fit_transform(data_standard[metric_features])

# See what the fit method is doing (notice the trailing underscore):
print("Parameters fitted:\n", scaler.mean_, "\n", scaler.var_)


data_standard[metric_features] = scaled_feat
data_standard.head()

# Checking mean and variance of standardized variables
data_standard[metric_features].describe().round(2)



##################################################################
####################### ONE-HOT ENCODING #######################
df = data_standard.copy()

data_ohc = df.copy()

####################################################################

# Cast de float para int

columns_float = data_ohc.loc[:, data_ohc.dtypes == np.float64].columns

data_ohc[columns_float] = data_ohc[columns_float].astype(int)

# Cast de Object para string

columns_obj = data_ohc.loc[:, data_ohc.dtypes == np.object].columns

data_ohc[columns_obj] = data_ohc[columns_obj].astype('|S')

# Use OneHotEncoder to encode the categorical features. Get feature names and create a DataFrame 
# with the one-hot encoded categorical features (pass feature names)
ohc = OneHotEncoder(sparse=False, drop="first")
ohc_feat = ohc.fit_transform(data_ohc[non_metric_features])
ohc_feat_names = ohc.get_feature_names()
ohc_df = pd.DataFrame(ohc_feat, index=data_ohc.index, columns=ohc_feat_names)  # Why the index=df_ohc.index?
ohc_df

# Reassigning df to contain ohc variables
df_ohc = pd.concat([data_ohc.drop(columns=non_metric_features), ohc_df], axis=1)

##################################################################
####################### DIMENSIONALITY REDUCTION #######################
df = df_ohc.copy()

df_pca = df.copy()

# Use PCA to reduce dimensionality of data
pca = PCA()
pca_feat = pca.fit_transform(df_pca[metric_features])
pca_feat  # What is this output?




































