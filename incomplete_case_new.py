import numpy as np
import pandas as pd
import math

from sklearn.impute import KNNImputer

#Path for incomplete dataset
path_incomplete = r"C:\Users\Paritosh\Desktop\Spring'22\Data Mining\Datasets\Incomplete datasets(1)\Data 8\Data_8_AE_1%.csv"

#Path for actual dataset
path_actual = r"C:\Users\Paritosh\Desktop\Spring'22\Data Mining\Datasets\Complete datasets(1)\Data_8.csv"

#Loading both the datasets
df = pd.read_csv(path_incomplete, header=None)
actual = pd.read_csv(path_actual, header=None)

#Setting the value of k as the squared-root of the number of instances
k = int(math.sqrt(len(df)))

#Instantiating the knn-imputer
imputer = KNNImputer(n_neighbors = k)

#Imputing the missing values
df2 = imputer.fit_transform(df)
imputed_df = pd.DataFrame(df2)
print(imputed_df.head())

#Checking for any missing values in the imputed dataset
print('Missing values:',imputed_df.isnull().any().sum())

