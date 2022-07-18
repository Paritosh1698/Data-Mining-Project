import glob
import numpy as np
import pandas as pd
import math
import os

from sklearn.impute import KNNImputer

#Path for incomplete dataset
path_incomplete = r"C:\Users\Paritosh\Desktop\Spring'22\Data Mining\Datasets\Incomplete datasets(1)\Data 1"

#Consolidating the incomplete datasets
incomplete_datasets = glob.glob(os.path.join(path_incomplete, '*.csv'))

#Path for actual dataset
path_actual = r"C:\Users\Paritosh\Desktop\Spring'22\Data Mining\Datasets\Complete datasets(1)\Data_1.csv"

#Loading the complete dataset
actual = pd.read_csv(path_actual)

count = 0

for f in incomplete_datasets:
    #Loading the incomplete datasets
    df = pd.read_csv(f, header=None)

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
    
    #Calculating the NRMS value
    numerator = np.linalg.norm(np.linalg.norm(imputed_df) - np.linalg.norm(actual))
    denomenator = np.linalg.norm(actual)
    nrms = numerator/denomenator
    print('NRMS value', nrms)

    count += 1
print(count)




