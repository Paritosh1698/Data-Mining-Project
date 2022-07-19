import glob
import time

import numpy as np
import pandas as pd
import math
import os

from sklearn.impute import KNNImputer

#Import the NRMS table
nrms_path = r"C:\Users\Paritosh\Desktop\Spring'22\Data Mining\Group Project\Table-NRMS.xlsx"
NRMS_table = pd.read_excel(nrms_path)
NRMS_array = NRMS_table.to_numpy()

#Path for incomplete dataset
path_incomplete = r"C:\Users\Paritosh\Desktop\Spring'22\Data Mining\Datasets\Incomplete datasets(1)\Data 1"

#Consolidating the incomplete datasets
incomplete_datasets = sorted(glob.glob(os.path.join(path_incomplete, '*.csv')))

#Path for actual dataset
path_actual = r"C:\Users\Paritosh\Desktop\Spring'22\Data Mining\Datasets\Complete datasets(1)\Data_1.csv"

#Loading the complete dataset
actual = pd.read_csv(path_actual)

#Initiating the count value to verify the number of files imputed
count = 0

for f in incomplete_datasets:
    #Mark the starting point to record the execution time
    start = time.time()

    print(f)
    #Loading the incomplete datasets
    incomplete_df = pd.read_csv(f, header=None)

    #Setting the value of k as the squared-root of the number of instances
    k = int(math.sqrt(len(incomplete_df)))

    #Instantiating the knn-imputer
    imputer = KNNImputer(n_neighbors = k)

    #Imputing the missing values
    df2 = imputer.fit_transform(incomplete_df)
    imputed_df = pd.DataFrame(df2)
    #print(imputed_df.head())

    #Checking for any missing values in the imputed dataset
    print('Missing values:', imputed_df.isnull().any().sum())

    #Calculating the NRMS value
    numerator = np.linalg.norm(np.linalg.norm(imputed_df) - np.linalg.norm(actual))
    denomenator = np.linalg.norm(actual)
    nrms = numerator/denomenator
    print('NRMS value:', nrms)

    #Extracting the dataset name from the file-path
    split_f = os.path.split(f)
    dataset = split_f[1].split(".")

    #Matching the dataset name with the elements in the NRMS array
    element = np.where(NRMS_array == dataset[0])
    row = element[0][0]
    column = element[1][0]
    
    #Mark the ending to record the execution time
    end = time.time()
    time_taken = end - start

    #Updating the array with the values of nrms and execution time
    NRMS_array[row][column + 1] = nrms
    NRMS_array[row][column + 2] = time_taken

    count += 1

    print("Time taken:", time_taken)
print("Files Imputed:", count)
pd.DataFrame(NRMS_array).to_excel(r"C:\Users\Paritosh\Desktop\Spring'22\Data Mining\Group Project\Table-NRMS.xlsx", index=False)


