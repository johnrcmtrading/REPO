from sklearn import datasets
dataset = datasets.load_iris()


import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np


df = pd.DataFrame(data= np.c_[dataset['data'], dataset['target']],
                  columns= dataset['feature_names'] + ['Species'])


#print(df)
Y = []
Y = df['Species']
HOW_MANY_CLASSES = 3    # WHEN THE BINARY CLASSIFIER CALLS IT, HOW_MANY_CLASSES = 2 
# IN FACT, CLASSIFICATION IS RUN: 0 AGAINST 1, OR 0 AGAINST 2
#===============================================================================
# 'Iris-setosa': 0
# 'Iris-versicolor': 1
# 'Iris-virginica': 2
#===============================================================================


    
df = df.drop(['Species'],axis=1)
preX = df.values.tolist()

X = np.empty((np.shape(preX)[0],np.shape(preX)[1]))


# PREP (NORMALIZATION)
X_max = np.max(preX)
X_min = np.min(preX)
deltazero = -(X_min-0)
for idx, r in enumerate(preX):
    r += deltazero
    r /= X_max+deltazero
    X[idx] = r

#print(np.shape(X))
#print(np.shape(Y))


def main(classnumbers, start_index, max_data):

    storei = 999
    DATASET_1 = []
    LABELS_1 = []
    count = 0
    for i in range(len(Y)):
        if Y[i] == classnumbers[0]:
            if i < start_index:
                continue
            DATASET_1.append(X[i])
            LABELS_1.append(Y[i])
            count += 1
        if count >= max_data:
            break
    print('shape(DATASET_1)',np.shape(DATASET_1),'np.shape(LABELS_1)',np.shape(LABELS_1))
    storei = i


    DATASET_2 = []
    LABELS_2 = []
    count = 0
    for i in range(len(Y)):
        if Y[i] ==  classnumbers[1]:
            if i == storei or i < start_index:
                continue
            DATASET_2.append(X[i])
            LABELS_2.append(Y[i])
            count += 1
        if count >= max_data:
            break
    print('shape(DATASET_2)',np.shape(DATASET_2),'np.shape(LABELS_2)',np.shape(LABELS_2))
    
    
    if len(classnumbers) > 2:
        
        DATASET_3 = []
        LABELS_3 = []
        count = 0
        for i in range(len(Y)):
            if Y[i] ==  classnumbers[2]:
                if i == storei or i < start_index:
                    continue
                DATASET_3.append(X[i])
                LABELS_3.append(Y[i])
                count += 1
            if count >= max_data:
                break
        print('shape(DATASET_3)',np.shape(DATASET_3),'np.shape(LABELS_3)',np.shape(LABELS_3))
    

    
        MERGED_MATRIX = np.append(DATASET_1, DATASET_2, axis=0)
        MERGED_MATRIX = np.append(MERGED_MATRIX, DATASET_3, axis=0)
        MERGED_LABELS = np.append(LABELS_1, LABELS_2, axis=0)
        MERGED_LABELS = np.append(MERGED_LABELS, LABELS_3, axis=0)
        print('shape(MERGED_MATRIX)',np.shape(MERGED_MATRIX))


    
    
    else:
    
        MERGED_MATRIX = np.append(DATASET_1, DATASET_2, axis=0)
        MERGED_LABELS = np.append(LABELS_1, LABELS_2, axis=0)
        print('shape(MERGED_MATRIX)',np.shape(MERGED_MATRIX))





    return MERGED_MATRIX, MERGED_LABELS
