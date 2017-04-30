"""
Author: Normando Ali Zubia Hernández
This file is created to explain the use of dimensionality reduction
with different tools in sklearn library.
Every function contained in this file belongs to a different tool.
"""
from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn import metrics

import pandas as pd
import numpy

def get_feacture_subset(data, *args):
    featureDic = []
    for arg in args:
        featureDic.append(arg)

    subset = data[featureDic]
    return subset

def attribute_subset_selection_with_trees(data):
    # import data
    X = data[:,1:-1]
    Y = numpy.asarray(data[:,-1], dtype="int16")

    # First 10 rows
    print('Training Data:\n\n' + str(X[:20]))
    print('\n')
    print('Targets:\n\n' + str(Y[:20]))

    # Model declaration
    extra_tree = ExtraTreesClassifier()

    # Model training
    extra_tree.fit(X, Y)

    # Model information:
    print('\nModel information:\n')

    # display the relative importance of each attribute
    print('Importance of every feature:\n' + str(extra_tree.feature_importances_))

    # If model was training before prefit = True
    model = SelectFromModel(extra_tree, prefit = True)

    # Model transformation
    new_feature_vector = model.transform(X)

    # First 10 rows of new feature vector
    print('\nNew feature vector:\n')
    print(new_feature_vector[:10])

def principal_components_analysis(data, columns, n_components):
    # import data
    X = data[:,1:-1]
    Y = numpy.asarray(data[:,-1], dtype="int16")

    # First 10 rows
    print('Training Data:\n\n' + str(X[:10]))
    print('\n')
    print('Targets:\n\n' + str(Y[:10]))

    # Model declaration
    if n_components < 1:
        pca = PCA(n_components = n_components, svd_solver = 'full')
    else:
        pca = PCA(n_components = n_components)

    # Model training
    pca.fit(X)

    # Model transformation
    new_feature_vector = pca.transform(X)

    # Model information:
    print('\nModel information:\n')
    print('Number of components elected: ' + str(pca.n_components))
    print('New feature dimension: ' + str(pca.n_components_))
    print('Variance of every feature:   \n' + str(pca.explained_variance_ratio_))

    # First 10 rows of new feature vector
    #print('\nNew feature vector:\n')
    #print(new_feature_vector[:10])

    #print(pd.DataFrame(pca.components_,columns=columns[1:-1]))

    # Print complete dictionary
    # print(pca.__dict__)

def z_score_normalization(data):
    print('----- z_score_normalization -------\n')
    # import data
    X = data[:,1:-1]
    Y = numpy.asarray(data[:,-1], dtype="int16")

    # First 10 rows
    print('Training Data:\n\n' + str(X[:10]))
    print('\n')
    print('Targets:\n\n' + str(Y[:10]))

    # Data standarization
    standardized_data = preprocessing.scale(X)

    # First 10 rows of new feature vector
    print('\nNew feature vector:\n')
    print(standardized_data[:10])

    return standardized_data

def min_max_scaler(data):
    print('----- min_max_scaler -------\n')
    # import data
    X = data[:,1:-1]
    Y = numpy.asarray(data[:,-1], dtype="int16")

    # First 10 rows
    print('Training Data:\n\n' + str(X[:10]))
    print('\n')
    print('Targets:\n\n' + str(Y[:10]))

    # Data normalization
    min_max_scaler = preprocessing.MinMaxScaler()

    min_max_scaler.fit(X)

    # Model information:
    print('\nModel information:\n')
    print('Data min: \n' + str(min_max_scaler.data_min_))
    print('Data max: \n' + str(min_max_scaler.data_max_))

    new_feature_vector = min_max_scaler.transform(X)

    # First 10 rows of new feature vector
    print('\nNew feature vector:\n')
    print(new_feature_vector[:10])

def fill_missing_values_with_constant(data, column, constant):
    temp = data[column].fillna(constant)
    data[column] = temp
    return data

def fill_missing_values_with_mean(data, column):
    temp = data[column].fillna(data[column].mean())
    data[column] = temp
    return data

def fill_missing_values_with_mode(data, column):
    temp = data[column].fillna(data[column].mode()[0])
    data[column] = temp
    return data

def convert_data_to_numeric(data):
    numpy_data = data.values
    for i in range(len(numpy_data[0])):
        temp = numpy_data[:,i]
        if(type(temp[0]).__name__  == 'str'):
            dict = numpy.unique(numpy_data[:,i])
            # print(dict)
            for j in range(len(dict)):
                # print(numpy.where(numpy_data[:,i] == dict[j]))
                temp[numpy.where(numpy_data[:,i] == dict[j])] = j
            numpy_data[:,i] = temp
    return numpy_data

def save(data):
    data.to_csv('clean_dataset.csv', index = False)

if __name__ == '__main__':

    data = pd.read_csv('train.csv')
    data['LotFrontage'] = data['LotFrontage'].replace('NaN', -1, regex=False)

    #Outlier
    data = fill_missing_values_with_constant(data, 'MasVnrArea', 0)

    #Outlier
    data = fill_missing_values_with_constant(data, 'GarageYrBlt', -1)

    data = data.fillna('NaN')

    columns = data.columns
    #print(columns)

    data = convert_data_to_numeric(data)

    data = z_score_normalization(data)
    #min_max_scaler(data)

    #attribute_subset_selection_with_trees(data)
    principal_components_analysis(data,columns,.80)

    feature_vector = data[:,1:-1]
    targets = data[:,-1]

    data_features_train, data_features_test, data_targets_train, data_targets_test = \
        train_test_split(feature_vector,
                         targets,
                         test_size=0.25)

    # Model declaration
    """
    Parameters to select:
    criterion: "mse"
    max_depth: maximum depth of tree, default: None
    """
    dec_tree_reg = DecisionTreeRegressor(criterion='mse', max_depth=7)
    dec_tree_reg.fit(data_features_train, data_targets_train)

    # Model evaluation
    test_data_predicted = dec_tree_reg.predict(data_features_test)

    error = metrics.mean_absolute_error(data_targets_test, test_data_predicted)

    print('Total Error: ' + str(error))
