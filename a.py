"""
*This module was create for Data Mining subject in Universidad Autonóma de Chihuahua
*Professor: M.I.C Normando Ali Zubia Hernández

Module information:
The principal functions of this module are:
*Create violin graphs
*Create box-Graphs
*Create Histograms

Information contact:
email: azubiah@uach.mx

"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pandas.tools.plotting import scatter_matrix

def open_file(fileName):
    '''
    This method will open a file in csv format
    :param fileName: file to open (Complete PATH)
    :return: Pandas Data Frame
    '''
    #TODO csv file validation

    data = pd.read_json(fileName)

    return data

def create_histogram(data):
    data.hist(column = 'bedrooms')

    plt.show()

def create_density_plot(data):
    data.plot(kind='density', subplots=True, layout=(3, 3), sharex=False)
    plt.show()

def create_whisker_plots(data):
    data.plot(kind='box', subplots=True, layout=(3, 3), sharex=False, sharey=False)
    plt.show()

def show_data_info(data):
    print("Number of instance: " + str(data.shape[0]))
    print("Number of fetures: " + str(data.shape[1]))

    print('------------------------------------------')

    print("Initial instances:\n")
    print(data.head(10))

    print("Numerical Information:\n")
    numerical_info = data.iloc[:, :data.shape[1]]
    print(numerical_info.describe())

def get_feature_subset(data, *args):
    featureDict = []
    for arg in args:
        featureDict.append(arg)

    subset = data[featureDict]

    return subset

def delete_column(data, *args):
    for arg in args:
        data = data.drop(arg, 1)

    return data

def delete_missing_objects(data, type):
    type = 0 if type == 'instance' else 1

    data = data.dropna(axis = type)

    return data

def replace_missing_values_with_constant(data, column, constant):
    temp = data[column].fillna(constant)
    data[column] = temp

    return data

def replace_missing_values_with_mean(data, column):
    temp = data[column].fillna(data[column].mean())
    data[column] = temp

    return data

def numero_banios_influye_precio(data):

    numbBath = data['bathrooms'].value_counts()
    numbBathKeys = numbBath.keys()

    priceArray = []
    for number in numbBathKeys:
        subset = data.loc[data['bathrooms'] == number]
        print('Numero de banios:' + str(number))
        print(subset['price'])
        priceArray.append(subset["price"].mean())

    print(numbBathKeys)
    print(priceArray)

    width = .2
    plt.bar(numbBathKeys, priceArray, width, color="blue")

    plt.ylabel('precio')
    plt.xlabel('#banios')
    plt.title('banios inlfuye precio')
    plt.xticks(np.arange(0, max(numbBathKeys), .5))
    plt.yticks(np.arange(0, 60000, 5000))


    plt.show()

def numero_habitaciones_influye_precio(data):

    numbHab = data['bedrooms'].value_counts()
    numbHabKeys = numbHab.keys()

    priceArray = []
    for number in numbHabKeys:
        subset = data.loc[data['bedrooms'] == number]
        print('Numero de habitaciones:' + str(number))
        print(subset['price'])
        priceArray.append(subset["price"].mean())

    print(numbHabKeys)
    print(priceArray)

    width = .2
    plt.bar(numbHabKeys, priceArray, width, color="blue")

    plt.ylabel('precio')
    plt.xlabel('#habitaciones')
    plt.title('Habitaciones influye precio')
    plt.xticks(np.arange(0, max(numbHabKeys), .5))
    plt.yticks(np.arange(0, 15000, 1000))


    plt.show()

if __name__ == '__main__':
    filePath = "train.json"

    data = open_file(filePath)



    #headers = [x for x in data]
    #print(headers)
    #for head in headers:
    #    if head != 'description' and head != 'features' and head != 'photos':
    #        print(data[head].value_counts())
    #print(data.head)
    #show_data_info(data)
    #print(data[0:10])

    #numero_banios_influye_precio(data)
    numero_habitaciones_influye_precio(data)

    #create_histogram(data)
    #create_density_plot(data)
    #create_whisker_plots(data)
