'''
Data reduction methods

Author: Jefferson Tales Oliva
'''

import os


'''
This method generates the component labels. It is applied in other methods from this module

Paramenter:

- n_components (integer): number of components


return: labels of components
'''
def generate_component_labels(n_components):
    col = []
    for i in range(0, n_components):
        col.append('principal_component_' + str(i + 1))

    return col


'''
This methods generates a feature table, by concatening reducted features and instace labels, into the DataFrame format.

Params:

- pc: result of a data redution method or a table of real numbers

- instance_labels: labels of instances. The i-th label corresponds to i-th pc line


Return: the reducted feature table into DataFrame format
'''
def generate_data_frame(pc, instance_labels):
    from pandas import DataFrame, concat

    # Generate component labels for pc collumns
    col = generate_component_labels(len(pc[0]))

    # Convert the reducted data into DataFrame format
    pDf = DataFrame(data = pc, columns = col)

    # Generate an instanced feature table and return it
    return concat([pDf, instance_labels], axis = 1)


'''
This method aims to separate from the table (DataFrame) the feature matrix (x) and the class target (class_label)

Parameters:
- table (DataFrame): it is a feature dataset, where the lines are instances and columns are attribute-values.

- features (list): it is a list of feature labels. The i-th feature label must correspond to a the i-th table column

- class_label (string): it is the label of an attribute used as class (e.g. class, target, etc)


Return: the matrix feature x and the classes of the instances (matrix line)
'''
def split_features_target(table, features, class_label):
    from numpy import isnan, isinf

    # get all features
    x = table.loc[:, features].values

    # replace nan and infinite values. These operations avoid some error processing
    x[isnan(x)] = 0
    x[isinf(abs(x))] = 0

    y = table.loc[:, [class_label]].values

    return x, y


'''
Falta Implementar métodos para converter csv, arrf, etc para DataFrame
'''
def arff_to_csv(arff_file):
    if arff_file.endswith(".arff"):
        with open(arff_file, "r") as inFile:
            content = inFile.readlines()
