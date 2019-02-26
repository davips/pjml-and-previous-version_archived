'''
Author: Jefferson Tales Oliva
'''

import os

'''
This method aims to separate from the table (DataFrame) the feature matrix (x) and the class target (class_label)

Parameters:
- table (DataFrame): it is a feature dataset, where the lines are instances and columns are attribute-values.

- features (list): it is a list of feature labels. The i-th feature label must correspond to a the i-th table column

- class_label (string): it is the label of an attribute used as class (e.g. class, target, etc)


Return: the matrix feature x and the classes of the instances (matrix line)
'''
def split_features_target(table, features, class_label):
    # get all features
    x = table.loc[:, features].values

    y = table.loc[:, [class_label]].values

    return x, y



'''
Falta terminar a implementação
'''
def arff_to_csv(arff_file):
    if arff_file.endswith(".arff"):
        with open(arff_file, "r") as inFile:
            content = inFile.readlines()
