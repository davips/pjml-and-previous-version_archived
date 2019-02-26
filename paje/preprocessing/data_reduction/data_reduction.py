'''
Data reduction methods

Author: Jefferson Tales Oliva
'''


'''
This method generates the component labels. It is applied in other methods from this module

Paramenter:

- n_components (integer): number of components


return: labels of components
'''
def genete_component_labels(n_components):
    col = []
    for i in range(0, n_components):
        col.append('principal_component_' + str(i + 1))

    return col


'''
This method is a PCA (principal component analysis) implementation. Its params is presented bellow:

- table (DataFrame): it is a feature dataset, where the lines are instances and columns are attribute-values.

- features (list): it is a list of feature labels. The i-th feature label must correspond to a the i-th table column

- class_label (string): it is the label of an attribute used as class (e.g. class, target, etc)

- n_components (integer): number of features for the redimensioned table

return: redimensioned table of features
'''
def apply_PCA(table, features, class_label, n_components):
    from numpy import isnan, isinf
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    from pandas import DataFrame, concat
    from paje import feature_file_processor

    #x = table.loc[:, features].values
    #y = table.loc[:, [class_label]].values

    # get all features and class instances
    x, y = feature_file_processor.split_features_target(table, features, class_label)

    # replace nan and infinite values
    x[isnan(x)] = 0
    x[isinf(abs(x))] = 0

    # standardize features
    x = StandardScaler().fit_transform(x)

    # generate pca columns
    col = genete_component_labels(n_components)

    # apply PCA
    pca = PCA(n_components = n_components)
    pc = pca.fit_transform(x)
    pDf = DataFrame(data = pc, columns = col)

    return concat([pDf, table[[class_label]]], axis = 1)


'''
Factor analysis implementation

Paramenters:

- table (DataFrame): it is a feature dataset, where the lines are instances and columns are attribute-values.

- features (list): it is a list of feature labels. The i-th feature label must correspond to a the i-th table column

- class_label (string): it is the label of an attribute used as class (e.g. class, target, etc)

- n_components (integer): number of features for the redimensioned table

return: redimensioned table of features

'''
def apply_factor_analysis(table, features, class_label, n_comp):
    from sklearn.decomposition import FactorAnalysis
    from pandas import DataFrame, concat
    from paje import feature_file_processor

    # get all features and class instances
    x, y = feature_file_processor.split_features_target(table, features, class_label)

    # generate factor columns
    col = genete_component_labels(n_comp)

    factor = FactorAnalysis(n_components = n_comp, random_state=0).fit_transform(x)

    pDf = DataFrame(data = factor, columns = col)

    return concat([pDf, table[[class_label]]], axis = 1)
