'''
Data reduction methods

Author: Jefferson Tales Oliva
'''


'''
This method is a PCA (principal component analysis) implementation. Its parameters is presented bellow:

- table (DataFrame): it is a feature dataset, where the lines are instances and columns are attribute-values.

- features (list): it is a list of feature labels. The i-th feature label must correspond to a the i-th table column

- class_label (string): it is the label of an attribute used as class (e.g. class, target, etc)

- n_components (integer): number of features for the redimensioned table

return: redimensioned table of features
'''
def apply_PCA(table, features, class_label, n_components):
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    from pandas import DataFrame, concat
    from paje import feature_file_processor

    # get all features and class instances
    x, y = feature_file_processor.split_features_target(table, features, class_label)

    # standardize features
    x = StandardScaler().fit_transform(x)

    # apply PCA
    pca = PCA(n_components = n_components)
    pc = pca.fit_transform(x)

    # generate a feature table and return it
    return feature_file_processor.generate_data_frame(pc, table[[class_label]])


'''
Factor analysis implementation

Parameters:

- table (DataFrame): it is a feature dataset, where the lines are instances and columns are attribute-values.

- features (list): it is a list of feature labels. The i-th feature label must correspond to a the i-th table column

- class_label (string): it is the label of an attribute used as class (e.g. class, target, etc)

- n_components (integer): number of features for the redimensioned table

return: redimensioned table of features

'''
def apply_factor_analysis(table, features, class_label, n_components):
    from sklearn.decomposition import FactorAnalysis
    from pandas import DataFrame, concat
    from paje import feature_file_processor

    # get all features and class instances
    x, y = feature_file_processor.split_features_target(table, features, class_label)

    pc = FactorAnalysis(n_components = n_components, random_state=0).fit_transform(x)

    # generate a feature table and return it
    return feature_file_processor.generate_data_frame(pc, table[[class_label]])


'''
Singular value decomposition

- table (DataFrame): it is a feature dataset, where the lines are instances and columns are attribute-values.

- features (list): it is a list of feature labels. The i-th feature label must correspond to a the i-th table column

- class_label (string): it is the label of an attribute used as class (e.g. class, target, etc)

- n_components (integer): number of features for the redimensioned table

return: redimensioned table of features
'''

def apply_SVD(table, features, class_label, n_components):
    from pandas import DataFrame, concat
    from numpy.linalg import eig
    from numpy import diag
    from scipy.sparse.linalg import svds
    from paje import feature_file_processor

    # get all features and class instances
    x, y = feature_file_processor.split_features_target(table, features, class_label)

    #A = USV^T. For data reduction, the trasformation only must be US
    # U is a m x m orthonormal matrix of 'left-singular' (eigen)vectors of  xx^T
    #U = eig(x @ x.T)

    # V is a n x n orthonormal matrix of 'right-singular' (eigen)vectors of  x^Tx
    #lmbV, _ = eig(x.T @ x)

    # S is a m x n diagonal matrix of the square root of nonzero eigenvalues of U or V
    #S = sqrt(diag(abs(lmbV))[:n_components,:])

    # apply SVD
    u, s, _ = svds(x, n_components)
    pc = u @ diag(s) # If we use V^T in this operation, the pc will have the original dimension

    # generate a feature table and return it
    return feature_file_processor.generate_data_frame(pc, table[[class_label]])
