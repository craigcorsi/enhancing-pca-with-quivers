import pandas as pd
import numpy as np

from sklearn.decomposition import PCA




def num_components_above_variance_threshold(evr, threshold=0.95):
    evr = np.cumsum(evr)
    return int(1 + np.argmax(evr >= threshold))

# Wrapper for PCA on any data train/test set
def top_principal_components(train, test, name='w', num_components=None, use_threshold=False, threshold=0.95):
    dim = len(train.columns)
    if not num_components:
        num_components = dim
    
    if dim == 1:
        train_PCA = train.values
        test_PCA = test.values
        columns = [f"{name}_0"]
    else:
        PCA_model = PCA()
        PCA_model.fit(train)

        if use_threshold:
            num_components = num_components_above_variance_threshold(
                PCA_model.explained_variance_ratio_, 
                threshold
            )
        
        train_PCA = PCA_model.transform(train)[:,:num_components]
        test_PCA = PCA_model.transform(test)[:,:num_components]
        columns = [f"{name}_{i}" for i in range(train_PCA.shape[1])]

    train_PCA = pd.DataFrame(train_PCA, columns=columns)
    test_PCA = pd.DataFrame(test_PCA, columns=columns)

    return train_PCA, test_PCA



    