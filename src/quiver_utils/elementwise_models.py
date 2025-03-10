import pandas as pd
import numpy as np

from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression


class TopPrincipalComponents(PCA):
    """
    Subclass of sklearn.decomposition.PCA for specifying either the number of principal components or an explained
        variance threshold.

    Parameters:
        num_components (int): Specifies the number of principal components if use_threshold is set to False
        use_threshold (bool): Indicates whether the number of principal components is determined by an explained
            variance ratio threshold
        threshold (float): If use_threshold is set to True, then self.transform returns the fewest number of
            principal components whose explained variance ratio exceeds this value.
        label_name (str): Column name for the ith principal component, e.g. "z_0"
    """

    def __init__(
        self, num_components=None, use_threshold=False, threshold=0.95, label_name="z"
    ):
        super(TopPrincipalComponents, self).__init__()
        self.num_components = num_components
        self.use_threshold = use_threshold
        self.threshold = threshold
        self.label_name = label_name

    def fit(self, X):
        """
        Fits PCA.

        Parameters:
            X: Pandas DataFrame with input data

        Returns:
            None
        """
        super(TopPrincipalComponents, self).fit(X)

    def transform(self, X):
        """
        Transforms the data and returns the specified number of principal components.

        Parameters:
            X: Pandas DataFrame with input data

        Returns:
            The top principal components
        """
        num_c = self.num_components or len(X.columns)
        # Determine the number of principal components if using the explained variance ratio threhold
        if self.use_threshold:
            num_c = self.num_components_above_variance_threshold(
                self.explained_variance_ratio_, threshold=self.threshold
            )

        X_proj = super(TopPrincipalComponents, self).transform(X)[:, :num_c]
        X_columns = [f"{self.label_name}_{i}" for i in range(X_proj.shape[1])]
        X_proj = pd.DataFrame(X_proj, columns=X_columns)
        return X_proj

    def num_components_above_variance_threshold(self, evr, threshold=0.95):
        """
        Determines the minimum number of principal components that comprise a specified explained variance
            ratio threshold.

        Parameters:
            evr: NumPy array of explained variance ratio values
            threshold: The explained variance ratio threshold

        Returns:
            The minimum integer i such that the sum of the first i entries exceeds the threshold
        """
        evr = np.cumsum(evr)
        return int(1 + np.argmax(evr >= threshold))


class NodewisePCA:
    """
    Custom PCA implementation for datasets whose features are partitioned into subsets that correspond with the nodes
        of a NetworkX DiGraph. Determines the top principal components for the data features at each node separately.

    Parameters:
        node_structure (dict): Each node is associated with a list of column names arising from a Pandas DataFrame
        num_components (int): Specifies a fixed number of principal components to take (whenever this number is less than
            the dimension of the feature space)
        use_threshold (bool): When this is set to True, the number of principal components selected is a proportion of
            explained variance ratio specified by the value of threshold
        threshold (float): The threshold of explained variance ratio. This is only used when use_threshold is set to True
    """

    def __init__(
        self, node_structure, num_components=None, use_threshold=True, threshold=0.95
    ):
        self.node_structure = node_structure
        self.node_list = list(self.node_structure.keys())
        self.nodewise_models = {k: None for k in node_structure.keys()}
        self.reduced_node_structure = {k: None for k in node_structure.keys()}

        self.num_components = num_components
        self.use_threshold = use_threshold
        self.threshold = threshold

    def fit(self, X):
        """
        Fits PCA at each node.

        Parameters:
            X: Pandas DataFrame whose columns must include the column names contained in self.node_structure

        Returns:
            None
        """

        # Fit PCA using TopPrincipalComponents at each node
        for node in self.node_list:
            features_in_node = self.node_structure[node]
            X_at_node = X[features_in_node]
            model_at_node = TopPrincipalComponents(
                num_components=self.num_components,
                use_threshold=self.use_threshold,
                threshold=self.threshold,
                label_name=node,
            )
            model_at_node.fit(X_at_node)
            self.nodewise_models[node] = model_at_node

    def transform(self, X):
        """
        Projects the data in X onto each node's top principal components.

        Parameters:
            X: Pandas DataFrame whose columns must include the column names contained in self.node_structure

        Returns:
            A new Pandas DataFrame with the transformed data
        """
        X_nodewise_PCA = []
        for node in self.node_list:
            features_in_node = self.node_structure[node]
            X_at_node = X[features_in_node]
            model_at_node = self.nodewise_models[node]
            X_at_node_reduced = model_at_node.transform(X_at_node)
            X_nodewise_PCA.append(X_at_node_reduced)
            self.reduced_node_structure[node] = list(X_at_node_reduced.columns)
        return pd.concat(X_nodewise_PCA, axis=1)

    def get_reduced_node_structure(self):
        """
        Retrieves the node structure of the PCA-transformed data.
        """
        return self.reduced_node_structure


class EdgeMapRegression:
    """
    LinearRegression implementation for datasets whose features are partitioned into subsets that are to correspond with
    the nodes of a NetworkX DiGraph. Infers a linear map between each distinct ordered pair of nodes.

    Parameters:
        node_structure (dict): Each node is associated with a list of column names arising from a Pandas DataFrame
        fit_intercept (bool): Set to False by default. When set to False, the intercept of the linear regression is
            constrained to be the zero vector
    """

    def __init__(self, node_structure, fit_intercept=False):
        self.node_structure = node_structure
        self.fit_intercept = fit_intercept
        self.edge_maps = None

    def fit_transform(self, X, n_splits=5, random_state=None):
        """
        Fits the linear maps with k-fold cross-validation and stores the maps in a dict-of-dicts.

        Parameters:
            X: Pandas DataFrame with the data to be transformed
            n_splits (int): The number of splits for k-fold validation
            random_state: Optional random state for k-fold validation

        Returns:
            A dict-of-dicts indexing edge maps by initial node, followed by final node
        """
        node_labels = list(self.node_structure.keys())
        all_edge_maps = {node: {} for node in node_labels}
        kfold = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        for head in node_labels:
            tails = (node for node in node_labels if node != head)
            for tail in tails:
                edge_maps = [[] for a in range(n_splits)]
                X_1 = X[self.node_structure[head]]
                X_2 = X[self.node_structure[tail]]
                kfold.split(X_1)

                for j, (train_index, test_index) in enumerate(kfold.split(X)):
                    X_1_train_train = X_1.iloc[train_index, :]
                    X_2_train_train = X_2.iloc[train_index]
                    X_1_holdout = X_1.iloc[test_index, :].to_numpy()
                    X_2_holdout = X_2.iloc[test_index].to_numpy()

                    for i in range(len(list(X_2.columns))):
                        y = X_2_train_train.iloc[:, i]
                        model = LinearRegression(fit_intercept=self.fit_intercept)
                        model.fit(X_1_train_train, y)
                        edge_maps[j].append(model.coef_)
                    edge_maps[j] = np.array(edge_maps[j])

                mean_edge_map = np.array(edge_maps).mean(axis=0)
                all_edge_maps[head][tail] = mean_edge_map
        self.edge_maps = all_edge_maps
        return self.edge_maps

    def get_edge_maps(self):
        """
        Retrieves the edge maps.
        """
        return self.edge_maps
