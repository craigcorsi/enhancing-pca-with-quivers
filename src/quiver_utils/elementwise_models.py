import pandas as pd
import numpy as np

from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression


class TopPrincipalComponents(PCA):
    def __init__(
        self, num_components=None, use_threshold=False, threshold=0.95, label_name="z"
    ):
        super(TopPrincipalComponents, self).__init__()
        self.num_components = num_components
        self.use_threshold = use_threshold
        self.threshold = threshold
        self.label_name = label_name

    def fit(self, X):
        super(TopPrincipalComponents, self).fit(X)

    def transform(self, X):
        # Set the number of components to either the provided value, the number of components above the model's
        # explained variance ratio, or the total number of columns
        num_c = self.num_components or len(X.columns)
        if self.use_threshold:
            num_c = self.num_components_above_variance_threshold(
                self.explained_variance_ratio_, threshold=self.threshold
            )

        X_proj = super(TopPrincipalComponents, self).transform(X)[:, :num_c]
        X_columns = [f"{self.label_name}_{i}" for i in range(X_proj.shape[1])]
        X_proj = pd.DataFrame(X_proj, columns=X_columns)
        return X_proj

    def num_components_above_variance_threshold(self, evr, threshold=0.95):
        evr = np.cumsum(evr)
        return int(1 + np.argmax(evr >= threshold))


class NodewisePCA:
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
        return self.reduced_node_structure


class EdgeMapRegression:
    def __init__(self, node_structure, fit_intercept=False):
        self.node_structure = node_structure
        self.fit_intercept = fit_intercept
        self.edge_maps = None

    def fit_transform(self, X, n_splits=5, random_state=None):
        # Determine representation maps by fitting linear maps between each pair of nodes
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
        return self.edge_maps
