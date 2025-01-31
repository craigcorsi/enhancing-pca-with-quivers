# Enhancing PCA with Quivers
Author: Craig Corsi

## Introduction
In this project, we develop a data transformation process that can be combined with any machine learning model to enhance explainability while (in some cases) improving model performance. We refer to this process as Quiver Principal Component Analysis (**Quiver PCA**) as it uses quivers (directed graphs) to engineer new features by projecting the data onto combinatorially-defined subspaces. Our analysis of Quiver PCA on a dataset of polarimetric measurements at agricultural crop sites shows potential usefulness for analyzing IoT and other technology data. More generally, it is applicable to any dataset with many numerical features that are grouped together as multivariate features.

## Background
Representation theory is the study of understanding abstract mathematical structures by describing them more concretely in the language of linear algebra. In the context of machine learning, representation theory can also impose an abstract structure on a dataset to identify important relationships in the data. For example, quantum geometric machine learning uses group representation theory to identify symmetries, which can reduce the dimensionality of data and/or improve the efficiency of model training.

Of particular interest to us is a recent paper from Seigal, Harrington, and Nanda [2], in which a numerical dataset has a quiver representation associated with it. A **quiver** is another name for a directed graph, used in the context of representation theory. A **quiver representation** has the following structure:
1. Abstractly, a quiver consists of a collection of nodes (or vertices) and a collection of directed edges (or arrows), with each edge pointing from one node (its tail node) to another (its head node).
2. The nodes are associated with disjoint subsets of the dataset’s features. The features at each node are considered combined as one multivariate feature. 
3. There is a matrix associated with each edge that models a linear relationship between features at two different nodes.

(We do not allow for quivers to have loops or multiple edges, and in practice the quivers will be acyclic.) At each node, we may define a vector space, which is the set of all possible values of the feature at that node. Then the matrix associated with an edge defines a linear map between vector spaces, and we can ask how closely it maps one component of a data point to another.

The **sections** of a quiver representation are vectors in which the product of each edge’s matrix with the vector’s component at the tail of the edge is equal to the component at the head of the edge. The **space of sections** is the vector space of all sections, and using an algorithm in [2], we can compute a basis for this space. Thus, if the data is closely approximated by a quiver representation, then we can project each data point onto a nearby point in the space of sections. This drastically reduces the dimensionality of the data while still encoding important relationships between features.

Quiver PCA infers multiple quiver representations from a dataset. With each quiver representation, it computes the space of sections and projects the data onto it, creating a new multivariate feature that has an inherent combinatorial interpretation. By aggregating features from multiple quivers, Quiver PCA transforms the data, identifying many complex relationships without increasing computational complexity during model training. In the analysis of feature importance, one can determine which quivers are most relevant to a model’s performance and summarize the data in an intuitive, visual way.

## Stakeholders
- Businesses managing IoT devices, sensors and other technology
- Product research teams investigating causal relationships in data
- General data science audience

## KPIs
- Improvement of model performance
- Explainability of model results
- Compatibility with a variety of machine learning models

## Data
We analyzed the current instance of Quiver PCA using the dataset ‘Crop mapping using fused optical-radar data set’ from UC Irvine’s Machine Learning Repository [1]. This dataset consists of polarimetric and optical measurements of crop sites as well as class labels with 7 different classes.

We chose a selection of the data sample and of the feature set based on properties that would be useful for the evaluation of Quiver PCA. 
We restricted our analysis to a subset of crop sites based on the distribution of the ‘sigHH’ parameter. On this subset, many polarimetric features had a distribution somewhat close to Gaussian, which we wanted for the sake of linear regression analysis between pairs of features.
We chose several features that followed an approximate Gaussian distribution and grouped these together based on their feature label prefix. For example, {‘sigHH’, ‘sigHV’, and ‘sigVV’} was chosen and considered as one multivariate feature. A total of 11 features were chosen this way, and we only considered polarimetric features.
The data consisted of the same set of crop sites with samples taken twice, 9 days apart. We used the first sample as a training set to generate quiver representations, and we used the second sample as an evaluation set. Note that what we are evaluating is not the machine learning models themselves, but rather Quiver PCA compared to standard PCA in how it improves classification accuracy. Thus, the evaluation set is not a test set in the usual sense.
Our data selection had unequal classification label frequencies, with class sizes ranging from 229 to 17105, and this may have affected model accuracy. 

Data cleaning consisted only of relabeling columns and splitting the data into training and evaluation sets. The data did not have any missing values.

## Quiver PCA
To construct quiver representations from the data, we needed to determine not only which pairs of features had the strongest relationships, but also which linear maps best described the relationships. To do this, we first modeled all possible linear maps between ordered pairs of features using linear regression. Then, we designed an algorithm that constructs a quiver based on how well the linear maps describe the data together. The algorithm can generate several quivers depending on the choice of an initial reference node. The quivers are always acyclic graphs with no loops or multiple edges, and as of now the quivers are in fact directed trees. We note that while our algorithm is similar to a causal discovery algorithm used for structure learning in the field of causal inference, it is not quite the same paradigm.

The data transformation process is as follows:
1. Scale the data to have mean 0 and standard deviation 1 for each univariate feature. 
2. Reduce dimension using standard PCA at each node, to eliminate collinearity between grouped features.
3. Perform linear regression between each pair of nodes with zero intercept, thereby modeling each multivariate feature as a linear function of another feature, in matrix form. This is interpreted as a partial causal relationship from one feature to another. 
4. Generate a list of quiver representations using the training data, by iteratively adding edges that minimize an entropy score (see below).
5. Calculate the space of sections for each quiver representation. Project the data onto the spaces of sections to transform the data.
6. Reduce dimension again using standard PCA on each quiver feature.

When using standard PCA in the above process, we chose the minimum number of components that exceeded a variance threshold of 95% of the total variance.

Currently, the process for generating a quiver representation is as follows:
1. Choose a reference node as the first node to include in the quiver, at layer 0.
2. At each step, choose a new node to connect by an edge to a node already included in the quiver. The chosen edge minimizes (a normalized version of) the Shannon differential entropy.
    1. The new edge must start at some layer i and end at layer i+1. Thus, there is a unique directed path between the new node and the reference node, directed toward the reference node if i < 0 and away from the reference node if i > 0.
    2. For each edge that could be added, calculate a path map, which is the product of matrices of edges along the path between the new node and the reference node.
    3. At the head of the path there are two sets of data: the actual data for this feature, and the product of the path map with the data for the feature at the tail of the path. Calculate the residuals by taking the difference.
    4. Score each possible edge by estimating (a variant of) the Shannon differential entropy of the residuals. Our approximation of the data’s probability density function uses a Gaussian KDE. In our variant, we divide by the product of the dimensions of the path map’s matrix.
    5. The edge with the minimum entropy score is added to the quiver.
3. Terminate the process after the number of nodes reaches a predefined threshold.

In our analysis, we created one quiver for each possible reference node. We set the threshold for the number of nodes in each quiver to 7. Nodes at layer less than 0 are interpreted as causes of the feature at the reference node. Nodes at layer greater than 0 are interpreted as effects of the feature at the reference node.

We stored multiple types of combinatorial data in JSON format: the collection of all possible representation maps, the quiver representations, and a glossary detailing which univariate features are associated to each node. We implemented a custom JSON encoder/decoder to store objects with numpy arrays, which are not ordinarily compatible with JSON encoding.

The total runtime for Quiver PCA on the crop mapping dataset selection (40637 entries in each of the training/eval sets, 31 features grouped into 11 nodes) is about 10 minutes.

## Evaluation
We compared classification accuracy of several machine learning models on two copies of the evaluation data: a copy transformed using Quiver PCA, and another copy transformed only with classical PCA. The number of principal components taken for the second dataset is the minimum number of components that exceeded a variance threshold of 95% of the total variance. We used k-fold cross-validation with 5 splits.

Our main KPI for the crop mapping dataset is the decrease in accuracy loss, namely:

$$(A_q - A_s)/(1 - A_s)$$

where $A_q$ is the accuracy score on the Quiver PCA-processed data, and $A_s$ is the accuracy score on the standard PCA-processed data.

We also ran an F-test comparing classification accuracy on each cross-validation split. The table below shows the mean and standard deviation of the decrease in accuracy loss, as well as the mode F-test result on the 5 splits.

| Model | Accuracy <br> w/Quiver PCA | Mean Decrease <br> in Acc. Loss | Std. Deviation of <br> Decrease in Acc. Loss | F-test | Hyperparameters |
|:---:|:---:|:---:|:---:|:---:|:---:|
| kNN | 93.55% | -12.39% | 4.10% | False | `n_neighbors=3` |
|         | 93.23% | -12.09% | 2.32% | False | `n_neighbors=5` |
| LightGBM | 84.82% | 1.59% | 2.96% | False | `num_leaves=10`, <br> `min_data_in_leaf=25`, <br> `n_estimators=10` |
|  | 89.73% | 2.57% | 3.22% | False | `num_leaves=35`, <br> `min_data_in_leaf=15`, <br> `n_estimators=10` |
|  | 90.43% | 3.65% | 2.01% | False | `num_leaves=50`, <br> `min_data_in_leaf=10`, <br> `n_estimators=10` |
| Logistic Regression | 83.79% | 15.74% | 0.68% | True | `penalty=’l2’` |
|  | 86.80% | 31.35% | 0.89% | True | `penalty=None` |
| Perceptron | 91.14% | 25.45% | 4.94% | True | `hidden_layer_sizes=(10,)`, <br> `activation=’logistic’` |
| | 91.85% | 20.62% | 1.73% | True | `hidden_layer_sizes=(5,5)`, <br> `activation=’logistic’` |
| Random Forest | 77.66% | 14.79% | 2.29% | True | `max_depth=3`, <br> `n_estimators=20` |
| | 82.77% | 5.29% | 2.81% | True | `max_depth=5`, <br> `n_estimators=20` |
| SVM | 85.77% | 24.31% | 1.83% | True | `kernel=’linear’` |
| | 91.34% | 2.60% | 1.13% | False | `kernel=’rbf’` |
| XGBoost | 88.14% | 7.71% | 1.95% | True | `max_depth=3`, <br> `n_estimators=10` |
| | 91.86% | 12.89% | 1.27% | True | `max_depth=5`, <br> `n_estimators=10` |

We see that Quiver PCA improved accuracy the most for classifiers that use a linear decision boundary (Logistic Regression, Perceptron with one hidden layer, SVM with linear kernel), but a Perceptron with two layers also had improved accuracy. Ensemble-based classifiers had varying results, with XGBoost having the best improvement in accuracy loss and the lowest standard deviation. The variance values for LightGBM show that accuracy changes are inconsistent and highly dependent on the data; this is true to a lesser extent for the Random Forest classifier. While SVM with an RBF kernel was not worse on Quiver PCA-processed data, the accuracy improvement was not statistically significant. Finally, kNN consistently performed worse on the Quiver PCA-processed data compared to standard PCA.

## Future Work
We would like to expand the capabilities of Quiver PCA to generate more varied quiver structures that can detect more complex relationships in the data. We would also like to improve the computational efficiency of Quiver PCA to handle larger datasets.

## Summary of Notebooks
In the `data_processing` folder:
- `00_load_data` - Download and extract crop mapping data; label columns with feature names extracted from the dataset’s web page
- `01_preprocess_data` - Label data columns with feature names; split data into training and evaluation sets
- `02_select_features` - Analyze data features; choose a selection of the data; choose features and group them together into nodes

In the `research` folder:
- `10_process_nodewise_PCA` - Reduce dimension at each node using standard PCA; set aside a copy of the data on which PCA is applied to the entire dataset.
- `11_generate_all_edge_maps` - Use linear regression to estimate the matrix of the representation map between each ordered pair of features
- `12_construct_quiver_reps` - Generate quiver representations, one for each possible reference node
- `13_project_onto_quiver_sections` - Compute each quiver representation’s space of sections; project the data onto the spaces of sections
- `feature_importance_example` - Determine the most important features for an instance of XGBoost; retrieve and display the corresponding quivers

In the `eval` folder:
- Several notebooks organizing model evaluation by model type.

## References

[1] Crop mapping using fused optical-radar data set [Dataset]. (2020). UCI Machine Learning Repository. https://doi.org/10.24432/C5G89D.

[2]  Seigal, A., Harrington, H., and Nanda, V. (2023). Principal Components along Quiver Representations. Foundations of Computational Mathematics (2023) 23:1129–1165.
https://doi.org/10.1007/s10208-022-09563-x
