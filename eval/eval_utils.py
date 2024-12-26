import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedKFold

from mlxtend.evaluate import ftest



# Evaluate the difference in model performance on two sets Xq and Xs arising from
# transforming the dataset using quiver representations as opposed to just using principal component analysis
def evaluate_model_performance_from_preprocessing(Xq, Xs, y, kfold, model_constructor, params):
    y_holdovers = []
    split_accuracy_scores_q = []
    split_accuracy_scores_s = []
    ftests = []
    for j, (train_index, test_index) in enumerate(kfold.split(Xq, y)):
        y_holdovers.append(y[test_index])
        
        Xq_tr = Xq[train_index]
        Xq_ho = Xq[test_index]
        Xs_tr = Xs[train_index]
        Xs_ho = Xs[test_index]
        y_tr = y[train_index]
        y_ho = y[test_index]
    
        model_q = model_constructor(**params)
        model_q.fit(Xq_tr, y_tr)
        yq_pr = model_q.predict(Xq_ho)
        accuracy_score_1 = model_q.score(Xq_ho, y_ho)
        split_accuracy_scores_q.append(accuracy_score_1)
        
        model_s = model_constructor(**params)
        model_s.fit(Xs_tr, y_tr)
        ys_pr = model_s.predict(Xs_ho)
        accuracy_score_s = model_s.score(Xs_ho, y_ho)
        split_accuracy_scores_s.append(accuracy_score_s)

        split_ftest = ftest(y_holdovers[j], yq_pr, ys_pr)
        split_ftest_res = (accuracy_score_1 > accuracy_score_s and split_ftest[1] < 0.05)
    
        ftests.append(split_ftest_res)
        print(f"Models trained and evaluated on split {j}.\nAccuracy on quiver-processed data: {accuracy_score_1:.2%}\nAccuracy Score on data processed with standard PCA: {accuracy_score_s:.2%}\nF-test: {accuracy_score_1 > accuracy_score_s and split_ftest_res} ({split_ftest})\n")


    split_accuracy_scores_q = np.array(split_accuracy_scores_q)
    split_accuracy_scores_s = np.array(split_accuracy_scores_s)
    accuracy_losses = (split_accuracy_scores_q - split_accuracy_scores_s)/(1 - split_accuracy_scores_s)
    mean_accuracy_loss = accuracy_losses.mean()
    std_accuracy_loss = accuracy_losses.std()
    mean_accuracy_q = split_accuracy_scores_q.mean()

    ftest_result = len([test for test in ftests if test]) > 0.5*len(ftests)

    
    print(f"Mean accuracy on QPCA-processed data: {mean_accuracy_q:.2%}")
    print(f"Mean decrease in accuracy loss: {mean_accuracy_loss:.2%}")
    print(f"Standard deviation of decrease in accuracy loss: {std_accuracy_loss:.2%}")
    if ftest_result:
        print("There is a significant improvement in accuracy loss when the data is preprocessed with QPCA in comparison to standard PCA.")
    else: 
        print("There is no significant improvement in accuracy loss when the data is preprocessed with QPCA in comparison to standard PCA.")


    