import pandas as pd
import numpy as np

from sklearn.preprocessing import OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn import linear_model

from sklearn.metrics import precision_score, recall_score, precision_recall_curve, roc_curve, roc_auc_score, confusion_matrix


import scipy.stats as stat

import matplotlib.pyplot as plt

def replace_missing(X, obj_replace = '-999', num_replace = -999):
    '''
    Function to replace missing for a user pre defined value
    replace values for categorical and numerical data

    It's nedeed a error handling for empty dataframes or dataframes that have only one data type
    '''

    # identify data types
    __dtypes__ = X.dtypes

    # Put in list the name of columns for object and numerical columns
    __ObjCols__ = __dtypes__[__dtypes__ == 'object'].index.to_list()
    __NumCols__ = __dtypes__[__dtypes__ != 'object'].index.to_list()

    # Split data set
    __ObjDf__ = X[__ObjCols__]
    __NumDf__ = X[__NumCols__]

    # replace missing
    __ObjDf__.fillna(obj_replace, inplace=True)
    __NumDf__.fillna(num_replace, inplace=True)

    #Return DataFrame
    df_hand_missing = pd.concat([__ObjDf__, __NumDf__], axis=1)

    #keep columns order
    df_hand_missing = df_hand_missing[__dtypes__.index]

    return df_hand_missing


if __name__ == "__main__":
    main()


def encode_df (column_transformer, X, feature_names, type='fit_transform'):

    '''
    type receive "fit_transform" or "transform"
    '''

    if type == 'fit_transform':

        __X__ = X.copy()

        X_LE = pd.DataFrame( column_transformer.fit_transform(__X__), 
                                columns=feature_names, 
                                index=__X__.index)

        X_LE = pd.concat([X_LE, __X__.drop(columns=feature_names)], axis=1)
        X_LE = X_LE[__X__.columns]
        
        return X_LE
    
    elif type == 'transform':

        __X__ = X.copy()

        X_LE = pd.DataFrame( column_transformer.transform(__X__), 
                                columns=feature_names, 
                                index=__X__.index)

        X_LE = pd.concat([X_LE, __X__.drop(columns=feature_names)], axis=1)
        X_LE = X_LE[__X__.columns]
        
        return X_LE        
    
    else:
        print('Error')

# Using solution found on  https://gist.github.com/rspeare/77061e6e317896be29c6de9a85db301d to fit a Logistic Regression and return P_values
class LogisticReg:
    """
    Wrapper Class for Logistic Regression which has the usual sklearn instance 
    in an attribute self.model, and pvalues, z scores and estimated 
    errors for each coefficient in 
    
    self.z_scores
    self.p_values
    self.sigma_estimates
    
    as well as the negative hessian of the log Likelihood (Fisher information)
    
    self.F_ij
    """
    
    def __init__(self,*args,**kwargs):#,**kwargs):
        self.model = linear_model.LogisticRegression(*args,**kwargs)#,**args)

    def fit(self,X,y):
        self.model.fit(X,y)
        #### Get p-values for the fitted model ####
        denom = (2.0*(1.0+np.cosh(self.model.decision_function(X))))
        denom = np.tile(denom,(X.shape[1],1)).T
        F_ij = np.dot((X/denom).T,X) ## Fisher Information Matrix
        Cramer_Rao = np.linalg.inv(F_ij) ## Inverse Information Matrix
        sigma_estimates = np.sqrt(np.diagonal(Cramer_Rao))
        z_scores = self.model.coef_[0]/sigma_estimates # z-score for eaach model coefficient
        p_values = [stat.norm.sf(abs(x))*2 for x in z_scores] ### two tailed test for p-values
        
        self.z_scores = z_scores
        self.p_values = p_values
        self.sigma_estimates = sigma_estimates
        self.F_ij = F_ij


def evaluate_model(y_true, y_pred_score, y_challenger_score):

    # Naive Model
    naive_score = [0 for _ in range(len(y_true))]

    # Compute AUC Scores

    naive_auc = roc_auc_score(y_true, naive_score)
    model_auc = roc_auc_score(y_true, y_pred_score)
    challenger_auc = roc_auc_score(y_true, y_challenger_score)

    
    print('*'*40)
    print('-'*10,'AUC Scores''-'*10)
    print(f'Naive: {naive_auc}')
    print(f'Model: {model_auc}')
    print(f'Challenger: {challenger_auc}')


    # Compute False Positive Rate and True Positive Rate

    naive_fpr, naive_tpr, _ = roc_curve(y_true, naive_score)
    model_fpr, model_tpr, _ = roc_curve(y_true, y_pred_score)
    challenger_fpr, challenger_tpr, _ = roc_curve(y_true, y_challenger_score)

    # Roc Curve

    fig_roc = plt.figure(figsize=(14, 8))

    naive_plot = plt.plot(naive_fpr, naive_tpr, linestyle='--', label='Naive')
    model_plot = plt.plot(model_fpr, model_tpr, marker='.', label='Model')
    challenger_plot = plt.plot(challenger_fpr, challenger_tpr, marker='.', label='Challenger')

    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()

    plt.show()


    # Precision vs Recall Curve

    precisions_model, recalls_model, thresholds_model = precision_recall_curve(y_true, y_pred_score)
    precisions_challenger, recalls_challenger, thresholds_challenger = precision_recall_curve(y_true, y_challenger_score)

    fig_pvr = plt.figure(figsize=(14, 8))
    prec_plot_model = plt.plot(thresholds_model, precisions_model[:-1], "-", color='blue', label="Precision Model")
    recal_plot_model = plt.plot(thresholds_model, recalls_model[:-1], "-", color='green', label="Recall Model")

    prec_plot_challenger = plt.plot(thresholds_challenger, precisions_challenger[:-1], "*",  color='red', label="Precision Challenger")
    recal_plot_challenger = plt.plot(thresholds_challenger, recalls_challenger[:-1], "*", color='purple', label="Recall Challenger")

    plt.xlabel('Treshold')
    plt.legend()

    plt.show()

    fig_pvr2 = plt.figure(figsize=(14, 8))

    prec_rec_model_plot = plt.plot(recalls_model[:-1], precisions_model[:-1], marker="*", label="Model")
    prec_rec_challenger_plot = plt.plot(recalls_challenger[:-1], precisions_challenger[:-1], marker=".", label="Challenger")

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend()

    plt.show()


def define_treshold(y_true, y_pred_score, margin, cost, range_treshold):
    
    model_treshold_results = pd.DataFrame() 

    for __treshold__ in range_treshold:

        __treshold__ = __treshold__/1000

        __y_pred__ = [1 if x >= __treshold__ else 0 for x in y_pred_score]

        tn, fp, fn, tp = confusion_matrix(y_true, __y_pred__).ravel()

        __ExpectedProfit__ = margin*tn - cost*fn

        __precision__ = precision_score(y_true, __y_pred__)
        __recall__ = recall_score(y_true, __y_pred__)

        __results__ = pd.DataFrame({'treshold': __treshold__, 
                    'ExpectedProfit': __ExpectedProfit__, 
                    'precision': __precision__, 
                    'recall': __recall__, 
                    'TrueNegatives': tn,
                    'FalsePositives':fp,
                    'FalseNegatives':fn,
                    'TruePositives':tp},
                    index=[__treshold__],
                    )

        model_treshold_results = pd.concat([model_treshold_results, __results__])

        model_treshold_results['profit_precision'] = model_treshold_results['ExpectedProfit'] * model_treshold_results['precision']
        model_treshold_results['profit_recall'] = model_treshold_results['ExpectedProfit'] * model_treshold_results['recall']
        model_treshold_results['profit_precision_recall'] = model_treshold_results.apply(lambda x: np.mean([x['profit_precision'], x['profit_recall']] ), axis=1 )


        model_treshold_results = model_treshold_results.loc[
            (model_treshold_results['recall']>0) 
            & (model_treshold_results['precision']>0)
            & (model_treshold_results['ExpectedProfit']>0)
            ,:]

        best_profit = model_treshold_results.sort_values('ExpectedProfit', ascending=False).iloc[:1,:]
        best_profit['Criteria'] = 'best_profit'

        best_recall = model_treshold_results.sort_values('recall', ascending=False).iloc[:1,:]
        best_recall['Criteria'] = 'best_recall'

        best_precision = model_treshold_results.sort_values('precision', ascending=False).iloc[:1,:]
        best_precision['Criteria'] = 'best_precision'

        best_profit_precision = model_treshold_results.sort_values('profit_precision', ascending=False).iloc[:1,:]
        best_profit_precision['Criteria'] = 'best_profit_precision'

        best_profit_recall = model_treshold_results.sort_values('profit_recall', ascending=False).iloc[:1,:]
        best_profit_recall['Criteria'] = 'best_profit_recall'

        best_profit_precision_recall = model_treshold_results.sort_values('profit_precision_recall', ascending=False).iloc[:1,:]
        best_profit_precision_recall['Criteria'] = 'best_profit_precision_recall'


        best_tresholds = pd.concat([best_profit, best_recall,  best_precision, best_profit_precision, best_profit_recall, best_profit_precision_recall])

    return {'all_results': model_treshold_results,
            'best_tresholds':best_tresholds}