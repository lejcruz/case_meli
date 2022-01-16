import pandas as pd

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