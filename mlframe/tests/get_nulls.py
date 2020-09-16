import pandas as pd
import numpy as np
def main(df):
    """Testing the get_nulls function"""
    df2 = pd.DataFrame(np.arange(12).reshape(3, 4),
                columns=['A', 'B', 'C', 'D'])
    df.__init__(df2)
    #df['A'].loc[1:3] = np.nan
    df.loc[1:3, 'A'] = np.nan
    return df.get_nulls(verbose=False)