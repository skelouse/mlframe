import numpy as np
import pandas as pd
def main(df):
    """Testing the fill_na_mode function"""
    df2 = pd.DataFrame(np.arange(12).reshape(3, 4),
                columns=['A', 'B', 'C', 'D'])
    df.__init__(df2)
    df['A'].loc[1:3] = np.nan
    df['B'].loc[0] = np.nan
    df.fill_na_mode(inplace=True, verbose=False)
    return df.get_nulls(verbose=False)