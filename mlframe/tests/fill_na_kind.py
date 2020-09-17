import numpy as np
import pandas as pd
def main(df):
    """Testing the fill_na_kind function"""
    df['horsepower'][0:3] = np.nan
    nulls = df['horsepower'].isna()
    # def show_nulls(df):
    # print(df[nulls]['horsepower'])
    mode_df = df.fill_na_kind(kind='mode', inplace=False, verbose=False)
    mean_df = df.fill_na_kind(kind='mean', inplace=False, verbose=False)
    median_df = df.fill_na_kind(kind='median', inplace=False, verbose=False)

    perc_df = 0
    custom_df = 0

    def show(df):
        return dict(df['horsepower'][0:3].round(4))
    return dict(
        mode=show(mode_df),
        mean=show(mean_df),
        median=show(median_df),
        perc=perc_df,
        custom=custom_df
    )
