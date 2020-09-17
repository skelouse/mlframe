import pandas as pd
pd.set_option('display.float_format', lambda x: '%.3f' % x)
def main(df):
    """Testing the train_test_split function"""
    df.fill_na_kind('mode', inplace=True)
    df = df.clean_col_names(verbose=False).drop(
                ['car_name', 'origin'], axis=1)
    model = df.train_test_split(
        'mpg', test_size=5, plot=True, verbose=False)
    return round(model.pvalues['cylinders'], 4)
