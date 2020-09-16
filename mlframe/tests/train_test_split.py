import pandas as pd
pd.set_option('display.float_format', lambda x: '%.3f' % x)
def main(df):
    """Testing the train_test_split function"""
    df.clean_col_names(inplace=True, verbose=False)
    df.drop(['car_name', 'origin'], axis=1, inplace=True)
    model = df.train_test_split(
        'mpg', test_size=5, verbose=False)
    return round(model.pvalues['cylinders'], 4)

