def main(df):
    """Testing the log function"""
    return list(df.drop(['car name'], axis=1).log(
                   columns='mpg', verbose=False)['mpg'].iloc[0:20])