def main(df):
    """Testing the scale function"""
    return list(df.drop(['car name'], axis=1).scale(
                   columns='mpg', verbose=False)['mpg'].iloc[0:20])