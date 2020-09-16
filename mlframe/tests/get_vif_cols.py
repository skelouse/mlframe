def main(df):
    """Testing the get_vif_cols function"""
    return list(df.drop(['car name'], axis=1).get_vif_cols(
                'mpg', verbose=False).index)