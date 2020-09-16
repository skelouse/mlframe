def main(df):
    """Testing the get_vif function"""
    return list(dict(df.drop(['car name'], axis=1).get_vif(
                'mpg', verbose=False)).values())
