def main(df):
    """Testing the get_vif function"""
    df = df.drop(['car name'], axis=1)
    vif = df.get_vif('mpg', verbose=False)
    vif_dict = dict(round(vif, 4)).values()
    return list(vif_dict)
