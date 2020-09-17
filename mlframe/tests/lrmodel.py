def main(df):
    """Testing the lr_model function"""
    df.clean_col_names(inplace=True, verbose=False)
    df.lrmodel('mpg', verbose=False, inplace=True)
    return round(df.model.pvalues[0], 2)