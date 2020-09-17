def main(df):
    """Testing the qq_plot function"""
    df.clean_col_names(inplace=True, verbose=False)
    df.lrmodel('mpg', inplace=True, verbose=False)
    df.qq_plot().savefig('./mlframe/tests/plots/qq_plot.png')
    return 1