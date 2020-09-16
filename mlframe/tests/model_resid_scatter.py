import matplotlib.pyplot as plt
def main(df):
    """Testing the model_resid_scatter function"""
    df.clean_col_names(inplace=True, verbose=False)
    df.lrmodel('mpg', inplace=True, verbose=False)
    fig = plt.figure()
    ax = fig.gca()
    df.model_resid_scatter('mpg', ax=ax)
    fig.savefig('./mlframe/tests/plots/resid_scatter.png')
    return 1