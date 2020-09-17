import matplotlib.pyplot as plt


def main(df):
    """Testing the distplot function"""
    fig, ax = plt.subplots()
    df.distplot('mpg', ax=ax)
    # Causes SVD did not converge error in train_test_split
    # fig.savefig('./mlframe/tests/plots/distplot.png')
    return 1
