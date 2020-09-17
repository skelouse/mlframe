import matplotlib.pyplot as plt


def main(df):
    """Testing the boxplot function"""
    fig, ax = plt.subplots()
    df.boxplot('mpg', ax=ax)
    fig.savefig('./mlframe/tests/plots/boxplot.png')
    return 1
