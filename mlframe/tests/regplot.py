import matplotlib.pyplot as plt
def main(df):
    """Testing the regplot function"""
    fig, ax = plt.subplots()
    df.regplot('horsepower', 'mpg', ax=ax)
    fig.savefig('./mlframe/tests/plots/regplot.png')
    return 1