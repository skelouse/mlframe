import matplotlib.pyplot as plt
def main(df):
    """Testing the jointplot function"""
    jp = df.jointplot('horsepower', 'mpg')
    jp.savefig('./mlframe/tests/plots/jointplot.png')
    return 1