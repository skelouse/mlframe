def main(df):
    """Testing the plot_coef function"""
    df.clean_col_names(inplace=True, verbose=False)
    df.drop('car_name', axis=1, inplace=True)
    df.lrmodel('mpg', inplace=True, verbose=False)
    cor = df.plot_coef()
    with open('./mlframe/tests/plots/coef.html', 'w') as f:
        f.write(cor.render())
    return 1