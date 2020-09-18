def main(df):
    """Testing the outlier_removal function"""
    num = len(df)
    df = df.outlier_removal('horsepower',
                       cooks_d=True,
                       verbose=False)
    return (num-len(df))