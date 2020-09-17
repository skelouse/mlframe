def main(df):
    """Testing the find_outliers_Z function"""
    idx_outliers = df.find_outliers_Z('horsepower', verbose=False)
    df = df[~idx_outliers]
    return idx_outliers.sum()