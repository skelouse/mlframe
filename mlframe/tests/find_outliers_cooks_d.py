def main(df):
    idx_outliers = df.find_outliers_cooks_d('horsepower', verbose=False)
    df = df[~idx_outliers]
    return idx_outliers.sum()
