def main(df):
    """Testing the one_hot_encode function"""
    df.clean_col_names(verbose=False, inplace=True)
    df['model'] = df['car_name'].apply(
        lambda x: x.split(' ')[0]
    )
    return df.one_hot_encode(columns=['model'], verbose=False
        )["model_volvo"].sum()