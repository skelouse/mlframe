# def test_car():
#     df = MLFrame(pd.read_csv('mlframe/tests/auto-mpg.csv'))
#     df.clean_col_names(inplace=True)
#     df['model'] = df['car_name'].apply(lambda x: x.split(' ')[0])
#     df.drop(['car_name'], axis=1, inplace=True)
#     df['model'] = df['model'].astype('category')
#     df_ohe = df.one_hot_encode(['model'])
#     df_ohe.clean_col_names(inplace=True)
#     df_ohe.model_and_plot('horsepower', inplace=True)
#     print(df_ohe.model.summary())

# def test_houses():
#     df = MLFrame(pd.read_csv('kc_house_data.csv'))
#     first_model = df.fill_na_mode(  # Fill na for the model
#         ).drop(['date', 'sqft_basement'], axis=1  # Dropping date and
#         ).model_and_plot('price')                 # sqft_basement for example
#     # df.model = first_model
#     # df.plot_coef()
#     # df.plot_corr(annot=True)
#     df['sqft_basement'] = df['sqft_basement'].apply(
#         lambda x: 0 if x == '?' else x)
#     df['sqft_basement'] = df['sqft_basement'].astype(float)
#     cat_cols = ['zipcode', 'condition', 'view']
#     for col in cat_cols:
#         df[col] = df[col].astype('category')
#     df_ohe = df.drop(['date', 'id', 'lat', 'long'], axis=1
#         ).one_hot_encode(cat_cols)
#     df_ohe['waterfront'].fillna(0, inplace=True)
#     df_ohe.fill_na_mode(inplace=True)
#     drop_cols = [x for x in df_ohe.columns if 'condition' in x]
#     drop_cols.append('sqft_living')
#     df_ohe.drop(drop_cols, axis=1, inplace=True)
#     df_ohe.clean_col_names(inplace=True)
#     df_ohe.model_and_plot('price')
#     print(df_ohe.model.summary())
