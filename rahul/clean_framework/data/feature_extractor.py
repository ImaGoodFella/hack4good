# TODO
def get_time_series_features_df(label_df, num_features=0):

    columns = []
    for i in range(num_features):
        label_df[f'dummy{i}'] = 1
        columns.append(f'dummy{i}')
        
    return label_df, columns