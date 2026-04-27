import pandas as pd


def select_features(df, threshold=0.9):
    non_feature_columns = ['date', 'station_name', 'latitude', 'longitude']

    feature_columns = []
    for column in df.columns:
        if column not in non_feature_columns:
            feature_columns.append(column)

    completeness = {}
    for column in feature_columns:
        not_null_count = df[column].notnull().sum()
        completeness[column] = not_null_count / len(df[column])

    print("Feature Completeness:")
    sorted_columns = sorted(completeness, key=completeness.get, reverse=True)
    for key in sorted_columns:
        print(key + ":", round(completeness[key] * 100, 2), "%")

    selected_features = []
    for key, value in completeness.items():
        if value >= threshold:
            selected_features.append(key)

    print("Selected features (more than", int(threshold * 100), "% filled):", selected_features)

    final_columns = []
    for col in non_feature_columns:
        if col in df.columns:
            final_columns.append(col)

    final_columns += selected_features

    filtered_df = df[final_columns].copy()
    filtered_df = filtered_df.dropna(subset=selected_features, how='all')

    return filtered_df
