
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import kstest, normaltest


class DataQualityLayer:
    def __init__(self):
        pass
    
    # Data extraction
    def extract_data(self, directory: str):
        """Extracts all CSV files in a directory and loads them into a list of DataFrames"""
        file_paths = os.listdir(directory)
        data_frames = [pd.read_csv(os.path.join(directory, path)) for path in file_paths]
        return data_frames

    # Data Distribution Visualization
    def boxplot(self, df, feature, color='blue'):
        """Generates a boxplot for a given feature in a DataFrame"""
        plt.figure(figsize=(10, 6))
        sns.boxplot(x=df[feature], color=color)
        plt.xlabel(feature)
        plt.title(f'Distribution of {feature}')
        plt.show()

    # Outlier Analysis
    def outlier_check(self, df, feature, value, x, y):
        """Visualizes potential outliers for a specific feature"""
        plt.figure(figsize=(10, 6))
        data = df.loc[df[feature] == value].reset_index(drop=True)
        sns.scatterplot(data=data, x=x, y=y, color='green')
        plt.ylabel(y)
        plt.xlabel(x)
        plt.title(f'Outlier check for {feature} = {value}')
        plt.show()

    # Check for Duplicates and NA values
    def duplicates_na_check(self, dfs, na_threshold=3, duplicate_threshold=3):
        """Checks for and handles duplicates and missing values in DataFrames list"""
        for idx, df in enumerate(dfs):
            # Missing values check
            if df.isna().values.any():
                print(f"DataFrame[{idx}] has missing values:\n", df[df.isna().any(axis=1)])
                na_percentage = (df.isna().sum().sum() / df.size) * 100
                print(f'Percentage of missing values: {na_percentage:.2f}%')
                if na_percentage < na_threshold:
                    df.dropna(inplace=True)
                    print('Missing values successfully dropped')
            # Duplicates check
            if df.duplicated(keep=False).any():
                print(f"DataFrame[{idx}] has duplicates:\n", df[df.duplicated()])
                duplicate_percentage = (df.duplicated().sum() / len(df)) * 100
                print(f'Percentage of duplicates: {duplicate_percentage:.2f}%')
                if duplicate_percentage < duplicate_threshold:
                    df.drop_duplicates(inplace=True)
                    print('Duplicates successfully dropped')

    # Check for shops with similar names
    def shop_name_comparison(self, df_shops, df_train, shop_name_1, shop_name_2):
        """Compares the sales of two shops with similar names"""
        shop_id1 = df_shops[df_shops.shop_name == shop_name_1].shop_id.min()
        shop_id2 = df_shops[df_shops.shop_name == shop_name_2].shop_id.min()
        sales_1 = df_train[df_train.shop_id == shop_id1].groupby('date_block_num')['item_cnt_month'].mean()
        sales_2 = df_train[df_train.shop_id == shop_id2].groupby('date_block_num')['item_cnt_month'].mean()

        plt.figure(figsize=(10, 6))
        plt.bar(sales_1.index, sales_1.values, color='green', label=shop_name_1, alpha=0.8)
        plt.bar(sales_2.index, sales_2.values, color='blue', label=shop_name_2, alpha=0.7)
        plt.xlabel('Date_block_num')
        plt.ylabel('Average item count')
        plt.legend()
        plt.title('Comparison of Shops with Similar Names')
        plt.show()

    # Replace duplicate shops by name
    def replace_shops(self, shops, train, test, shop_name_1, shop_name_2):
        """Replace duplicate shops based on similar names by replacing shop_id"""
        shop_id1 = shops[shops.shop_name == shop_name_1].shop_id.min()
        shop_id2 = shops[shops.shop_name == shop_name_2].shop_id.min()
        # Replace shop_id2 with shop_id1 across datasets
        train.loc[train.shop_id == shop_id1, 'shop_id'] = shop_id2
        test.loc[test.shop_id == shop_id1, 'shop_id'] = shop_id2
        shops.loc[shops.shop_id == shop_id1, 'shop_id'] = shop_id2

    # Completeness check between datasets
    def completeness_check(self, df1, df2, feature):
        """Checks completeness of feature data between two DataFrames"""
        missing_count = sum(~df1[feature].isin(df2[feature].unique()))
        print(f'Total missing entries in {feature}: {missing_count}')
        return missing_count

    # Export DataFrame to CSV
    def save_to_csv(self, df, filename):
        """Saves the DataFrame to a CSV file"""
        if os.path.exists(filename):
            os.remove(filename)
        df.to_csv(filename, index=False)
        print(f'Saved DataFrame to {filename}')


#EDA

#loader
def loader(file: str) -> pd.DataFrame:

    df = pd.read_csv(file)
    return df


#Distribution
def distribution(df, feature):
    p1 = df[feature].quantile(0.01)
    p99 = df[feature].quantile(0.99)

    lower_set = df[df[feature] <= p1]
    middle_set = df[(df[feature] > p1) & (df[feature] < p99)]
    upper_set = df[df[feature] > p99]

    fig, axes = plt.subplots(1, 4, figsize = (15,4))

    axes[0].hist(x = df[feature], edgecolor='black', linewidth=1.2, color = 'b')
    axes[0].set_title(f'Distribution of {feature}')

    axes[1].hist(x = lower_set[feature], edgecolor='black', linewidth=1.2, color = 'b')
    axes[1].set_title(f'Lower {p1}')

    axes[2].hist(x = middle_set[feature], edgecolor='black', linewidth=1.2, color = 'b')
    axes[2].set_title(f'Between {p1} and {p99}')

    axes[3].hist(x = upper_set[feature], edgecolor='black', linewidth=1.2, color = 'b')
    axes[3].set_title(f'Higher {p99}')

    for ax in axes:
        ax.set_ylabel('Frequency')
        ax.set_xlabel(f'{feature}')

    plt.tight_layout()
    plt.show()


