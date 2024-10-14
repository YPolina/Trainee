
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import kstest, normaltest


#Ectraction the data from dataset
def exctract(dir):
    
    file_paths = os.listdir(dir)
    return list(map(lambda path: pd.read_csv(f'{dir}/{path}'), file_paths))

#Distribution using boxplot
def boxplot(df, feature):
    plt.figure(figsize = (10,6))

    sns.boxplot(x = df[feature], color='blue')

    plt.xlabel(f"{feature}")
    plt.title(f'Distribution of {feature}')

    plt.show()

#understanding the outlier's nature
def add_outlier_check(df, feature, num, x, y):

    plt.figure(figsize = (10,6))

    data = df.loc[df[feature] == num].reset_index(drop = True)

    sns.scatterplot(data = data, x = x, y = y, color = 'green')

    plt.ylabel(f'{y}')
    plt.xlabel(f'{x}')
    plt.title(f'Outlier check for {feature} = {num}')

    plt.show()

#duplicates and na check
def duplicates_na(dfs):
    for i in range(len(dfs)):
        #check for any missing values
        if dfs[i].isna().values.any():
            print(dfs[i].loc[dfs[i].isna().any(axis=1)])
            percentage_na = (dfs[i].isna().any().sum()/len(dfs[i]))*100
            print(f'Percentage of missing values: {percentage_na}%')

            #Drop if less than 3% of missing values
            if percentage_na < 3:
                dfs[i].drop_na()
                print('Missing values successfully dropped')
        #check for any duplicates
        if dfs[i].duplicated(keep = False).values.any():
            print(dfs[i][dfs[i].duplicated()])
            percentage = dfs[i].duplicated().sum()/len(dfs[i]) * 100
            print(f'\nPercentage of duplicates in df_s[{i}]: {(dfs[i].duplicated().sum()/len(dfs[i])):.4%}')
            #Drop if less than 3% of duplicates
            if percentage < 3:
                dfs[i].drop_duplicates()
                print('Duplicates successfully dropped')

#check of similar-named shops
def shop_name_check(df_shops, df_train, shop_name_1, shop_name_2):
    shop_id1 = df_shops.loc[df_shops.shop_name == shop_name_1, 'shop_id'].min()
    c1 = df_train[df_train['shop_id'] == shop_id1].reset_index(drop = True)
    c1 = pd.DataFrame(c1.groupby('date_block_num').agg({'item_cnt_month': 'mean'}))

    shop_id2 = df_shops.loc[df_shops.shop_name == shop_name_2, 'shop_id'].min()
    c2 = df_train[df_train['shop_id'] == shop_id2].reset_index(drop = True)
    c2 = pd.DataFrame(c2.groupby('date_block_num').agg({'item_cnt_month': 'mean'}))
    
    plt.figure(figsize = (10,6))

    plt.bar(c1.index, c1['item_cnt_month'], color = 'g', label = shop_name_1, alpha = 0.8)
    plt.bar(c2.index, c2['item_cnt_month'], color = 'b', label = shop_name_2, alpha = 0.7)

    plt.xticks(range(0,34));
    plt.legend()
    plt.xlabel('Date_block_num')
    plt.ylabel('item_cnt_month')
    plt.title('Shops comparison')

    plt.show()

#Merge two parts of one shop
def shop_corr(df_shops, df_train, shop_name_1, shop_name_2):
    shop_id1 = df_shops.loc[df_shops.shop_name == shop_name_1, 'shop_id'].min()
    shop_id2 = df_shops.loc[df_shops.shop_name == shop_name_2, 'shop_id'].min()
    df_train.replace({'shop_id': {shop_id2: shop_id1}}, inplace = True)
    df_train.replace({'shop_id': {shop_id2: shop_id1}}, inplace = True)

#Check for missing data according 
def completeness_check(df1, df2, feature):
    count = 0
    for id in df1[feature].unique():
        if len(df2[df2[feature] == id]) == 0:
            count += 1
    return f'The total amount of missing data:{count}'

#Convertation to csv
def to_csv(df, filename):
    if os.path.isfile(filename):
        os.remove(filename)
    return df.to_csv(f'{filename}')

#Load from csv file
def load(filename):

    return pd.read_csv(filename, index_col = 0)


#EDA

#Distribution
def distribution(df, feature):
    p1 = df[feature].quantile(0.01)
    p99 = df[feature].quantile(0.99)

    lower_set = df[df[feature] <= p1]
    middle_set = df[(df[feature] > p1) & (df[feature] < p99)]
    upper_set = df[df[feature] > p99]

    fig, axes = plt.subplots(1, 4, figsize = (15,4))

    axes[0].hist(x = df[feature], edgecolor='black', linewidth=1.2, color = 'b')
    axes[0].set_title(f'Distribution of Item_cnt_month')

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




#Outliers specification
#feature_sel - features, by which outliers where detected
def outliers_spec(outliers, feature_sel, df_cat, df_items):
    outliers_spec = {}
    for it_id in outliers.item_id.unique():
        category_name = df_cat[df_cat.item_category_id == df_items[df_items.item_id == it_id].item_category_id.values[0]].item_category_name.values[0]
        item_name = df_items[df_items.item_id == it_id].item_name.values[0]
        sales = outliers[(outliers.item_id == it_id)][['date_block_num', feature_sel]]
        outliers_spec[category_name] = [item_name, sales]

    for key, values in outliers_spec.items():
        print(f'Категория: {key}')
        for val in values:
            print(val)

#using spearman correlation for two numerical features
def correlation_table(numerical):
    columns, correlations = [], []

    for col in numerical:
        columns.append(col)
        correlations.append(stats.spearmanr(group[col], group['item_cnt_month'])[0])

    num_corr = pd.DataFrame({'column': columns, 'correlation': correlations})

    return num_corr.style.background_gradient()


#Using correlation ratio for numerical and multicategorical features
def correlation_ratio(categories, values, cat):
    categories = np.array(categories)
    values = np.array(values)
    
    ssw = 0
    ssb = 0
    for category in set(categories):
        subgroup = values[np.where(categories == category)[0]]
        ssw += sum((subgroup-np.mean(subgroup))**2)
        ssb += len(subgroup)*(np.mean(subgroup)-np.mean(values))**2

    coef =  (ssb / (ssb + ssw))**.5
    print(f'Correlation between sales and {cat}')
    print('Eta_squared: {:.4f}\nEta: {:.4f}'.format(coef**2, coef))
