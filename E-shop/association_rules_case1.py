from os import rename
from mlxtend.frequent_patterns.association_rules import association_rules
import pandas as pd
import numpy as np
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
from patsy import dmatrix

# Import dataset
# ------------------------------------------------------------------
original_df = pd.read_csv("dataset/e_shop_clothing.csv",
                          sep=';', delimiter=None, header='infer')
original_df = original_df.drop(
    labels=['year', 'month', 'day', 'order', 'session ID'], axis=1)
# print(original_df.head())
# quit()

# Discretization - price
# ------------------------------------------------------------------
price = pd.qcut(original_df['price'].values, 5, labels=False)

# New Dataframe
# ------------------------------------------------------------------
# Discretize continuous variables
new_df = pd.concat(
    [original_df.drop(labels=['price'], axis=1), pd.DataFrame(price)], axis=1)
new_df.columns = ['country', 'main_category', 'clothing_model',
                  'colour', 'location', 'model_photography', 'price2', 'page', 'price']
new_df = new_df.drop(labels=['price'], axis=1)

# Transform into Categorical Variables
new_df = dmatrix('C(country) + C(main_category) + C(clothing_model) + C(colour) + C(location) + C(model_photography) + C(price2) + C(page)- 1',
                 new_df, return_type='dataframe')

# print(new_df.head())
# quit()


# Apply apriori
# ------------------------------------------------------------------
frequent_itemsets = apriori(new_df, min_support=0.05, use_colnames=True)
# Add length
frequent_itemsets['length'] = frequent_itemsets['itemsets'].apply(
    lambda x: len(x))

# frequent_itemsets = frequent_itemsets[(frequent_itemsets['length'] >= 2) &
#                                       (frequent_itemsets['support'] >= 0.1)]
print("Regras")
rules = association_rules(frequent_itemsets, metric='confidence', min_threshold=0.3, support_only=False).sort_values(
    by='consequents', ascending=False).iloc[:, [0, 1, 4, 5, 6]]
pd.set_option("display.max_rows", None)
print(rules)


# --------------------------------------------------------
print("Filtragem")
rules = rules[(rules['confidence'] >= 0.7)
              & (rules['support'] >= 0.3)]

print(rules)
