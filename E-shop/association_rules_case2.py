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
df = pd.concat([original_df['session ID'],
               original_df['clothing model']], axis=1)
df.columns = ['session_ID', 'clothing_model']


df = df.groupby("session_ID")['clothing_model'].apply(
    lambda tags: ','.join(tags))
df = df.reset_index()


def transaction_format(text):
    return text.split(',')


df['Transaction_list'] = df['clothing_model'].apply(transaction_format)

te = TransactionEncoder()
te_ary = te.fit(df['Transaction_list']).transform(df['Transaction_list'])
new_df = pd.DataFrame(te_ary, columns=te.columns_)
# print(new_df)
# quit()


# Apply apriori
# ------------------------------------------------------------------
frequent_itemsets = apriori(new_df, min_support=0.01, use_colnames=True)
# Add length
frequent_itemsets['length'] = frequent_itemsets['itemsets'].apply(
    lambda x: len(x))

# frequent_itemsets = frequent_itemsets[(frequent_itemsets['length'] >= 2) &
#                                       (frequent_itemsets['support'] >= 0.01)]
print("Regras")
rules = association_rules(frequent_itemsets, metric='confidence', min_threshold=0.85, support_only=False).sort_values(
    by='confidence', ascending=False).iloc[:, [0, 1, 4, 5, 6]]
pd.set_option("display.max_rows", None)
print(rules)


# --------------------------------------------------------
# print("Filtragem")
# rules = rules[(rules['confidence'] >= 0.7)
#               & (rules['support'] >= 0.3)]

# print(rules)
