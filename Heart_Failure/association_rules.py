from os import rename
from mlxtend.frequent_patterns.association_rules import association_rules
import pandas as pd
import numpy as np
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
from patsy import dmatrix

# Import dataset
# ------------------------------------------------------------------
original_df = pd.read_csv("dataset/heart_failure_clinical_records.csv",
                          sep=',', delimiter=None, header='infer')
original_df = original_df.drop(labels=['time'], axis=1)
# print(original_df.head())
# quit()

# Discretization - age, creatinine_phosphokinase, ejection_fraction, platelets, serum_creatinine, serum_sodium
# ------------------------------------------------------------------
age = pd.qcut(original_df['age'].values, 5, labels=False)
creatinine_phosphokinase = pd.qcut(
    original_df['creatinine_phosphokinase'].values, 5, labels=False)
ejection_fraction = pd.qcut(
    original_df['ejection_fraction'].values, 5, labels=False)
platelets = pd.qcut(original_df['platelets'].values, 5, labels=False)
serum_creatinine = pd.qcut(
    original_df['serum_creatinine'].values, 5, labels=False)
serum_sodium = pd.qcut(original_df['serum_sodium'].values, 5, labels=False)


# New Dataframe
# ------------------------------------------------------------------
# Discretize continuous variables
new_df = pd.concat([pd.DataFrame(age), pd.DataFrame(creatinine_phosphokinase), pd.DataFrame(
    ejection_fraction), pd.DataFrame(platelets), pd.DataFrame(serum_creatinine), pd.DataFrame(serum_sodium)], axis=1)
new_df.columns = ['age', 'creatinine_phosphokinase',
                  'ejection_fraction', 'platelets', 'serum_creatinine', 'serum_sodium']
# Transform into Categorical Variables
new_df = dmatrix('C(age) + C(creatinine_phosphokinase) + C(ejection_fraction) + C(platelets) + C(serum_creatinine) + C(serum_sodium) - 1',
                 new_df, return_type='dataframe')
# Addition of original categorical variables
new_df = pd.concat([new_df, pd.DataFrame(original_df['anaemia']), pd.DataFrame(original_df['diabetes']), pd.DataFrame(
    original_df['high_blood_pressure']), pd.DataFrame(original_df['sex']), pd.DataFrame(original_df['smoking']), pd.DataFrame(original_df['DEATH_EVENT'])], axis=1)

# Apply apriori
# ------------------------------------------------------------------
frequent_itemsets = apriori(new_df, min_support=0.05, use_colnames=True)
# Add length
frequent_itemsets['length'] = frequent_itemsets['itemsets'].apply(
    lambda x: len(x))

# frequent_itemsets = frequent_itemsets[(frequent_itemsets['length'] == 2) &
#                                       (frequent_itemsets['support'] >= 0.05)]

# print(frequent_itemsets)

rules = association_rules(frequent_itemsets, metric='confidence', min_threshold=0.3, support_only=False).sort_values(
    by='lift', ascending=False).iloc[:, [0, 1, 4, 5, 6]]
print(rules)

rules.to_csv("rules.csv")

print("Meta-Regras")
rules = rules[(rules['consequents'] == 'DEATH_EVENT')
              & (frequent_itemsets['support'] >= 0.05)]

print(rules)
