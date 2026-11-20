import pandas as pd

df = pd.read_csv('data/Groceries_dataset.csv')


# imporove column data types
df = df.dropna(subset=['Member_number', 'Date' , "itemDescription"])


# create basket column by combining 'Member_number' and 'Date' columns

transactions = (
    df.groupby(["Member_number", "Date"])["itemDescription"]
    .apply(lambda items: list(set(items)))
    .tolist()
)



# Converts the list of transactions into a Boolean matrix 

from mlxtend.preprocessing import TransactionEncoder

encoder = TransactionEncoder()

encoded = encoder.fit(transactions).transform(transactions)

basket = pd.DataFrame(
    encoded,
    columns=encoder.columns_
)

# FP Execution and set the minimum support

from mlxtend.frequent_patterns import fpgrowth

minsup = 0.005

frequent_itemsets = fpgrowth(
    basket,
    min_support=minsup,
    use_colnames=True
)

# Association Rule Extraction


from mlxtend.frequent_patterns import association_rules

minconf = 0.10

rules = association_rules(
    frequent_itemsets,
    metric="confidence",
    min_threshold=minconf
)


# Save Results in file
from pathlib import Path

Path("results").mkdir(exist_ok=True)


# Frequently Occurring Sets and support
frequent_itemsets.to_csv(
    "results/frequent_itemsets.csv",
    index=False
)

# confidence
rules.to_csv(
    "results/association_rules.csv",
    index=False
)