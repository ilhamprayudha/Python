#import packages.

import pandas as pd
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

#datasets "using Online Retail.xlsx"
dataset = pd.read_excel ('OnlineRetail.xlsx')
dataset.head()

#Using Country as data Unique , so the dataset is used as per the Country
dataset.Country.unique()

dataset_UK = dataset[dataset.Country == "Japan"]

#reorganize data using pandas.pivot
basket = pd.pivot_table(dataset_UK, index='InvoiceNo', columns='Description', values = 'Quantity',fill_value=0)

#apply apriori algorithm 
# I use minimal support 50%
frequent_item = apriori(basket, min_support=0.5, use_colnames=True)
frequent_item.head()

#rules
#I use minimal threshold 80%
rules = association_rules(frequent_item,metric='confidence', min_threshold=0.8)
rules.head()
