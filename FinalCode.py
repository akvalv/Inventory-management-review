#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 19 14:38:22 2023

@author: akashvalathappan
"""


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor 
#from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn import tree

#read the dataset
adidas_df = pd.read_excel("Adidas US Sales Datasets1.xlsx")

#Check info of dataset
adidas_df.info()

#Count NaN values in dataset
adidas_df.isna().sum()
#no missing value are there in dataset

adidas_df.describe() # summary 

#remove columns that are not relevant 
adidas_df=adidas_df.drop(['Retailer ID'],axis=1)


#Find top selling products
top_products = adidas_df.groupby('Product')['Total Sales'].sum().reset_index() #group by products and add sales values
top_products.index+=1 
top_products

fig , ax = plt.subplots(figsize=(8,6))

sns.barplot(data = top_products,y= 'Product',x ='Total Sales',palette='Greens_r')
ax.set_yticklabels(ax.get_yticklabels() , fontsize = 16)

ax.set_xlabel(' Total Sales ', fontsize = 16)
ax.set_ylabel('Product Names ', fontsize = 16)
ax.set_title('Top Selling Products', fontsize = 16)

    
#Top Retailers
top_retailers = adidas_df.groupby('Retailer')['Operating Profit'].sum().sort_values(ascending=False).reset_index()
top_retailers.index += 1
top_retailers
colors = sns.color_palette('muted')[0:6]
fig, ax = plt.subplots(figsize=(8,6))
plt.pie(top_retailers['Operating Profit'], 
       labels=top_retailers['Retailer'], 
       colors=colors,
       autopct='%.0f%%', 
       startangle=90, shadow =True)
plt.show()

#Top Sales Method:
top_methods = adidas_df.groupby('Sales Method')['Total Sales'].sum().sort_values(ascending=False).reset_index()

fig, ax = plt.subplots(figsize=(10,8))
colors = sns.color_palette('bright')[0:5]
sns.barplot(x='Total Sales', y='Sales Method', data=top_methods, palette='Purples_r', ax=ax)
ax.set_xlabel('Total Sales')
ax.set_ylabel('Sales Method')
ax.set_title('Total Sales by Sales Method')
plt.show()


#Top Operating Profit:
top_profit = adidas_df.groupby('Sales Method')['Operating Profit'].sum().sort_values(ascending=False).reset_index()

fig, ax = plt.subplots(figsize=(10,8))
colors = sns.color_palette('bright')[0:5]
sns.barplot(x='Operating Profit', y='Sales Method', data=top_profit, palette='Purples_r', ax=ax)
ax.set_xlabel('Operating Profit')
ax.set_ylabel('Sales Method')
ax.set_title('Total Profits by Sales Method')
plt.show()
    
#units sold
top_units = adidas_df.groupby('Sales Method')['Units Sold'].sum().sort_values(ascending=False).reset_index()

fig, ax = plt.subplots(figsize=(10,8))
colors = sns.color_palette('bright')[0:5]
sns.barplot(x='Units Sold', y='Sales Method', data=top_units, palette='Purples_r', ax=ax)
ax.set_xlabel('UnitsSold')
ax.set_ylabel('Sales Method')
ax.set_title('Total Profits by Sales Method')
plt.show()


# Group data by Retailer and Product, and calculate the sum of Total Sales
x = adidas_df.groupby(['Retailer', 'Product'])['Total Sales'].sum().reset_index()

import matplotlib.pyplot as plt

# create a pivot table to reshape the data
pivot_df = x.pivot(index='Product', columns='Retailer', values='Total Sales')

# create a stacked bar plot
fig, ax = plt.subplots(figsize=(8, 6))
pivot_df.plot(kind='bar', stacked=True, ax=ax)
ax.set_xlabel('Product')
ax.set_ylabel('Total Sales')
ax.set_title('Total Sales by Retailer and Product')
plt.show()


#correlation between different variables.
cor = adidas_df.corr()
sns.heatmap(cor, cmap="Blues_r", annot = True)
plt.show()



import seaborn as sns

import seaborn as sns

import seaborn as sns

g = sns.lmplot(data=adidas_df, x="Operating Profit", y="Total Sales", hue="Sales Method", col="Sales Method", legend_out=True)
g = g.set_titles("{col_name}")
g = g.set_axis_labels("Price per Unit", "Total Sales")
g = g.add_legend(title="Sales Method Code")


sns.lmplot(data=adidas_df, x="Units Sold", y="Total Sales", hue="Sales Method", col="Sales Method")

g = sns.lmplot(data=adidas_df, x="Units Sold", y="Total Sales", hue="Sales Method", col="Sales Method", legend_out=True)
g = g.set_titles("{col_name}")
g = g.set_axis_labels("Units Sold", "Total Sales")
g = g.add_legend(title="Sales Method Code")




#DASHBOARD

# Load data from CSV# Group data by Retailer and Product, and calculate the sum of Operating Profit
data_profit = adidas_df.groupby(['Retailer', 'Product'])['Operating Profit'].sum().reset_index()

# Find the average Operating Profit by Retailer
data_avg_profit = data_profit.groupby('Retailer')['Operating Profit'].mean().reset_index()

# Merge data_profit with data_avg_profit
data_merge = pd.merge(data_profit, data_avg_profit, on='Retailer', suffixes=('', '_avg'))

# Filter data_merge to only show underperforming products (Operating Profit < average Operating Profit by Retailer)
data_underperform = data_merge[data_merge['Operating Profit'] < data_merge['Operating Profit_avg']]

# Group data_underperform by Retailer and Product, and calculate the sum of Operating Profit
data_grouped = data_underperform.groupby(['Retailer', 'Product'])['Operating Profit'].sum().reset_index()

# Sort data_grouped by Operating Profit in ascending order
data_sorted = data_grouped.sort_values('Operating Profit', ascending=True)

# Print the result
print(data_sorted)

#export 

# Export data_sorted as an Excel file
data_sorted.to_excel('underperforming_products.xlsx', index=False)


#visualisation

# Plot the bar chart
plt.bar(data_sorted['Product'], data_sorted['Operating Profit'])
plt.title('Underperforming Products by Operating Profit')
plt.xlabel('Product')
plt.ylabel('Operating Profit')
plt.xticks(rotation=90)  # Rotate x-axis labels by 45 degrees
plt.show()


#Filter rows where Units Sold is zero
zero_sales = adidas_df[adidas_df['Units Sold'] == 0]

# Print the result
print(zero_sales)

# Export data_sorted as an Excel file
zero_sales.to_excel('Zerosales.xlsx', index=False)



'''Analyze the scatter plot to determine whether adjustments need to be made to the pricing strategy. If there is a strong positive correlation between the price per unit and total sales, then the product may be priced too low and the price can be increased to improve profitability. If there is a weak or negative correlation between the price per unit and total sales, then the product may be priced too high and the price can be decreased to increase sales.'''

avg_operating_margin = adidas_df.groupby("Product")["Operating Margin"].mean().reset_index()

sorted_data = avg_operating_margin.sort_values("Operating Margin", ascending=False)

low_margin_products = sorted_data[sorted_data["Operating Margin"] < 10]

#
df = adidas_df

# Group the data by product and sales method
grouped_data = df.groupby(['Product', 'Sales Method'])

# Calculate the total sales and units sold for each product and sales method
product_sales = grouped_data.agg({'Total Sales': 'sum', 'Units Sold': 'sum'})

# Calculate the average price per unit for each product and sales method
product_sales['Avg Price per Unit'] = product_sales['Total Sales'] / product_sales['Units Sold']

# Calculate the percentage of total sales for each product by sales method
product_sales['% of Total Sales'] = product_sales.groupby('Sales Method')['Total Sales'].apply(lambda x: (x / x.sum()) * 100)

# Calculate the percentage of total units sold for each product by sales method
product_sales['% of Total Units Sold'] = product_sales.groupby('Sales Method')['Units Sold'].apply(lambda x: (x / x.sum()) * 100)

# Sort the data by the % of total sales for each product by sales method, in descending order
product_sales_sorted = product_sales.sort_values(by=['% of Total Sales'], ascending=False)

product_sales_sorted
