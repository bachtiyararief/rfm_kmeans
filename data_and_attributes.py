import pandas as pd
import numpy as np

"""
Created on Wed May 25 21:14:20 2022

@author: Bachtiyar M. Arief
"""

class DataSource():
    
    def __init__(self):
        # List of data source 
        filesource = ['10%_original_randomstate=42/retail_data_from_1_until_3_reduce.csv', 
                      '10%_original_randomstate=42/retail_data_from_4_until_6_reduce.csv',
                      '10%_original_randomstate=42/retail_data_from_7_until_9_reduce.csv',
                      '10%_original_randomstate=42/retail_data_from_10_until_12_reduce.csv']
        
        self.listsource = list(map(lambda ls: 'https://dataset.dqlab.id/' + ls, filesource))
        
        self.columntype = dict(
                order_id    = 'object',
                customer_id = 'object',
                city        = 'object',
                province    = 'object',
                product_id  = 'object',
                brand       = 'object',
                quantity    = 'double',
                item_price  = 'double',
                total_price = 'double'
                )
        
        self.__getdata = self.get_data()

    def __read_data(self) -> pd.DataFrame:
        
        for index in range(len(self.listsource)):
            data = pd.read_csv(self.listsource[index], 
                               dtype = self.columntype,
                               parse_dates = ['order_date'])
            
            if(index == 0):
                dataintegration = data
            else:
                dataintegration = pd.concat([dataintegration, data[dataintegration.columns]], ignore_index = True)
               
        return dataintegration
    
    def get_data(self) -> pd.DataFrame:
        
        # Get data integration
        data = self.__read_data().copy()
        
        # 1. Drop if any null value in order_id or customer_id
        data = data.dropna(subset = ['order_id', 'customer_id'], how = 'any')
        
        # 2. Make sure that order_id is numeric format only
        # 3. Drop row which containing empty string, 0 or NaN value in customer_id column
        data = data.loc[(data['order_id'].str.isnumeric()) & (~data['customer_id'].isin(['', '0']))]
        
        # 4. Total price must be positive value
        data['total_price'] = data['total_price'].abs()
            
        # 5. Reset index
        data = data.reset_index(drop = True)
        
        return data
        
    def get_attribute(self, columns : str) -> list:
        return sorted(self.__getdata[columns].unique().tolist())

class Formater():
    
    def __init__(self, **parameter):
        self.data = parameter.get('data')
        self.text = parameter.get('text')
    
    def format_show_data(self, formats : dict):
        return self.data.style.format(formats)
    
    def text_markdown(self, **parameter):
        align = parameter.get('align', 'left')
        color = parameter.get('color', 'black')
        text  = "<p style='text-align: {}; color: {};'>{}</p>".format(align, color, self.text)
        return text
        