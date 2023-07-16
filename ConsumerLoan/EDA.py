import pandas as pd
import numpy as np

import warnings
warnings.filterwarnings('ignore')

pd.set_option('display.max_columns',None)


# Assumptions:
# 1) Assume that to have a bank account with the bank, clients must be at least 10 years old, as even a student account requires a minimum age of 10.
# 2) Impute the age by adding 10 years to the tenure of the client.
def process_data():
    file_path = 'Data/Financial dataset for propensity.xlsx'

    df_demog = pd.read_excel(file_path, engine='openpyxl', sheet_name='Soc_Dem')
    df_prod = pd.read_excel(file_path, engine='openpyxl', sheet_name='Products_ActBalance')
    df_in_out = pd.read_excel(file_path, engine='openpyxl', sheet_name='Inflow_Outflow')
    df_sales = pd.read_excel(file_path, engine='openpyxl', sheet_name='Sales_Revenues')

    # Merging the datasets
    df = pd.merge(df_demog, df_prod, how="left", on=["Client"])
    df = pd.merge(df, df_in_out, how="left", on=["Client"])
    df_train = pd.merge(df, df_sales[['Client', 'Sale_CL', 'Revenue_CL']], how="inner", on=["Client"])

    # Dropping unnecessary columns
    columns_sale_cl = ['Count_CL', 'ActBal_CL']
    df_train.drop(columns_sale_cl, inplace=True, axis=1)

    # Handling missing values
    df_train.Sex = df_train.Sex.replace(np.nan, "U", regex=True)
    df_train.Sex = df_train.Sex.replace({'M': 1, 'F': 0, 'U': 2})
    df_train.fillna(0, inplace=True)
    df_train.Age = np.where((df_train.Age * 12 <= df_train.Tenure), round(df_train.Tenure / 12) + 10, df_train.Age)

    return df_train
