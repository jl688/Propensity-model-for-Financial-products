import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

pd.set_option('display.max_columns',None)

# Assumptions:
# 1) Assume that to have a bank account with the bank, clients must be at least 10 years old, as even a student account requires a minimum age of 10.
# 2) Impute the age by adding 10 years to the tenure of the client.

def processed_data():
    # Read the datasets
    df_demog = pd.read_excel(r"Data/Financial dataset for propensity.xlsx", engine='openpyxl', sheet_name='Soc_Dem')
    df_prod = pd.read_excel(r"Data/Financial dataset for propensity.xlsx", engine='openpyxl', sheet_name='Products_ActBalance')
    df_in_out = pd.read_excel(r"Data/Financial dataset for propensity.xlsx", engine='openpyxl', sheet_name='Inflow_Outflow')
    df_sales = pd.read_excel(r"Data/Financial dataset for propensity.xlsx", engine='openpyxl', sheet_name='Sales_Revenues')

    # Merge the datasets
    df = pd.merge(df_demog, df_prod, how="left", on=["Client"])
    df = pd.merge(df, df_in_out, how="left", on=["Client"])
    df_train = pd.merge(df, df_sales[['Client', 'Sale_MF', 'Revenue_MF']], how="inner", on=["Client"])

    # Drop unnecessary columns
    columns_sale_cl = ['Count_MF', 'ActBal_MF']
    df_train.drop(columns_sale_cl, inplace=True, axis=1)

    # Replace missing values in the 'Sex' field with 'U' (Unknown)
    df_train.Sex = df_train.Sex.replace(np.nan, "U", regex=True)

    # Convert 'M' and 'F' to 1 and 0, respectively, in the 'Sex' field
    df_train.Sex = df_train.Sex.replace({'M': 1, 'F': 0, 'U': 2})

    # Impute age with tenure + 120 months
    df_train.Age = np.where((df_train.Age * 12 <= df_train.Tenure), round(df_train.Tenure / 12) + 10, df_train.Age)

    # Impute other missing values with 0
    df_train.fillna(0, inplace=True)

    # Return the processed data
    return df_train
