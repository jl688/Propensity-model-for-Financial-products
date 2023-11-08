import pandas as pd
import warnings
warnings.filterwarnings('ignore')

pd.set_option('display.max_columns',None)

df_demog = pd.read_excel(r"C:\Open source softwares\propensityModel_Financials\Data\Financial dataset for propensity.xlsx",engine='openpyxl',sheet_name='Soc_Dem')
df_prod = pd.read_excel(r"C:\Open source softwares\propensityModel_Financials\Data\Financial dataset for propensity.xlsx",engine='openpyxl',sheet_name='Products_ActBalance')
df_in_out = pd.read_excel(r"C:\Open source softwares\propensityModel_Financials\Data\Financial dataset for propensity.xlsx",engine='openpyxl',sheet_name='Inflow_Outflow')
df_sales = pd.read_excel(r"C:\Open source softwares\propensityModel_Financials\Data\Financial dataset for propensity.xlsx",engine='openpyxl',sheet_name='Sales_Revenues')

df = pd.merge(df_demog, df_prod, how="left", on=["Client"])
df = pd.merge(df, df_in_out, how="left", on=["Client"])
df_train = pd.merge(df, df_sales, how="outer", on=["Client"])
df_test = df_train[df_train['Sale_CL'].isna()]
df_test.drop(['Sale_CL','Revenue_CL','Sale_MF','Revenue_MF','Sale_CC','Revenue_CC'],inplace = True,axis =1)
# creates test Dataset for further operation
df_test.to_excel(r"C:\Open source softwares\propensityModel_Financials\Data\testDatasetCreation\test.xlsx")
