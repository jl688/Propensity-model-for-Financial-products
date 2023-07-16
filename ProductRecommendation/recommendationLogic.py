#This code derives the max probability of a product for each customer then recommends a specific product for them
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

pd.set_option('display.max_columns',None)

df = pd.read_excel(r"C:\Open source softwares\propensityModel_Financials\Data\df_pred.xlsx",engine='openpyxl')

df.drop(['Unnamed: 0'],inplace = True,axis =1)

df.head(3)

df['ExpectedRevenueMF'] = df.ProbablitySaleMF * df.RevenueMF
df['ExpectedRevenueCL'] = df.ProbablitySaleCL * df.RevenueCL

df['ExpectedRevenue'] = df[["ExpectedRevenueMF", "ExpectedRevenueCL"]].max(axis=1)

df = df.query("ExpectedRevenue != 0.0")

df['recommendedOffer'] = df[["ExpectedRevenueMF", "ExpectedRevenueCL"]].idxmax(axis=1).str[-2:]

df.sort_values(by=['ExpectedRevenue'], inplace=True, ascending=False)

df_final_recomend = df.iloc[0:100,:]

print("The Expected Revenue from Consumer Loan and Mutual Fund is: ",df_final_recomend.ExpectedRevenue.sum())
#Creates a dataset
df_final_recomend.to_excel("C:\Open source softwares\propensityModel_Financials\Data\df_final_recommend.xlsx")
