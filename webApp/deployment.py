import pandas as pd
import streamlit as st
from PIL import Image
import os

#this shows the output of this model & how this can be useful.
st.set_page_config(page_title='Model output')
st.header('Model output for both products')


predicted_dataset = r'webApp/data/df_pred.xlsx'
recommended_dataset = r'webApp/data/df_final_recommend.xlsx'

st.subheader('See Predicted values')
df_predicted = pd.read_excel(predicted_dataset)
st.dataframe(df_predicted)

st.subheader('See Recommended dataset')
df_recommendedDataset = pd.read_excel(recommended_dataset)
st.dataframe(df_recommendedDataset)

