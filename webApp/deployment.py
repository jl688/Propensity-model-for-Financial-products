import pandas as pd
import streamlit as st
from PIL import Image

st.set_page_config(page_title='Model output')
st.header('Model output for both products')


predicted_dataset = r'Data/df_pred.xlsx'
test_dataset = r'Data/df_test.xlsx'
recommended_dataset = r'Data/df_final_recommend.xlsx'

st.subheader('See Predicted values')
df_predicted = pd.read_excel(predicted_dataset)
st.dataframe(df_predicted)

st.subheader('See test Dataset')
df_testDataset = pd.read_excel(test_dataset)
st.dataframe(df_testDataset)

st.subheader('See Recommended dataset')
df_recommendedDataset = pd.read_excel(recommended_dataset)
st.dataframe(df_recommendedDataset)

