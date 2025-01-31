import streamlit as st
import pandas as pd

# Check if 'results_df' is in session state
if 'results_df' in st.session_state:
    df = st.session_state['results_df']
else:
    df = pd.DataFrame({
        'Column1': [1, 2, 3],
        'Column2': ['A', 'B', 'C']
    })

st.title('Display DataFrame')

st.write('Here is the DataFrame stored in the session:')
st.dataframe(df)