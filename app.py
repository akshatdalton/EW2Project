import streamlit as st
import numpy as np

from polynomial_regression import replace_from_dict, poly, poly_reg_model

st.set_page_config(page_title="EW2 Project")

st.title("EW2 Project")

form = st.form(key='my-form')
p1 = form.text_input('Process1')
p2 = form.text_input('Process2')
p3 = form.text_input('Process3')
temperature = form.number_input('Temperature')
v_supply = form.number_input('V-supply')

submit = form.form_submit_button('Submit')

if submit:
    p1 = replace_from_dict[p1]
    p2 = replace_from_dict[p2]
    p3 = replace_from_dict[p3]
    poly_features = poly.fit_transform(np.array([p1, p2, p3, temperature, v_supply]).reshape(1, -1))
    st.write(f'Predicted v-ref is: {poly_reg_model.predict(poly_features)[0][0]}')
