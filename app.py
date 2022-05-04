import streamlit as st
import numpy as np

from polynomial_regression import replace_from_dict, poly, poly_reg_model

st.set_page_config(page_title="EW2 Project")

st.title("EW2 Project")

form = st.form(key='my-form')
p1 = form.selectbox('Process1', ('tt_lp_bjt', 'ff_lp_bjt', 'ss_lp_bjt'))
p2 = form.selectbox('Process2', ('tt_lp_rvt12', 'ff_lp_rvt12', 'ss_lp_rvt12'))
p3 = form.selectbox('Process3', ('tt_lp_io25', 'ff_lp_io25', 'ss_lp_io25'))
temperature = form.number_input('Temperature')
v_supply = form.number_input('V-supply')

submit = form.form_submit_button('Submit')

if submit:
    p1 = replace_from_dict[p1]
    p2 = replace_from_dict[p2]
    p3 = replace_from_dict[p3]
    poly_features = poly.fit_transform(np.array([p1, p2, p3, temperature, v_supply]).reshape(1, -1))
    st.write(f'Predicted v-ref is: {poly_reg_model.predict(poly_features)[0][0]}')
