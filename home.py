import streamlit as st
import time

def mock_login(uname, password):
    time.sleep(1)
    return uname=='jack' and  password=='123'
username = st.text_input('Username','jack')
password = st.text_input('Password','123')

if st.button('Login'):
    with st.spinner('Logging in...'):
        login_result = mock_login(username, password)
        text = '登录成功' if login_result else '登录失败'
        st.write(text)

