from re import U
import streamlit as st
import pandas as pd
import plotly.express as px

@st.cache
def load_data(path):
    return pd.read_csv(path)


file_path = "/Users/ruirui/Desktop/IEQ---RA/ATF.csv"
with st.spinner(text="Loading data..."):
    df = load_data(file_path)



Northdf = df[(df["wing"] == 'North')]
Southdf = df[(df["wing"] == 'South')]
Eastdf = df[(df["wing"] == 'East')]

st.title("temp and humid Explorable")
with st.spinner(text="Loading data..."):
     st.text("Visualize the overall dataset and some distributions here...")
if st.checkbox("Show Raw Data"):
    st.write(df.head(10))


st.title("Basic Analysis")

option = st.selectbox(
          'Which wing humidity would you like know about?',
         ("North wing", "South wing", "East wing"))
Northhumid = px.histogram(Northdf, x='km103.rhumid')
Southhumid = px.histogram(Southdf, x='km103.rhumid')
Easthumid = px.histogram(Eastdf, x='km103.rhumid')
if option == "North wing":
    st.write(Northhumid)
if option == "South wing":
    st.write(Southhumid)
if option == "East wing":
    st.write(Easthumid)


humitime = st.selectbox(
     'Which wing humidity change would you like know about?',
     ("North wing", "South wing", "East wing"))
Northhumitime = px.scatter(Northdf, x='Time Point',y='km103.rhumid')
Southhumitime = px.scatter(Southdf, x='Time Point',y='km103.rhumid')
Easthumitime = px.scatter(Eastdf, x='Time Point',y='km103.rhumid')
if humitime == "North wing":
    st.write(Northhumitime)
if humitime == "South wing":
    st.write(Southhumitime)
if humitime == "East wing":
    st.write(Easthumitime)

temp = st.selectbox(
         'Which wing temperature would you like know about?',
        ("North wing", "South wing", "East wing"))
Northtemp = px.histogram(Northdf, x='km103.rtemp')
Southtemp = px.histogram(Southdf, x='km103.rtemp')
Easttemp = px.histogram(Eastdf, x='km103.rtemp')
if temp == "North wing":
    st.write(Northtemp)
if temp == "South wing":
    st.write(Southtemp)
if temp == "East wing":
    st.write(Easttemp)

temptime = st.selectbox(
     'Which wing temperature change would you like know about?',
     ("North wing", "South wing", "East wing"))
Northtemptime = px.scatter(Northdf, x='Time Point',y='km103.rtemp')
Southtemptime = px.scatter(Southdf, x='Time Point',y='km103.rtemp')
Easttemptime = px.scatter(Eastdf, x='Time Point',y='km103.rtemp')
if temptime == "North wing":
    st.write(Northtemptime)
if temptime == "South wing":
    st.write(Southtemptime)
if temptime == "East wing":
    st.write(Easttemptime)



relation = st.selectbox(
     'Which wing relationship between temp and humid point would you like know about?',
     ("North wing", "South wing", "East wing"))
Northrela = px.line(Northdf, x='km103.rhumid',y='km103.rtemp')
Southrela = px.line(Southdf, x='km103.rhumid',y='km103.rtemp')
Eastrela = px.line(Eastdf, x='km103.rhumid',y='km103.rtemp')
if temp == "North wing":
    st.write(Northrela)
if temp == "South wing":
    st.write(Southrela)
if temp == "East wing":
    st.write(Eastrela)