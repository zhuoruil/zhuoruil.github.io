from re import U
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

@st.cache
def load_data18():
    df = pd.read_csv('18 sensors.csv')
    return df

@st.cache
def load_data5():
    df = pd.read_csv('5 sensors.csv')
    return df


getATF18 = pd.read_csv('18 sensors.csv',parse_dates=['Time Point'], na_values='-')

getATF5 = pd.read_csv('5 sensors.csv',parse_dates=['Time Point'], na_values='-')

TheFullOne = getATF18

TheFullOne.append(getATF5, ignore_index=True)

getTrueATF18 = getATF18[["Sensor", "# Time Zone:", "Time Point", "km100.rpm10c", "km100.rpm25c", "km103.rhumid", "km103.rtemp",
         "km103.rtvoc (ppb)", "rco2 (ppm)"]]
getTrueATF5 = getATF5[["Sensor", "# Time Zone:", "Time Point", "km100.rpm10c", "km100.rpm25c", "km103.rhumid", "km103.rtemp",
         "km103.rtvoc (ppb)", "rco2 (ppm)"]]

getTrueATF23 = getTrueATF18

df = getTrueATF23.append(getTrueATF5, ignore_index=True)
df = getTrueATF23.iloc[:,[0,1,2,5,6]]
df['km103.rhumid'] = df['km103.rhumid'].astype('float')
df['km103.rtemp'] = df['km103.rtemp'].astype('float')

East = ['VC21411021','VC21410972','VC21410978','VC21410975','VC21410570','VC21410977']
North = ['VC21410705','VC21410989','DC21410602','KC20710690','VC21410706','VC21410682','VC21410650','VC21410733','VC21411024']
South = ['VC21410644','KC20710673','VC21410995','VC21410651','VC21411018','VC21411016','VC21410591','VC21410979','VC21410889','VC21410701']
def wing(x):
  if x in East:
    return "East"
  elif x in North:
    return "North"
  elif x in South:
    return "South"
df['wing'] = df['Sensor'].apply(lambda x: wing(x))

OpenOffice = ['VC21411021','VC21410975','VC21410570','VC21410977','VC21410705','VC21410989','DC21410602','KC20710690','VC21410650','KC20710673','VC21410995','VC21411018','VC21411016','VC21410591','VC21410979','VC21410889']
Conference = ['VC21410972','VC21410682','VC21411024','VC21410644']
EnclosedOffice = ['VC21410978','VC21410733','VC21410651','VC21410701']
def officetype(x):
  if x in OpenOffice:
    return "OpenOffice"
  elif x in Conference:
    return "Conference"
  elif x in EnclosedOffice:
    return "EnclosedOffice"
df['officetype'] = df['Sensor'].apply(lambda x: officetype(x))

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