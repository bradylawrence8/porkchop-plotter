import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import datetime as dt
import math
import pandas as pd
import juliandate as jd
from orbittools import *
from time import perf_counter
import streamlit as st

st.set_page_config(layout="wide")
col4, col5 = st.columns([1, 2])
c1 = 0
c2 = 0
c3 = 0
c4 = 0
mdv = 0
fig = plt.figure()
maxv = 20

def toDate(j):
    date = jd.to_gregorian(j)
    year = date[0]
    month = date[1]
    day = date[2]
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    if day < 10:
        return "0{}-{}-{}".format(day, months[month-1], year)
    else:
        return "{}-{}-{}".format(day, months[month-1], year)
    
earthData = pd.read_csv("earth_data_1.csv", header=None).to_numpy()
earthPos = earthData[:, 2:5].astype(float)
earthVel = earthData[:, 5:8].astype(float) * 365.25

marsData = pd.read_csv("mars_data_1.csv", header=None).to_numpy()
marsPos = marsData[:, 2:5].astype(float)
marsVel = marsData[:, 5:8].astype(float) * 365.25

with col4:

    col1, col2, col3 = st.columns(3)

    with col1:
        month1 = st.number_input("Departure Date (MDY):", 1, 12, 1, 1, key=1)
        month2 = st.number_input(".", 1, 12, 1, 1, label_visibility="collapsed", key=2)
        month3 = st.number_input("Arrival Date (MDY):", 1, 12, 1, 1, key=3)
        month4 = st.number_input(".", 1, 12, 1, 1, label_visibility="collapsed", key=4)

    with col2:
        day1 = st.number_input(".", 1, 31, 1, 1, label_visibility="hidden", key=5)
        day2 = st.number_input(".", 1, 31, 1, 1, label_visibility="collapsed", key=6)
        day3 = st.number_input(".", 1, 31, 1, 1, label_visibility="hidden", key=7)
        day4 = st.number_input(".", 1, 31, 1, 1, label_visibility="collapsed", key=8)

    with col3:
        year1 = st.number_input(".", step=1, value=2025, label_visibility="hidden", key=9)
        year2 = st.number_input(".", step=1, value=2028, label_visibility="collapsed", key=10)
        year3 = st.number_input(".", step=1, value=2025, label_visibility="hidden", key=11)
        year4 = st.number_input(".", step=1, value=2028, label_visibility="collapsed", key=12)

    res = st.number_input("Plot resolution (days):", 1, value=1, step=1, key=13)
    maxv = st.number_input("Maximum delta-v (km/s):", value=20, key=14)
    col6, col7 = st.columns([1, 4])
    with col6:
        run = st.button("Run")
    with col7:
        autorun = st.checkbox("Auto-update", False)

with col5:

    if (run or autorun):
        date1 = np.array([year1, month1, day1])
        date2 = np.array([year2, month2, day2])
        date3 = np.array([year3, month3, day3])
        date4 = np.array([year4, month4, day4])

        d1 = jd.from_gregorian(date1[0], date1[1], date1[2])
        d2 = jd.from_gregorian(date2[0], date2[1], date2[2])
        d3 = jd.from_gregorian(date3[0], date3[1], date3[2])
        d4 = jd.from_gregorian(date4[0], date4[1], date4[2])

        c1 = np.where(earthData[:, 0] == d1)[0][0]
        c2 = np.where(earthData[:, 0] == d2)[0][0]
        c3 = np.where(marsData[:, 0] == d3)[0][0]
        c4 = np.where(marsData[:, 0] == d4)[0][0]

        s1 = c2 - c1
        s2 = c4 - c3

        eis = np.arange(c1, c2, res)
        mis = np.arange(c3, c4, res)

        dvs = np.zeros((math.ceil(s2/res), math.ceil(s1/res)))
        i = 0
        j = 0
        mdv = 100000

        marsDate = np.empty(math.ceil(s2/res), dtype='U11')
        earthDate = np.empty(math.ceil(s1/res), dtype='U11')

        for mi in mis:
            for ei in eis:
                if ei < (mi):
                    v1, v2 = lambert(earthPos[ei, :], marsPos[mi, :], 4*math.pi**2, (mi-ei)/365.25)

                    dv1 = v1 - earthVel[ei, :]
                    dv2 = marsVel[mi, :] - v2
                    dv = (np.linalg.norm(dv1) + np.linalg.norm(dv2)) * 4.74047
                    if dv < mdv:
                        mdv = dv
                    if dv < maxv:
                        dvs[j, i] = dv
                    else:
                        dvs[j, i] = np.nan
                else:
                    dvs[j, i] = np.nan
                i += 1
            marsDate[j] = toDate(marsData[mi, 0])
            j += 1
            i = 0
        for ei in eis:
            earthDate[i] = toDate(earthData[ei, 0])
            i += 1

        r, c = np.where(dvs == mdv)

        plt.contourf(earthDate, marsDate, dvs)
        plt.colorbar()
        plt.plot(earthDate[c], marsDate[r], 'xk')
        plt.xticks(np.linspace(0, math.ceil((s1-1)/res), 4))
        plt.yticks(np.linspace(0, math.ceil((s2-1)/res), 7))
    st.pyplot(fig)

with col4:
    st.write("Minimum delta-v: ", mdv)