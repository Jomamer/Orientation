import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn import linear_model, svm

st.write('''
# Prediction of academic orientations
This application predicts the learner's ability and learner's stream''')

st.sidebar.header("Insert learner's averages from 6th grade to 9th grade")
def user_input():
        mm6 = st.sidebar.slider('Mathematics 6th grade', 0, 20, 10)
        mpc6 = st.sidebar.slider('PCT 6th grade', 0, 20, 10)
        ms6 = st.sidebar.slider('Biology 6th grade', 0, 20, 10)
        mm5 = st.sidebar.slider('Mathematics 7th', 0, 20, 10)
        mpc5 = st.sidebar.slider('PCT 7th grade', 0, 20, 10)
        ms5 = st.sidebar.slider('Biology 7th grade', 0, 20, 10)
        mm4 = st.sidebar.slider('Mathematics 8th grade', 0, 20, 10)
        mpc4 = st.sidebar.slider('PCT 8th grade', 0, 20, 10)
        ms4 = st.sidebar.slider('Biology 8th grade', 0, 20, 10)
        mm3 = st.sidebar.slider('Mathematics 9th grade', 0, 20, 10)
        mpc3 = st.sidebar.slider('PCT 9th grade', 0, 20, 10)
        ms3 = st.sidebar.slider('Biology 9th grade', 0, 20, 10)
        moyennes = {
            'mm6': mm6,
            'mpc6': mpc6,
            'ms6': ms6,
            'mm5': mm5,
            'mpc5': mpc5,
            'ms5': ms5,
            'mm4': mm4,
            'mpc4': mpc4,
            'ms4': ms4,
            'mm3': mm3,
            'mpc3': mpc3,
            'ms3': ms3,
        }
        moyennes_entrees = pd.DataFrame(moyennes, index=[0])
        return moyennes_entrees

df = user_input()

st.subheader('Averages inserted')
st.write(df)

dataa = pd.read_csv('MoyennesBinSeptembre22.csv',encoding='latin-1')
datas = pd.read_csv('MoyennesBinairesSeptembre22.csv',encoding='latin-1')

Xa = dataa.iloc[:, 0:12]
ya = dataa.iloc[:, 13]
Xs = datas.iloc[:, 0:12]
ys = datas.iloc[:, 13]
Xm = datam.iloc[:, 0:12]
ym = datam.iloc[:, 12]


lsvc = svm.SVC(kernel='linear')
rfc = RandomForestClassifier(n_estimators=100, oob_score=True)
lr = linear_model.LinearRegression()
modela = rfc.fit(Xa, ya)
models = lsvc.fit(Xs, ys)
modelm = lr.fit(Xm, ym)

predictiona = modela.predict(df)
predictions = models.predict(df)

col1, col2 = st.columns(2)

with col1:
    st.header("Ability :")
    if predictiona == 0:
        st.write('Literary')
    else:
        st.write('Scientific')

with col2:
    st.header("Streams:")
    if predictions == 0:
        st.write('Literature stream')
    if predictions == 1:
        st.write('Biology stream')
    if predictions == 2:
        st.write('MPS stream')


