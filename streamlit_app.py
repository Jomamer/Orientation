import streamlit as st
import pandas as pd
!pip install sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn import neighbors, linear_model

st.write('''
# Prédiction des orientations académiques
Cette application prédit l'aptitude, la série et la moyenne annuelle de l'apprenant''')

st.sidebar.header("Insérer les moyennes de l'apprenant")
def user_input():
        mm6 = st.sidebar.slider('Maths 6e', 0, 20, 10)
        mpc6 = st.sidebar.slider('PCT 6e', 0, 20, 10)
        ms6 = st.sidebar.slider('SVT 6e', 0, 20, 10)
        mm5 = st.sidebar.slider('Math 5e', 0, 20, 10)
        mpc5 = st.sidebar.slider('PCT 5e', 0, 20, 10)
        ms5 = st.sidebar.slider('SVT 5e', 0, 20, 10)
        mm4 = st.sidebar.slider('Maths 4e', 0, 20, 10)
        mpc4 = st.sidebar.slider('PCT 4e', 0, 20, 10)
        ms4 = st.sidebar.slider('SVT 4e', 0, 20, 10)
        mm3 = st.sidebar.slider('Maths 3e', 0, 20, 10)
        mpc3 = st.sidebar.slider('PCT 3e', 0, 20, 10)
        ms3 = st.sidebar.slider('SVT 3e', 0, 20, 10)
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

st.subheader('Moyennes insérées')
st.write(df)

dataa = pd.read_csv('MoyennesBinNew.csv',encoding='latin-1')
datas = pd.read_csv('MoyennesBinairesNew.csv',encoding='latin-1')
datam = pd.read_csv('MoyennesFinalesNew.csv',encoding='latin-1')

Xa = dataa.iloc[:, 0:12]
ya = dataa.iloc[:, 13]
Xs = datas.iloc[:, 0:12]
ys = datas.iloc[:, 13]
Xm = datam.iloc[:, 0:12]
ym = datam.iloc[:, 12]


knn = neighbors.KNeighborsClassifier(n_neighbors=15)
rfc = RandomForestClassifier(n_estimators=100, oob_score=True)
lr = linear_model.LinearRegression()
modela = rfc.fit(Xa, ya)
models = rfc.fit(Xs, ys)
modelm = lr.fit(Xm, ym)


predictiona = modela.predict(df)
predictions = models.predict(df)
predictionm = modelm.predict(df)

col1, col2, col3 = st.columns(3)

with col1:
    st.header("Aptitude :")
    if predictiona == 0:
        st.write('Littéraire')
    else:
        st.write('Scientifique')

with col2:
    st.header("Séries:")
    if predictions == 0:
        st.write('A')
    if predictions == 1:
        st.write('D')
    if predictions == 2:
        st.write('C')

with col3:
    st.header('Moyenne annuelle:')
    st.write(predictionm)


st.balloons()
