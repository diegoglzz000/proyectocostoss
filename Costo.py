import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

st.title("Predicción del costo de una actividad")

st.write("Introduce los datos para predecir el costo.")

def user_input():
    presupuesto = st.number_input("Presupuesto:", min_value=0.0, step=1.0)
    tiempo = st.number_input("Tiempo invertido (min):", min_value=0.0, step=1.0)
    tipo = st.number_input("Tipo de actividad:", min_value=0.0, step=1.0)
    momento = st.number_input("Momento del día:", min_value=0.0, step=1.0)
    personas = st.number_input("No. de personas:", min_value=0.0, step=1.0)

    data = {
        'Presupuesto': presupuesto,
        'Tiempo invertido': tiempo,
        'Tipo': tipo,
        'Momento': momento,
        'No. de personas': personas
    }

    return pd.DataFrame(data, index=[0])

df_user = user_input()

datos = pd.read_csv("registrosdgg_limpio.csv")

datos = datos.drop(columns=['Fecha (dd/mm/aa)', 'Nombre actividad', 'Número'], errors='ignore')

X = datos[['Presupuesto', 'Tiempo invertido', 'Tipo', 'Momento', 'No. de personas']]
y = datos['Costo']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1613808)

modelo = LinearRegression()
modelo.fit(X_train, y_train)

pred = modelo.predict(df_user)

st.subheader("Resultado")
st.write("El costo estimado de la actividad es:", pred[0])


