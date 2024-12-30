import streamlit as st
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Configura el backend
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error
import numpy as np
from io import BytesIO

# A continuacion se va a implemtar las funciones mas importantes a utilizar, con una breve explicacion de lo que hace cada una de ellas.
# Aplicamos programacion modular y orientada a objetos (POO)
# Función para generar estadísticas descriptivas
def generar_estadisticas(df):
    """
    Genera estadísticas descriptivas de un dataframe
    """
    return df.describe()

# Función para exportar resultados a Excel
def exportar_resultados_excel(df, estadisticas, graficos):
    """
    Exporta estadísticas descriptivas y gráficos a un archivo Excel
    """
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, sheet_name="Datos Originales", index=False)
        estadisticas.to_excel(writer, sheet_name="Estadísticas", index=False)
        for nombre, figura in graficos.items():
            hoja = writer.book.add_worksheet(nombre)
            hoja.insert_image("A1", nombre, {"image_data": figura})
    output.seek(0)
    return output

# Función para generar gráficos estadísticos
def generar_grafico(df, tipo_grafico, x_col=None, y_col=None):
    """
    Genera gráficos estadísticos según el tipo especificado y retorna análisis del gráfico
    """
    figura = BytesIO()
    plt.figure(figsize=(10, 6))
    analisis = ""
    try:
        if tipo_grafico == "Histograma" and x_col:
            sns.histplot(df[x_col], kde=True)
            analisis = f"Histograma generado para la columna {x_col}. Media: {df[x_col].mean():.2f}, Desviación estándar: {df[x_col].std():.2f}"
        elif tipo_grafico == "Dispersión" and x_col and y_col:
            sns.scatterplot(x=df[x_col], y=df[y_col])
            correlacion = df[x_col].corr(df[y_col])
            analisis = f"Gráfico de dispersión entre {x_col} y {y_col}. Correlación: {correlacion:.2f}"
        elif tipo_grafico == "Barras" and x_col:
            sns.countplot(x=df[x_col])
            analisis = f"Gráfico de barras para la columna {x_col}. Valores únicos: {df[x_col].nunique()}"
        elif tipo_grafico == "Circular" and x_col:
            df[x_col].value_counts().plot.pie(autopct="%1.1f%%")
            analisis = f"Gráfico circular para la columna {x_col}. Distribución porcentual generada."
        elif tipo_grafico == "Lineal" and x_col and y_col:
            sns.lineplot(x=df[x_col], y=df[y_col])
            analisis = f"Gráfico lineal entre {x_col} y {y_col}."
        elif tipo_grafico == "Cajas" and x_col:
            sns.boxplot(x=df[x_col])
            analisis = f"Diagrama de cajas generado para la columna {x_col}. Detecta valores atípicos y resume la distribución."
        elif tipo_grafico == "Pareto" and x_col:
            conteo = df[x_col].value_counts().sort_values(ascending=False)
            porcentaje_acumulado = conteo.cumsum() / conteo.sum() * 100
            fig, ax1 = plt.subplots()
            ax1.bar(conteo.index, conteo.values, color="C0")
            ax2 = ax1.twinx()
            ax2.plot(conteo.index, porcentaje_acumulado, color="C1", marker="D", ms=5)
            ax1.set_ylabel("Frecuencia")
            ax2.set_ylabel("Porcentaje acumulado (%)")
            analisis = f"Diagrama de Pareto para la columna {x_col}. Muestra los valores más frecuentes y su acumulado porcentual."
        else:
            st.error("Seleccione columnas válidas para el gráfico")
            return None, None
        plt.title(f"Gráfico de {tipo_grafico}")
        plt.tight_layout()
        plt.savefig(figura, format="png")
        figura.seek(0)
        st.pyplot(plt)
    except Exception as e:
        st.error(f"Error al generar el gráfico: {e}")
        return None, None

    return figura, analisis

# Función para realizar regresiones

def realizar_regresion(df, x_col, y_col, tipo_modelo="Lineal"):
    """
    Realiza un modelo de regresión y muestra resultados
    """
    try:
        X = df[[x_col]].values.reshape(-1, 1)
        y = df[y_col].values

        if tipo_modelo == "Lineal":
            modelo = LinearRegression()
        elif tipo_modelo == "Ridge":
            modelo = Ridge()
        elif tipo_modelo == "Lasso":
            modelo = Lasso()
        else:
            st.error("Modelo no soportado")
            return None

        modelo.fit(X, y)

        predicciones = modelo.predict(X)
        mse = mean_squared_error(y, predicciones)

        figura = BytesIO()
        plt.figure(figsize=(10, 6))
        plt.scatter(X, y, label="Datos originales")
        plt.plot(X, predicciones, color="red", label=f"Regresión {tipo_modelo}")
        plt.xlabel(x_col)
        plt.ylabel(y_col)
        plt.legend()
        plt.title(f"Regresión {tipo_modelo}")
        plt.tight_layout()
        plt.savefig(figura, format="png")
        figura.seek(0)
        st.pyplot(plt)

        analisis = f"### Resultados de la regresión {tipo_modelo}\n"
        if tipo_modelo != "Lasso":
            analisis += f"Coeficiente: {modelo.coef_[0]:.4f}\n"
        analisis += f"Intercepto: {modelo.intercept_:.4f}\nError cuadrático medio: {mse:.4f}"

        st.markdown(analisis.replace("\n", "  \n"))
    except Exception as e:
        st.error(f"Error al realizar la regresión: {e}")
        return None

    return figura

# Interfaz de Streamlit
st.title("Aplicacion interactiva de analisis de datos, por Jose Hidalgo Torres")
# 1. Carga de Datasets: Aqui podremos cargar nuestro archivo en los formatos programados
archivo_subido = st.file_uploader("Suba su archivo Excel o CSV a analizar", type=["xlsx", "xls", "csv"])

if archivo_subido is not None:
    st.write("El archivo ha sido cargado correctamente.")

    if archivo_subido.name.endswith("csv"):
        df = pd.read_csv(archivo_subido)
    else:
        df = pd.read_excel(archivo_subido)

    st.write("### Dataframe Original")
    st.dataframe(df)

    # Filtrar columnas numéricas, es importante para que el aplicativo pueda tomar en cuenta las columnas adecuadas para generar diversos reportes
    columnas_numericas = df.select_dtypes(include=[np.number]).columns.tolist()
    columnas_categoricas = df.select_dtypes(exclude=[np.number]).columns.tolist()

    # Mostrar estadísticas descriptivas
    st.write("### Estadísticas Descriptivas")
    df_estadisticas = generar_estadisticas(df)
    st.dataframe(df_estadisticas)

    if "graficos" not in st.session_state:
        st.session_state["graficos"] = {}

    # Gráficos estadísticos
    # La idea es que el usuario pueda escoger que tipo de grafico estadistico quiere generar, y que, ademas, quede guardado en memoria para despues incluir todos los graficos generados
    # en el reporte final que el usuario se puede descargar en formato xlsx
    st.write("### Gráficos Estadísticos")
    tipo_grafico = st.selectbox("Seleccione el tipo de gráfico", ["Histograma", "Dispersión", "Barras", "Circular", "Lineal", "Cajas", "Pareto"])

    if tipo_grafico in ["Histograma", "Lineal", "Dispersión", "Cajas"]:
        x_col = st.selectbox("Seleccione la columna X", columnas_numericas)
    else:
        x_col = st.selectbox("Seleccione la columna X", columnas_categoricas + columnas_numericas)

    y_col = None
    if tipo_grafico in ["Dispersión", "Lineal"]:
        y_col = st.selectbox("Seleccione la columna Y", columnas_numericas)

    if st.button("Generar Gráfico"):
        figura, analisis = generar_grafico(df, tipo_grafico, x_col, y_col)
        if figura:
            st.session_state["graficos"][f"Gráfico_{tipo_grafico}_{len(st.session_state['graficos']) + 1}"] = figura
            st.write(analisis)

    # Mostrar todos los gráficos generados
    for nombre, figura in st.session_state["graficos"].items():
        st.write(f"#### {nombre}")
        st.image(figura)

    # Modelos de regresión
    # Aqui se da la posibilidad de seleccionar las columnas (las que lo permitan), y el modelo de regresion a calcular y presentar)
    st.write("### Modelos de Regresión")
    tipo_modelo = st.selectbox("Seleccione el modelo de regresión", ["Lineal", "Ridge", "Lasso"])
    x_col = st.selectbox("Seleccione la variable independiente (X)", columnas_numericas, key="regresion_x")
    y_col = st.selectbox("Seleccione la variable dependiente (Y)", columnas_numericas, key="regresion_y")
    if st.button("Realizar Modelo de Regresión"):
        figura = realizar_regresion(df, x_col, y_col, tipo_modelo)
        st.session_state["graficos"][f"Regresión {tipo_modelo}"] = figura

    # Descargar resultados
    # Aqui podremos descargar todos los graficos y analisis presentados en la pagina, y exportarlos a un archivo de Excel.
    st.write("### Descargar Resultados")
    resultados_excel = exportar_resultados_excel(df, df_estadisticas, st.session_state["graficos"])
    st.download_button(
        label="Descargar todos los resultados en Excel",
        data=resultados_excel,
        file_name="resultados_analisis.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

else:
    st.write("Por favor, cargue un archivo para analizar.")
