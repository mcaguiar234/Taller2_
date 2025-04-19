# -*- coding: utf-8 -*-
"""
MAESTRÍA CIENCIA DE DATOS
FUNDAMENTOS DE CIENCIA DE DATOS
UNIVERSIDAD YACHAY TECH
FECHA: 15/04/2025
    
@author: Magaly_Aguiar
"""
#1. Instalar Scikit-Learn
#!pip install scikit-learn
import sklearn
print(f"Scikit-learn versión: {sklearn.__version__}")
    
import pandas as pd
import re
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.cluster import MeanShift
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
    
directorio0 = "C:/Users/aguia/OneDrive/MAESTRIA/FUNDAMENTOS CIENCIA DE DATOS/TALLER 3/graficos"
# =========================================
# PARTE 0: Adquisición y limpieza de datos
# =========================================

# 1. Ruta al archivo Excel
directorio = "C:/Users/aguia/OneDrive/MAESTRIA/FUNDAMENTOS CIENCIA DE DATOS/TALLER 3/online+retail+ii/online_retail_II.xlsx"
    
# 2. Leer ambas hojas directamente desde el archivo
df_2009 = pd.read_excel(directorio, sheet_name='Year 2009-2010', engine='openpyxl')
df_2010 = pd.read_excel(directorio, sheet_name='Year 2010-2011', engine='openpyxl')
    
# 3. Combinar las hojas
df_total = pd.concat([df_2009, df_2010], ignore_index=True)
    
# Vista previa
print(df_total.head())

# 4. Función que detecta si el último carácter es una letra
def clasificar_stockcode(codigo):
    if isinstance(codigo, str):
        if re.fullmatch(r'\d{5}', codigo):
            return 'Solo numérico'
        elif re.fullmatch(r'[A-Za-z]+', codigo):
            return 'Solo letras'
        elif re.fullmatch(r'\d+[A-Za-z]+', codigo):
            return 'Combinado'
    return 'Otro'

# Aplicar la clasificación
df_total['Grupo_StockCode'] = df_total['StockCode'].apply(clasificar_stockcode)

# Función para extraer letra final solo si está en grupo "Combinado"
def obtener_especificacion(codigo, grupo):
    if grupo == 'Combinado' and isinstance(codigo, str):
        match = re.search(r'([A-Za-z])$', codigo)
        if match:
            return match.group(1)
    return np.nan  # NA si no aplica

# Aplicar la función y crear la nueva columna
df_total['Especificaciones'] = df_total.apply(
    lambda row: obtener_especificacion(row['StockCode'], row['Grupo_StockCode']),
    axis=1
)

# Verificación
print(df_total[['StockCode', 'Grupo_StockCode', 'Especificaciones']].drop_duplicates().head(10))

df_total.drop(columns=['Grupo_StockCode'], inplace=True)
    
print(df_total.columns)
    
#Verificar duplicados completos

duplicados_exactos = df_total.duplicated(
    subset=['Customer ID', 'InvoiceDate', 'StockCode', 'Especificaciones', 'Quantity', 'Price'],
    keep=False
)

df_duplicados = df_total[duplicados_exactos].sort_values(by=['Customer ID', 'InvoiceDate'])
print(df_duplicados.head(10))

#Crear columna de Precio Total
df_total['TotalPrice'] = df_total['Quantity'] * df_total['Price']

# Consolidar por agrupación lógica de "mismo producto y momento"
df_consolidado = df_total.groupby(
    ['Customer ID', 'Invoice', 'InvoiceDate', 'StockCode', 'Especificaciones', 'Price'],
    as_index=False
).agg({
    'Quantity': 'sum',
    'TotalPrice': 'sum',
    'Description': 'first',  # conservar descripción
    'Country': 'first'       # conservar país
})

# Ordenar para revisar
df_consolidado = df_consolidado.sort_values(by=['Customer ID', 'InvoiceDate'])

# Vista rápida
print(df_consolidado.head(10))

#Detección de registros negativos o atípicos
# Mostrar registros con valores negativos o cero
negativos = df_consolidado[
    (df_consolidado['Quantity'] <= 0) |
    (df_consolidado['Price'] <= 0) |
    (df_consolidado['TotalPrice'] <= 0)
]

print(f"Cantidad de registros con valores negativos o cero: {negativos.shape[0]}")
print(negativos[['Customer ID', 'Invoice', 'InvoiceDate', 'StockCode', 'Quantity', 'Price', 'TotalPrice']].head())

#Creación de una columna auxiliar para identificar los valores negativos
df_consolidado['RegistroNegativo'] = (
    (df_consolidado['Quantity'] <= 0) |
    (df_consolidado['Price'] <= 0) |
    (df_consolidado['TotalPrice'] <= 0)
)

# Ver cuántos registros fueron marcados
print("Total de registros marcados como negativos:", df_consolidado['RegistroNegativo'].sum())

# Vista rápida de los primeros casos
print(df_consolidado[df_consolidado['RegistroNegativo'] == True].head())

#Creación de variables de tiempo
# Asegurar que InvoiceDate es datetime
df_consolidado['Fecha'] = pd.to_datetime(df_consolidado['InvoiceDate'])

# Extraer componentes temporales
df_consolidado['Año'] = df_consolidado['Fecha'].dt.year
df_consolidado['Mes'] = df_consolidado['Fecha'].dt.month
df_consolidado['DíaSemana'] = df_consolidado['Fecha'].dt.dayofweek  # 0=Lunes, 6=Domingo
df_consolidado['Hora'] = df_consolidado['Fecha'].dt.hour
df_consolidado['Mes_Año'] = df_consolidado['Fecha'].dt.to_period('M').astype(str)

# Indicar si es fin de semana (sábado=5 o domingo=6)
df_consolidado['Fin_de_Semana'] = df_consolidado['DíaSemana'].isin([5, 6])

#Verificación
print(df_consolidado[['InvoiceDate', 'Año', 'Mes', 'DíaSemana', 'Hora', 'Mes_Año', 'Fin_de_Semana']].head())

#Calculo total gastado por cliente
# Agrupar por cliente y calcular total gastado
gasto_por_cliente = df_consolidado.groupby('Customer ID')['TotalPrice'].sum().reset_index()

# Renombrar columna
gasto_por_cliente.rename(columns={'TotalPrice': 'TotalGastado'}, inplace=True)

# Ver resumen estadístico
print(gasto_por_cliente.describe())

# Ver top 5 clientes que más gastaron
print(gasto_por_cliente.sort_values(by='TotalGastado', ascending=False).head())

# Agregar la columna TotalGastado al df_consolidado
df_consolidado = df_consolidado.merge(gasto_por_cliente, on='Customer ID', how='left')

#Cálculo de umbrales
media_gasto = gasto_por_cliente['TotalGastado'].mean()
mediana_gasto = gasto_por_cliente['TotalGastado'].median()
q3_gasto = gasto_por_cliente['TotalGastado'].quantile(0.75)

print(f"Media: £{media_gasto:.2f}")
print(f"Mediana: £{mediana_gasto:.2f}")
print(f"Percentil 75 (Q3): £{q3_gasto:.2f}")

# Clasificación binaria basada en Q3
df_consolidado['TipoCliente'] = df_consolidado['TotalGastado'].apply(
    lambda x: 'Premium' if x > q3_gasto else 'Normal'
)

# Ver distribución
print(df_consolidado['TipoCliente'].value_counts())

print(df_consolidado[['Customer ID', 'TotalGastado', 'TipoCliente']].drop_duplicates().head())

# Calcular el umbral de gasto premium (Q3)
q3_gasto = gasto_por_cliente['TotalGastado'].quantile(0.75)

# Crear columna de clasificación binaria
df_consolidado['TipoCliente'] = df_consolidado['TotalGastado'].apply(
    lambda x: 'Premium' if x > q3_gasto else 'Normal'
)

# Ver distribución general
print(df_consolidado['TipoCliente'].value_counts())

# Vista rápida de clientes clasificados
print(df_consolidado[['Customer ID', 'TotalGastado', 'TipoCliente']].drop_duplicates().head())

# =========================================
# PARTE 1: Clasificación de clientes:
# =========================================

# Creamos la tabla base de clientes si no existe aún
df_clientes = df_consolidado.groupby('Customer ID').agg({
    'Invoice': pd.Series.nunique,
    'Quantity': 'sum',
    'TotalPrice': 'sum',
    'Fecha': lambda x: (df_consolidado['Fecha'].max() - x.max()).days
}).reset_index()

# Renombrar columnas
df_clientes.columns = ['Customer ID', 'Frecuencia', 'CantidadTotal', 'TotalGastado', 'Recencia']

# Añadir la etiqueta de TipoCliente (ya la calculamos antes)
df_clientes = df_clientes.merge(
    df_consolidado[['Customer ID', 'TipoCliente']].drop_duplicates(),
    on='Customer ID',
    how='left'
)

print(df_clientes.head())

#Convertir variable objetivo a formato binario

# Codificar la etiqueta: Normal = 0, Premium = 1
df_clientes['TipoCliente_bin'] = df_clientes['TipoCliente'].map({'Normal': 0, 'Premium': 1})

#Identificación de variables predictoras
# Variables predictoras
X = df_clientes[['Frecuencia', 'CantidadTotal', 'TotalGastado', 'Recencia']]

# Variable objetivo
y = df_clientes['TipoCliente_bin']

#División del conjunto de datos
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Escalamiento de datos
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#Entrenamiento del modelo (Árbol de decisión)
# Crear y entrenar modelo
modelo_arbol = DecisionTreeClassifier(random_state=42)
modelo_arbol.fit(X_train_scaled, y_train)

# Predecir
y_pred = modelo_arbol.predict(X_test_scaled)

# Evaluar
print(classification_report(y_test, y_pred))
print("\nMatriz de confusión:")
print(confusion_matrix(y_test, y_pred))

#Profundidad del arbol
print("Profundidad del árbol:", modelo_arbol.get_depth())

#COMPARACIÓN CON UN MODELO DE REGRESIÓN LINEAL
# Crear el modelo
modelo_logistico = LogisticRegression(random_state=42)

# Entrenar el modelo con los datos escalados
modelo_logistico.fit(X_train_scaled, y_train)

#Predición
# Predecir con el modelo
y_pred_log = modelo_logistico.predict(X_test_scaled)

# Evaluar el rendimiento
print("Reporte de clasificación (Regresión Logística):")
print(classification_report(y_test, y_pred_log))

print("Matriz de confusión:")
print(confusion_matrix(y_test, y_pred_log))

### Visualización

# Ruta de guardado
directorio0 = "C:/Users/aguia/OneDrive/MAESTRIA/FUNDAMENTOS CIENCIA DE DATOS/TALLER 3/graficos"
os.makedirs(directorio0, exist_ok=True)

matriz_arbol = confusion_matrix(y_test, y_pred)
matriz_log = confusion_matrix(y_test, y_pred_log)
# Crear figura
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Matriz Árbol de Decisión
ConfusionMatrixDisplay(matriz_arbol, display_labels=["Normal", "Premium"]).plot(ax=axes[0], cmap="Blues", values_format="d")
axes[0].set_title("Árbol de Decisión")

# Matriz Regresión Logística
ConfusionMatrixDisplay(matriz_log, display_labels=["Normal", "Premium"]).plot(ax=axes[1], cmap="Oranges", values_format="d")
axes[1].set_title("Regresión Logística")

plt.tight_layout()
plt.savefig(os.path.join(directorio0, "matrices_confusion_comparadas.png"))
plt.close()

##Generar y guardar la curva ROC
# Obtener probabilidades para clase positiva
y_scores_arbol = modelo_arbol.predict_proba(X_test_scaled)[:, 1]
y_scores_log = modelo_logistico.predict_proba(X_test_scaled)[:, 1]

# Calcular fpr (false positive rate) y tpr (true positive rate)
fpr_arbol, tpr_arbol, _ = roc_curve(y_test, y_scores_arbol)
fpr_log, tpr_log, _ = roc_curve(y_test, y_scores_log)

# Calcular AUC
auc_arbol = auc(fpr_arbol, tpr_arbol)
auc_log = auc(fpr_log, tpr_log)

# Graficar
plt.figure(figsize=(8, 6))
plt.plot(fpr_arbol, tpr_arbol, label=f"Árbol de Decisión (AUC = {auc_arbol:.2f})", color='blue')
plt.plot(fpr_log, tpr_log, label=f"Regresión Logística (AUC = {auc_log:.2f})", color='orange', linestyle='--')
plt.plot([0, 1], [0, 1], 'k--', label='Aleatorio (AUC = 0.5)')
plt.xlabel('Tasa de Falsos Positivos')
plt.ylabel('Tasa de Verdaderos Positivos')
plt.title('Curva ROC Comparativa')
plt.legend(loc='lower right')
plt.grid(True)
plt.tight_layout()

# Guardar el gráfico
plt.savefig(os.path.join(directorio0, "curvas_roc_comparadas.png"))
plt.close()

# =========================================
# PARTE 2: Segmentación de clientes:
# =========================================

# Reutilizar la estructura de clientes
df_segmentacion = df_clientes[['Customer ID', 'Frecuencia', 'CantidadTotal', 'TotalGastado', 'Recencia']].copy()

#Escalar las variables
#Los algoritmos de clustering requieren que las variables estén en una misma escala:

# Normalizar las variables numéricas
scaler = StandardScaler()
X_seg = scaler.fit_transform(df_segmentacion[['Frecuencia', 'CantidadTotal', 'TotalGastado', 'Recencia']])

#Agrupamiento de K-MEANS
# Definir y entrenar el modelo con 4 clusters (puedes variar el número)
kmeans = KMeans(n_clusters=4, random_state=42)
df_segmentacion['Cluster_KMeans'] = kmeans.fit_predict(X_seg)

# Ver distribución de clientes por grupo
print(df_segmentacion['Cluster_KMeans'].value_counts())

#Agrupamiento con Mean Shift
meanshift = MeanShift()
df_segmentacion['Cluster_MeanShift'] = meanshift.fit_predict(X_seg)

# Ver distribución de clientes por grupo
print(df_segmentacion['Cluster_MeanShift'].value_counts())

#Análisis descriptivo por grupo

print("Agrupamiento por K-Means:")
print(df_segmentacion.groupby('Cluster_KMeans')[['Frecuencia', 'CantidadTotal', 'TotalGastado', 'Recencia']].mean())

# Mean Shift
print("\nAgrupamiento por Mean Shift:")
print(df_segmentacion.groupby('Cluster_MeanShift')[['Frecuencia', 'CantidadTotal', 'TotalGastado', 'Recencia']].mean())

# Visualización con PCA y guardado de gráficos

# Aplicar PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_seg)

# Crear DataFrame para graficar
df_plot = pd.DataFrame(X_pca, columns=['PCA1', 'PCA2'])
df_plot['Cluster_KMeans'] = df_segmentacion['Cluster_KMeans']
df_plot['Cluster_MeanShift'] = df_segmentacion['Cluster_MeanShift']

# Gráfico K-Means
plt.figure(figsize=(12, 6))
for cluster in sorted(df_plot['Cluster_KMeans'].unique()):
    subset = df_plot[df_plot['Cluster_KMeans'] == cluster]
    plt.scatter(subset['PCA1'], subset['PCA2'], label=f'Cluster {cluster}', alpha=0.6)
plt.title('Visualización de Clusters (K-Means) con PCA')
plt.xlabel('Componente Principal 1')
plt.ylabel('Componente Principal 2')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.subplots_adjust(top=0.92)
plt.savefig(os.path.join(directorio0, "Clusters_KMeans_PCA.png"))
plt.close()

# Gráfico Mean Shift
plt.figure(figsize=(12, 6))
for cluster in sorted(df_plot['Cluster_MeanShift'].unique()):
    subset = df_plot[df_plot['Cluster_MeanShift'] == cluster]
    plt.scatter(subset['PCA1'], subset['PCA2'], label=f'Cluster {cluster}', alpha=0.6)
plt.title('Visualización de Clusters (Mean Shift) con PCA')
plt.xlabel('Componente Principal 1')
plt.ylabel('Componente Principal 2')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.subplots_adjust(top=0.92)
plt.savefig(os.path.join(directorio0, "Clusters_MeanShift_PCA.png"))
plt.close()


# =========================================
# PARTE 3: Predicción de ventas:
# =========================================

#Haciendo uso del apartado "Extraer componentes temporales" de la sección 0
# Variable objetivo: agregamos ventas por cliente + mes
df_ventas = df_consolidado.groupby(['Customer ID', 'Mes_Año']).agg({
    'Quantity': 'sum',
    'TotalPrice': 'sum'
}).reset_index()

# Renombrar columna objetivo
df_ventas.rename(columns={'TotalPrice': 'VentasMensuales'}, inplace=True)

# Extraer mes y año como variables numéricas
df_ventas['MesNum'] = pd.to_datetime(df_ventas['Mes_Año']).dt.month
df_ventas['AñoNum'] = pd.to_datetime(df_ventas['Mes_Año']).dt.year

#Preparar variables para modelamiento
# Variables predictoras
X = df_ventas[['MesNum', 'AñoNum', 'Quantity']]

# Variable objetivo
y = df_ventas['VentasMensuales']

#División del dataset para entrenamiento y prueba
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Entrenamiento del modelo
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Crear el modelo
modelo_arbol = DecisionTreeRegressor(random_state=42)

# Entrenar
modelo_arbol.fit(X_train, y_train)

# Predecir
y_pred = modelo_arbol.predict(X_test)

# Evaluar el modelo
mae = mean_absolute_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)
r2 = r2_score(y_test, y_pred)

print("MAE:", mae)
print("RMSE:", rmse)
print("R²:", r2)

#Prueba con otro modelo: Random Forest
from sklearn.ensemble import RandomForestRegressor

modelo_rf = RandomForestRegressor(random_state=42)
modelo_rf.fit(X_train, y_train)
y_pred_rf = modelo_rf.predict(X_test)

print("Random Forest R²:", r2_score(y_test, y_pred_rf))

#Visualización
# Crear DataFrame con resultados
df_resultados = pd.DataFrame({
    'y_real': y_test,
    'y_predicho_arbol': y_pred,
    'y_predicho_rf': y_pred_rf
}).reset_index(drop=True)

# Gráfico comparativo Árbol de Decisión
plt.figure(figsize=(10, 5))
plt.plot(df_resultados['y_real'].values, label='Real', marker='o')
plt.plot(df_resultados['y_predicho_arbol'].values, label='Árbol de Decisión', marker='x')
plt.title('Comparación de Ventas Reales vs Predichas (Árbol de Decisión)')
plt.xlabel('Observaciones')
plt.ylabel('Ventas Mensuales')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(directorio0, "Comparacion_Arbol_Predicciones.png"))
plt.close()

# Gráfico comparativo Random Forest
plt.figure(figsize=(10, 5))
plt.plot(df_resultados['y_real'].values, label='Real', marker='o')
plt.plot(df_resultados['y_predicho_rf'].values, label='Random Forest', marker='x')
plt.title('Comparación de Ventas Reales vs Predichas (Random Forest)')
plt.xlabel('Observaciones')
plt.ylabel('Ventas Mensuales')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(directorio0, "Comparacion_RF_Predicciones.png"))
plt.close()