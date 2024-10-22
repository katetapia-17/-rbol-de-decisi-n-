import pandas as pd

# Cargar el dataset
df = pd.read_csv("winequality-red.csv")

# Mostrar las primeras filas para revisar el contenido
df.head()

# Información general del dataset
df.info()

# Resumen estadístico
df.describe()

# Verificar si hay valores nulos
df.isnull().sum()

# Eliminar filas con valores nulos 
df.dropna(inplace=True)

# Definir X (todas las columnas menos la columna objetivo) e y (la variable objetivo)
X = df.drop("quality", axis=1) 
y = df["quality"]

from sklearn.model_selection import train_test_split

# Dividir los datos en entrenamiento (80%) y prueba (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.tree import DecisionTreeClassifier

# Crear el modelo de Árbol de Decisión
tree_model = DecisionTreeClassifier(random_state=42)

# Entrenar el modelo con los datos de entrenamiento
tree_model.fit(X_train, y_train)

from sklearn.metrics import accuracy_score, classification_report

# Hacer predicciones con los datos de prueba
y_pred = tree_model.predict(X_test)

# Calcular la precisión del modelo
accuracy = accuracy_score(y_test, y_pred)
print(f"Precisión del modelo: {accuracy}")

# Reporte de clasificación (precisión, recall, F1-score)
print(classification_report(y_test, y_pred))

from sklearn import tree
import matplotlib.pyplot as plt

# Graficar el árbol de decisión
plt.figure(figsize=(20,10))
tree.plot_tree(tree_model, filled=True, feature_names=X.columns, class_names=['Clase_1', 'Clase_2', 'Clase_3', 'Clase_4', 'Clase_5', 'Clase_6', 'Clase_7', 'Clase_8', 'Clase_9', 'Clase_10'])
plt.show()
