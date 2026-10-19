
# ============================================================
# PRÁCTICA 4: GENERACIÓN DE DATOS SINTÉTICOS CON FAKER
# Y PARTICIÓN ESTRATIFICADA TRAIN-TEST
# ============================================================

# Librerías para generación y manipulación de datos
from pathlib import Path
from faker import Faker
import pandas as pd
import numpy as np
import yaml

# Herramienta para dividir los datos en entrenamiento y prueba
from sklearn.model_selection import train_test_split


# ============================================================
# 1. CONFIGURACIÓN DEL PROYECTO
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

# Cargamos la configuración desde YAML
with open(BASE_DIR / "config.yaml", "r", encoding="utf-8") as archivo:
    config = yaml.safe_load(archivo)

# Parámetros de generación
total_registros = config["dataset"]["total_registros"]
idioma = config["dataset"]["idioma"]
semilla = config["dataset"]["semilla"]

estados_civiles = config["dataset"]["estados_civiles"]
probabilidades_churn = config["dataset"]["churn"]

ingreso_minimo = config["dataset"]["ingreso_anual"]["minimo"]
ingreso_maximo = config["dataset"]["ingreso_anual"]["maximo"]

# Parámetros de partición
porcentaje_prueba = config["particion"]["porcentaje_prueba"]
estratificada = config["particion"]["estratificada"]

# Parámetros de codificación
metodo_codificacion = config["codificacion"]["metodo"]

# Parámetros de salida
directorio_resultados = BASE_DIR / config["resultados"]["directorio"]
guardar_csv = config["resultados"]["guardar_csv"]

directorio_resultados.mkdir(parents=True, exist_ok=True)

# Configuración de visualización en terminal
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 160)
pd.set_option("display.max_colwidth", 35)

print("=" * 80)
print("PRÁCTICA 4: GENERACIÓN DE DATOS SINTÉTICOS Y PARTICIÓN ESTRATIFICADA")
print("=" * 80)

print("\n=== 1. CONFIGURACIÓN DEL PROYECTO ===")
print(f"Total de registros: {total_registros}")
print(f"Localización Faker: {idioma}")
print(f"Semilla aleatoria: {semilla}")
print(f"Porcentaje de prueba: {porcentaje_prueba * 100:.0f}%")
print(f"Partición estratificada: {estratificada}")
print(f"Codificación: {metodo_codificacion}")


# ============================================================
# 2. INICIALIZACIÓN DE FAKER Y SEMILLAS ALEATORIAS
# ============================================================

print("\n=== 2. CONFIGURACIÓN DE SEMILLAS ALEATORIAS ===")

# Faker genera información ficticia con formato mexicano.
fake = Faker(idioma)

# Ambas semillas permiten reproducir los mismos resultados
# al ejecutar el programa con la misma configuración.
Faker.seed(semilla)
np.random.seed(semilla)

print(f"[OK] Faker configurado con localización {idioma}")
print(f"[OK] Faker.seed({semilla})")
print(f"[OK] np.random.seed({semilla})")


# ============================================================
# 3. GENERACIÓN DE 1,000 REGISTROS SINTÉTICOS
# ============================================================

print("\n=== 3. GENERACIÓN DEL DATASET SINTÉTICO ===")

registros = []

# Obtenemos las categorías y probabilidades desde YAML.
categorias_estado = list(estados_civiles.keys())
probabilidades_estado = list(estados_civiles.values())

categorias_churn = [0, 1]
probabilidades_abandono = [
    probabilidades_churn["no_abandona"],
    probabilidades_churn["abandona"]
]

for i in range(total_registros):

    # Cada registro representa un cliente completamente ficticio.
    cliente = {
        "id_cliente": i + 1,

        # Datos sintéticos generados mediante Faker
        "nombre": fake.name(),
        "email": fake.email(),

        # Campos adicionales solicitados por la práctica
        "telefono": fake.phone_number(),
        "direccion": fake.address().replace("\n", ", "),

        # Variables generadas con NumPy
        "estado_civil": np.random.choice(
            categorias_estado,
            p=probabilidades_estado
        ),

        "ingreso_anual": round(
            np.random.uniform(
                ingreso_minimo,
                ingreso_maximo
            ),
            2
        ),

        # Variable objetivo: abandono del servicio
        # 0 = No abandona
        # 1 = Abandona
        "churn": np.random.choice(
            categorias_churn,
            p=probabilidades_abandono
        )
    }

    registros.append(cliente)

# Convertimos la lista de clientes a un DataFrame.
df_sintetico = pd.DataFrame(registros)

print("\nPrimeros 10 registros:")
print(df_sintetico.head(10).to_string(index=False))

print("\nDimensiones del dataset:")
print(df_sintetico.shape)

print("\nColumnas generadas:")
print(df_sintetico.columns.tolist())

print("\nTipos de datos:")
print(df_sintetico.dtypes.to_string())


# ============================================================
# 4. ANÁLISIS DE DISTRIBUCIÓN DE CHURN
# ============================================================

print("\n=== 4. DISTRIBUCIÓN ORIGINAL DE CHURN ===")

conteo_churn = df_sintetico["churn"].value_counts().sort_index()

porcentaje_churn = (
    df_sintetico["churn"]
    .value_counts(normalize=True)
    .sort_index() * 100
)

tabla_churn = pd.DataFrame({
    "Cantidad": conteo_churn,
    "Porcentaje (%)": porcentaje_churn.round(2)
})

tabla_churn.index.name = "Churn"

print(tabla_churn.to_string())

print(
    "\nProporción total de abandono: "
    f"{df_sintetico['churn'].mean() * 100:.2f}%"
)


# ============================================================
# 5. SEPARACIÓN DE CARACTERÍSTICAS X Y OBJETIVO y
# ============================================================

print("\n=== 5. SEPARACIÓN DE FEATURES (X) Y TARGET (y) ===")

# Excluimos los identificadores y datos personales ficticios
# porque no aportan al ejercicio de predicción.
#
# También excluimos churn de X porque es la variable objetivo.
X = df_sintetico.drop(
    columns=[
        "id_cliente",
        "nombre",
        "email",
        "telefono",
        "direccion",
        "churn"
    ]
)

# Variable objetivo: indica si el cliente abandona.
y = df_sintetico["churn"]

print("\nVariables predictoras X:")
print(X.head(10).to_string(index=False))

print("\nVariable objetivo y:")
print(y.head(10).to_string(index=False))

print("\nDimensiones de X:", X.shape)
print("Dimensiones de y:", y.shape)


# ============================================================
# 6. PARTICIÓN ESTRATIFICADA TRAIN-TEST
# ============================================================

print("\n=== 6. PARTICIÓN TRAIN-TEST ===")

# stratify=y conserva aproximadamente la proporción
# de clientes con churn=0 y churn=1 en ambos conjuntos.
#
# random_state fija la aleatoriedad de la partición.
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=porcentaje_prueba,
    random_state=semilla,
    stratify=y if estratificada else None
)

print("\nDimensiones de los conjuntos:")

print(f"X_train: {X_train.shape}")
print(f"X_test:  {X_test.shape}")
print(f"y_train: {y_train.shape}")
print(f"y_test:  {y_test.shape}")

print("\nProporciones de abandono:")

print(f"Dataset completo: {y.mean() * 100:.2f}%")
print(f"Entrenamiento:    {y_train.mean() * 100:.2f}%")
print(f"Prueba:           {y_test.mean() * 100:.2f}%")


# ============================================================
# 7. TABLA COMPARATIVA DE PROPORCIONES
# ============================================================

print("\n=== 7. COMPARACIÓN DE DISTRIBUCIONES DE CHURN ===")

comparacion_churn = pd.DataFrame({
    "Conjunto": [
        "Dataset completo",
        "Entrenamiento",
        "Prueba"
    ],

    "Registros": [
        len(y),
        len(y_train),
        len(y_test)
    ],

    "Churn 0 (%)": [
        (y == 0).mean() * 100,
        (y_train == 0).mean() * 100,
        (y_test == 0).mean() * 100
    ],

    "Churn 1 (%)": [
        (y == 1).mean() * 100,
        (y_train == 1).mean() * 100,
        (y_test == 1).mean() * 100
    ]
})

print(
    comparacion_churn
    .round(2)
    .to_string(index=False)
)


# ============================================================
# 8. CODIFICACIÓN DE VARIABLES CATEGÓRICAS
# ============================================================

print("\n=== 8. CODIFICACIÓN DE VARIABLES CATEGÓRICAS ===")

if metodo_codificacion != "one_hot":
    raise ValueError(
        "El método de codificación configurado no es compatible."
    )

# Convertimos estado_civil en columnas numéricas.
#
# Se ajusta la codificación sobre el conjunto de entrenamiento
# y se aplica el mismo esquema al conjunto de prueba.
X_train_codificado = pd.get_dummies(
    X_train,
    columns=["estado_civil"],
    dtype=int
)

X_test_codificado = pd.get_dummies(
    X_test,
    columns=["estado_civil"],
    dtype=int
)

# Aseguramos que ambos conjuntos tengan las mismas columnas.
X_test_codificado = X_test_codificado.reindex(
    columns=X_train_codificado.columns,
    fill_value=0
)

print("\nColumnas después de la codificación:")
print(X_train_codificado.columns.tolist())

print("\nPrimeros registros de entrenamiento:")
print(
    X_train_codificado
    .head(10)
    .to_string(index=False)
)

print("\nPrimeros registros de prueba:")
print(
    X_test_codificado
    .head(10)
    .to_string(index=False)
)


# ============================================================
# 9. VALIDACIÓN DE RESULTADOS
# ============================================================

print("\n=== 9. VALIDACIÓN DEL PROCESO ===")

# Comprobamos que se generó la cantidad solicitada.
assert len(df_sintetico) == total_registros

print("[OK] Cantidad de registros correcta.")

# Comprobamos que la partición conserva todos los registros.
assert len(X_train) + len(X_test) == total_registros

print("[OK] No se perdieron registros durante la partición.")

# Comprobamos que los conjuntos no comparten índices.
assert set(X_train.index).isdisjoint(set(X_test.index))

print("[OK] No existen registros repetidos entre Train y Test.")

# Comprobamos que las columnas codificadas coinciden.
assert list(X_train_codificado.columns) == list(
    X_test_codificado.columns
)

print("[OK] Las columnas codificadas coinciden.")

# Comprobamos que la codificación produjo datos numéricos.
assert all(
    pd.api.types.is_numeric_dtype(tipo)
    for tipo in X_train_codificado.dtypes
)

print("[OK] Las variables predictoras son numéricas.")

# Verificamos que no existen valores faltantes.
assert not df_sintetico.isna().any().any()

print("[OK] No existen valores faltantes.")


# ============================================================
# 10. EXPORTACIÓN DE RESULTADOS
# ============================================================

print("\n=== 10. EXPORTACIÓN DE ARCHIVOS CSV ===")

if guardar_csv:

    archivos = {
        "dataset_sintetico.csv": df_sintetico,
        "X_train.csv": X_train_codificado,
        "X_test.csv": X_test_codificado,
        "y_train.csv": y_train,
        "y_test.csv": y_test,
        "comparacion_churn.csv": comparacion_churn
    }

    for nombre, dataframe in archivos.items():

        ruta = directorio_resultados / nombre

        dataframe.to_csv(
            ruta,
            index=False,
            encoding="utf-8-sig"
        )

        print(f"[OK] Archivo generado: {ruta}")

else:
    print("La exportación CSV está desactivada.")


# ============================================================
# 11. RESUMEN FINAL
# ============================================================

print("\n=== 11. RESUMEN FINAL ===")

print(f"Clientes sintéticos generados: {len(df_sintetico)}")
print(f"Registros de entrenamiento: {len(X_train)}")
print(f"Registros de prueba: {len(X_test)}")

print(f"Proporción churn original: {y.mean() * 100:.2f}%")
print(f"Proporción churn Train: {y_train.mean() * 100:.2f}%")
print(f"Proporción churn Test: {y_test.mean() * 100:.2f}%")

print("\nPROCESO COMPLETADO CORRECTAMENTE")
