
# Práctica 4: Generación de Datos Sintéticos y Partición Estratificada

Implementación de un proceso de **generación de datos sintéticos de clientes mexicanos** utilizando Python y Faker, seguido de una partición de entrenamiento y prueba mediante `train_test_split` con estratificación.

El proyecto permite analizar cómo conservar la distribución de una variable objetivo (`churn`) al dividir un conjunto de datos para tareas de aprendizaje automático.

## 1. Objetivo

Generar un conjunto de 1,000 registros sintéticos utilizando Faker y NumPy, dividirlo en conjuntos de entrenamiento y prueba con una proporción 80/20, y verificar que la estratificación conserve aproximadamente la distribución de la variable objetivo.

Los objetivos específicos son:

- Generar información ficticia con localización `es_MX`.
- Utilizar semillas aleatorias para obtener resultados reproducibles.
- Incorporar campos adicionales de teléfono y dirección.
- Crear una variable objetivo binaria denominada `churn`.
- Separar las características predictoras (`X`) de la variable objetivo (`y`).
- Aplicar una partición estratificada de entrenamiento y prueba.
- Codificar variables categóricas mediante One-Hot Encoding.
- Comparar las proporciones de `churn` entre los conjuntos.
- Exportar los resultados a archivos CSV.

## 2. Tecnologías utilizadas

| Tecnología | Función |
|---|---|
| Python | Desarrollo del proceso |
| Faker | Generación de datos ficticios mexicanos |
| Pandas | Creación y manipulación de DataFrames |
| NumPy | Generación de valores y probabilidades |
| Scikit-learn | División estratificada Train-Test |
| PyYAML | Lectura de parámetros de configuración |
| AWS EC2 | Ejecución del programa en Ubuntu |
| Git y GitHub | Control de versiones y publicación |

## 3. Estructura del proyecto

```text
practica4/
├── practica4.py
├── config.example.yaml
├── requirements.txt
├── README.md
├── .gitignore
└── resultados/
    ├── dataset_sintetico.csv
    ├── X_train.csv
    ├── X_test.csv
    ├── y_train.csv
    ├── y_test.csv
    └── comparacion_churn.csv
```

El archivo `config.yaml` se genera localmente a partir de `config.example.yaml` y está excluido del repositorio mediante `.gitignore`.

## 4. Configuración del proyecto

La configuración se administra mediante YAML para separar los parámetros del código fuente.

| Parámetro | Valor |
|---|---|
| Total de registros | 1,000 |
| Localización Faker | `es_MX` |
| Semilla aleatoria | 42 |
| Porcentaje de entrenamiento | 80% |
| Porcentaje de prueba | 20% |
| Estratificación | Activada |
| Codificación | One-Hot |
| Probabilidad configurada de churn | 20% |

Se establecieron probabilidades de 50% para Soltero, 40% para Casado y 10% para Divorciado.

Los ingresos anuales se generan aleatoriamente dentro del intervalo de $100,000 a $1,000,000.

## 5. Generación del dataset sintético

Se utilizó Faker con localización `es_MX` para generar información ficticia de clientes.

El dataset contiene ocho columnas:

| Columna | Descripción |
|---|---|
| `id_cliente` | Identificador consecutivo |
| `nombre` | Nombre ficticio |
| `email` | Correo electrónico ficticio |
| `telefono` | Número telefónico ficticio |
| `direccion` | Dirección ficticia |
| `estado_civil` | Estado civil generado aleatoriamente |
| `ingreso_anual` | Ingreso anual sintético |
| `churn` | Variable objetivo binaria |

La variable `churn` representa:

- `0`: el cliente no abandona el servicio.
- `1`: el cliente abandona el servicio.

Se generaron correctamente **1,000 registros y 8 columnas**, sin valores faltantes.

## 6. Distribución original de churn

Se configuró una probabilidad de abandono del 20%.

La distribución observada en la ejecución fue:

| Churn | Cantidad | Porcentaje |
|---|---:|---:|
| 0 | 808 | 80.80% |
| 1 | 192 | 19.20% |
| **Total** | **1,000** | **100%** |

La proporción obtenida es cercana al 20% esperado. No es exactamente igual porque la variable se genera mediante muestreo aleatorio.

## 7. Separación de características y variable objetivo

Se separaron las variables predictoras (`X`) de la variable objetivo (`y`).

Las características utilizadas fueron:

- `estado_civil`
- `ingreso_anual`

La variable objetivo fue `churn`.

Los identificadores y los campos de nombre, correo, teléfono y dirección no se utilizaron como variables predictoras.

Dimensiones obtenidas:

| Conjunto | Dimensiones |
|---|---|
| X | (1000, 2) |
| y | (1000,) |

## 8. Partición estratificada Train-Test

Se utilizó la función `train_test_split` de Scikit-learn.

```python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)
```

El argumento `stratify=y` permite conservar aproximadamente la proporción de clases de `churn` en los conjuntos de entrenamiento y prueba.

### Resultados de la partición

| Conjunto | Registros | Porcentaje |
|---|---:|---:|
| Entrenamiento | 800 | 80% |
| Prueba | 200 | 20% |
| **Total** | **1,000** | **100%** |

### Comparación de distribuciones

| Conjunto | Registros | Churn 0 | Churn 1 |
|---|---:|---:|---:|
| Dataset completo | 1,000 | 80.80% | 19.20% |
| Entrenamiento | 800 | 80.75% | 19.25% |
| Prueba | 200 | 81.00% | 19.00% |

Las proporciones muestran diferencias mínimas entre los tres conjuntos.

Esto confirma que la estratificación conservó aproximadamente la distribución original de la variable objetivo.

## 9. Codificación de variables categóricas

Se aplicó One-Hot Encoding a la variable `estado_civil` utilizando `pd.get_dummies()`.

La variable categórica se transformó en tres columnas:

```text
estado_civil_Casado
estado_civil_Divorciado
estado_civil_Soltero
```

Las columnas finales fueron:

```text
ingreso_anual
estado_civil_Casado
estado_civil_Divorciado
estado_civil_Soltero
```

Cada categoría se representa mediante valores binarios `0` y `1`.

Se verificó que los conjuntos de entrenamiento y prueba conservaran las mismas columnas después de la codificación.

## 10. Validaciones realizadas

El programa incluye verificaciones automáticas para confirmar la integridad de los resultados.

| Validación | Resultado |
|---|---|
| Generación de 1,000 registros | Correcto |
| Conservación de registros en la partición | Correcto |
| Ausencia de índices compartidos entre Train y Test | Correcto |
| Coincidencia de columnas codificadas | Correcto |
| Variables predictoras numéricas | Correcto |
| Ausencia de valores faltantes | Correcto |

Todas las validaciones finalizaron satisfactoriamente.

## 11. Archivos generados

El programa exporta seis archivos CSV:

| Archivo | Contenido |
|---|---|
| `dataset_sintetico.csv` | Dataset original de 1,000 clientes |
| `X_train.csv` | Características codificadas de entrenamiento |
| `X_test.csv` | Características codificadas de prueba |
| `y_train.csv` | Variable objetivo de entrenamiento |
| `y_test.csv` | Variable objetivo de prueba |
| `comparacion_churn.csv` | Comparación de proporciones de churn |

Los archivos se guardan en el directorio `resultados/`.

Todos los registros son sintéticos y no corresponden a clientes reales.

## 12. Instalación y ejecución

### Clonar el repositorio

```bash
git clone https://github.com/Glass11114/practice_four_aws.git
cd practice_four_aws
```

### Crear y activar un entorno virtual

```bash
python3 -m venv venv
source venv/bin/activate
```

### Instalar las dependencias

```bash
python3 -m pip install -r requirements.txt
```

### Preparar la configuración

```bash
cp config.example.yaml config.yaml
```

### Ejecutar el programa

```bash
python3 practica4.py
```

### Verificar los archivos exportados

```bash
ls -lh resultados/
```

## 13. Análisis técnico

### ¿Por qué utilizar estratificación?

La estratificación es importante cuando las clases no están distribuidas de manera uniforme.

En este ejercicio, aproximadamente el 80% de los clientes pertenece a la clase `0` y el 20% a la clase `1`.

Una partición aleatoria sin estratificación podría producir proporciones diferentes entre entrenamiento y prueba.

Mediante `stratify=y`, se busca conservar una representación similar de ambas clases.

### ¿Por qué utilizar semillas aleatorias?

Las semillas permiten reproducir los resultados cuando se utiliza la misma configuración y un entorno compatible.

Esto facilita la depuración, la comparación de resultados y la documentación del proceso.

### ¿Por qué utilizar One-Hot Encoding?

Los algoritmos de aprendizaje automático suelen requerir características numéricas.

One-Hot Encoding transforma las categorías en indicadores binarios sin asignarles un orden artificial.

### Limitaciones del ejercicio

- Los registros son ficticios y no representan una población real.
- La variable `churn` se genera aleatoriamente, sin depender del ingreso ni del estado civil.
- La estratificación conserva proporciones de clases, pero no garantiza por sí sola un modelo predictivo preciso.
- Los resultados dependen de la configuración y de las semillas utilizadas.
- El ejercicio prepara los conjuntos de datos, pero no entrena un modelo de clasificación.

## 14. Conclusión

La práctica permitió implementar un proceso reproducible de generación y preparación de datos sintéticos utilizando Python, Faker, NumPy, Pandas y Scikit-learn.

Se generaron 1,000 registros de clientes ficticios y se realizó una partición estratificada de 800 registros para entrenamiento y 200 para prueba.

La proporción original de abandono fue del 19.20%, mientras que los conjuntos de entrenamiento y prueba conservaron porcentajes de 19.25% y 19.00%, respectivamente.

También se codificó la variable categórica `estado_civil` y se validó la integridad de los conjuntos resultantes.

En conclusión, la práctica demuestra la importancia de la generación reproducible de datos, la correcta separación de variables y la estratificación como etapas de preparación para proyectos de aprendizaje automático.
