import numpy as np
import os
import pandas as pd
import math
from fcmeans import FCM
import joblib
import pickle
import ast
import warnings
import uvicorn
from pathlib import Path

from fastapi import FastAPI, Request, Form, Depends
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold

from generadorpalabras import *

# Disable FutureWarning for regex parameter
warnings.filterwarnings("ignore")

templateJinja = Jinja2Templates(directory="templates")

app = FastAPI()
app.mount("/css", StaticFiles(directory="css"))
# Montar el directorio 'static' para servir archivos estáticos
app.mount("/static", StaticFiles(directory="static"), name="static")

app.m_score = []
app.m_word = []
app.s_score = []
app.L_MOSTRADAS = []
app.usuario = ""
app.curso = ""
app.prueba = ""
app.n_palabras = 0
G_NUM_P_INTERVENCION = 80
app.maximo = 0
app.minimo = 0
app.tendencia_positiva = 0
app.tendencia_negativa =  0
app.L_RESULTADOS = []
app.L_SCORES = []
# guardar los resultados obtenidos en un archivo .npy
def guardar_resultados(carpeta, nombre_archivo, mi_array):
    # Verificar si el archivo ya existe en la carpeta
    contador = 0
    if os.path.exists(os.path.join(carpeta, nombre_archivo + "_0.npy")):
        # Si el archivo ya existe, cambia el nombre
        nuevo_nombre_archivo = nombre_archivo + f'_{contador}.npy'

        while os.path.exists(os.path.join(carpeta, nuevo_nombre_archivo)):
            # Agrega un sufijo numérico para diferenciar
            contador += 1
            nuevo_nombre_archivo = nombre_archivo + f'_{contador}.npy'

        nombre_archivo = nuevo_nombre_archivo
    else:
        nombre_archivo = nombre_archivo + "_0"
    # Guardar el NumPy array en el archivo
    np.save(os.path.join(carpeta, nombre_archivo), mi_array)

# Función para entrenar el modelo mediante SGD
def train_SGD(train_ratings, n_factors, learning_rate, n_epochs):
    n_users, n_items = train_ratings.shape
    user_factors = np.random.rand(n_users, n_factors)
    item_factors = np.random.rand(n_items, n_factors)
    for epoch in range(n_epochs):
        for user in range(n_users):
            for item in range(n_items):
                if train_ratings[user][item] > 0:
                    error = train_ratings[user][item] - np.dot(user_factors[user], item_factors[item])
                    user_factors[user] += learning_rate * (2 * error * item_factors[item])
                    item_factors[item] += learning_rate * (2 * error * user_factors[user])
    return user_factors, item_factors


# Función para calcular el MSE en ratings no-nulos
def calculate_mse(predictions, actual_ratings):
    non_zero_indices = actual_ratings > 0
    error = predictions[non_zero_indices] - actual_ratings[non_zero_indices]
    return mean_squared_error(error, np.zeros_like(error))


# rellena la matriz M2 usando SGD
def rellenar_matriz(M2):
    ratings = np.array(M2)
    ratings = np.where(ratings == 0, np.nan, ratings)

    # Crear conjuntos de entrenamiento y prueba.
    pct_entrenamiento = 0.8
    total_ratings = np.count_nonzero(~np.isnan(ratings))
    num_entrenamiento = int(pct_entrenamiento * total_ratings)

    indices = np.column_stack(np.where(~np.isnan(ratings)))
    np.random.shuffle(indices)
    indices_entrenamiento = indices[:num_entrenamiento]
    indices_prueba = indices[num_entrenamiento:]

    train_data = np.full_like(ratings, np.nan)
    test_data = np.full_like(ratings, np.nan)

    for idx in indices_entrenamiento:
        train_data[idx[0], idx[1]] = ratings[idx[0], idx[1]]

    for idx in indices_prueba:
        test_data[idx[0], idx[1]] = ratings[idx[0], idx[1]]

    # Configuración de la validación cruzada y de los hiperparámetros a explorar
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    n_factors_values = [2, 3, 4]
    learning_rate_values = [0.01, 0.001]
    n_epochs_values = [50, 100]
    non_nan_idx = np.column_stack(np.where(~np.isnan(train_data)))

    # Grid search sobre los hiperparámetros
    min_mse = float('inf')
    best_params = None
    for n_factors in n_factors_values:
        for learning_rate in learning_rate_values:
            for n_epochs in n_epochs_values:
                fold_mses = []
                for train_idx, val_idx in kf.split(non_nan_idx):
                    train_subset = np.copy(train_data)
                    val_idx_tuple = non_nan_idx[val_idx]
                    train_subset[val_idx_tuple[:, 0], val_idx_tuple[:, 1]] = np.nan

                    # Entrenamiento
                    U, I = train_SGD(train_subset, n_factors, learning_rate, n_epochs)
                    predictions = np.dot(U, I.T)

                    # Validación
                    val_real_ratings = train_data[val_idx_tuple[:, 0], val_idx_tuple[:, 1]]
                    val_predicted_ratings = predictions[val_idx_tuple[:, 0], val_idx_tuple[:, 1]]
                    fold_mse = calculate_mse(val_predicted_ratings, val_real_ratings)
                    fold_mses.append(fold_mse)

                # Promedio de MSE en todos los folds
                avg_mse = np.mean(fold_mses)
                if avg_mse < min_mse:
                    min_mse = avg_mse
                    best_params = (n_factors, learning_rate, n_epochs)

    # Entrenamiento y evaluación con los mejores hiperparámetros sobre todos los datos
    best_factors, best_lr, best_epochs = best_params
    U, I = train_SGD(train_data, best_factors, best_lr, best_epochs)
    all_predictions = np.dot(U, I.T)

    print(f"Mejores hiperparámetros: {best_params}")
    min_rmse = math.sqrt(min_mse)
    print(f"RMSE de la CV con optimización de hiperparámetros: {min_rmse}")
    # Entrenamiento y test con los mejores hiperparámetros
    final_rmse_test = math.sqrt(calculate_mse(all_predictions, test_data))
    print(f"RMSE de la CV con optimización de hiperparámetros: {final_rmse_test}")

    return all_predictions


def matriz_palabras(prueba):
    """
    Genera la matriz de palabras para la prueba indicada.

    Args:
        prueba (int): Número de la prueba.

    Returns:
        list: Lista de listas con las palabras generadas para cada dificultad.
    """
    COLUMNAS_MATRIZ = 7
    G_MatrizM2_Palabras = []
    G_Palabras_Generadas = []
    G_PRUEBA = int(prueba)

    G_PSEUDO = 1  # 0 = Se intentan proponer palabras que sí que aparecen en el Corco
    # 1 = No se hacen búsquedas en el Corco (alta probabilidad de que se genere una pseudo-palabra)

    df_cribado = pd.read_excel('actividades_cribado.xlsx')
    # Convert the strings representing lists to actual lists:
    word_prueba = df_cribado.loc[df_cribado['Prueba'] == G_PRUEBA].drop(df_cribado.columns[0], axis=1).dropna(
        axis=1).values

    # Matriz W2
    for word in word_prueba[0]:
        for i in range(COLUMNAS_MATRIZ + 1):
            if i in [0]:
                p_generadas = genera_palabras(word, i, i, i, G_PSEUDO)
                if (len(p_generadas) > 0):
                    G_Palabras_Generadas.append(p_generadas)
                else:
                    G_Palabras_Generadas.append(np.nan)

            elif i in [3, 4, 6, 7]:
                p_generadas = genera_palabras(word, i, i, i, G_PSEUDO)
                if (len(p_generadas) > 0):
                    G_Palabras_Generadas.append(p_generadas)
                else:
                    G_Palabras_Generadas.append(np.nan)
            else:
                if i in [1, 2]:
                    for j in range(3, 0, -1):
                        p_generadas = genera_palabras(word, i, j, j, G_PSEUDO)
                        if (len(p_generadas) > 0):
                            G_Palabras_Generadas.append(p_generadas)
                        else:
                            G_Palabras_Generadas.append(np.nan)
                else:
                    for j in range(13, 0, -1):
                        p_generadas = genera_palabras(word, i, j, j, G_PSEUDO)
                        if (len(p_generadas) > 0):
                            G_Palabras_Generadas.append(p_generadas)
                        else:
                            G_Palabras_Generadas.append(np.nan)

        G_Palabras_Generadas = [sublista for sublista in G_Palabras_Generadas]

        G_MatrizM2_Palabras.append(G_Palabras_Generadas)
        G_Palabras_Generadas = []

    return G_MatrizM2_Palabras


def matriz_usuarios(user, curso, prueba):
    """
    Genera la matriz de usuarios para el usuario, curso y prueba indicados.

    Args:
        user (str): Nombre del usuario.
        curso (int): Número del curso.
        prueba (int): Número de la prueba.

    Returns:
        list: Lista de listas con las dificultades para cada palabra.
    """
    G_Modulador = [1.1, 0.9]
    COLUMNAS_MATRIZ = 7
    G_Score_Dificil = 0.8
    G_Score_Intermedio = 0.5
    G_Score_Facil = 0.2
    G_USER = user
    G_CURSO = curso
    G_MatrizM2 = []
    G_PRUEBA = prueba
    G_Dificultad_Cambio = []

    df = pd.read_csv('dificultad_palabras.csv')
    df_user_data = pd.read_csv('datos_modelo.csv')
    df_cribado = pd.read_excel('actividades_cribado.xlsx')
    # Convert the strings representing lists to actual lists:
    df_user_data['Respuesta'] = df_user_data['Respuesta'].replace(['NaN', 'nan', 'Nan'], '[]')
    df_user_data['Respuesta'] = df_user_data['Respuesta'].apply(ast.literal_eval)

    word_prueba = df_cribado[df_cribado['Prueba'] == G_PRUEBA].values

    usuarios_unicos = df_user_data['Usuario'].drop_duplicates()
    # Utiliza reset_index() para restablecer el índice del DataFrame resultante
    usuarios_unicos = usuarios_unicos.reset_index(drop=True)

    prob_base = []
    palabras = []

    for index, row in df.iterrows():
        word = row['Palabras']
        dificultad = row['Dificultad']
        if word in word_prueba:
            palabras.append(word)
            if dificultad == 'FACIL':
                prob_base.append(G_Score_Facil)
            elif dificultad == 'INTERMEDIA':
                prob_base.append(G_Score_Intermedio)
            elif dificultad == 'DIFICIL':
                prob_base.append(G_Score_Dificil)

    # PRUEBA
    df_filtrado = df_user_data[
        (df_user_data['Usuario'] == G_USER) & (df_user_data['Curso'] == G_CURSO) & (df_user_data['Prueba'] == G_PRUEBA)]

    # Matriz M2
    for index, row in df_filtrado.iterrows():
        answer = row['Respuesta']
        word = row['Items']
        for index, _answer in enumerate(answer):
            if _answer == 0:
                prob_base[index] = round(prob_base[index] * G_Modulador[0], 4)
            elif _answer == 1:
                prob_base[index] = round(prob_base[index] * G_Modulador[1], 4)
    # Matriz W2
    for word, score in zip(palabras, prob_base):
        for i in range(COLUMNAS_MATRIZ + 1):
            if i in [0]:
                G_Dificultad_Cambio.append(round(score, 4))

            elif i in [3, 4, 6, 7]:
                G_Dificultad_Cambio.append(0)

            else:
                if i in [1, 2]:
                    for j in range(3, 0, -1):
                        G_Dificultad_Cambio.append(0)

                else:
                    for j in range(13, 0, -1):
                        G_Dificultad_Cambio.append(0)

        G_MatrizM2.append(G_Dificultad_Cambio)
        G_Dificultad_Cambio = []

    return G_MatrizM2


def contar_elementos_no_cero(matriz):
    contador = 0
    for fila in matriz:
        for elemento in fila:
            if elemento != 0:
                contador += 1
    return contador

def cargar_matrices_scores(usuario, curso, prueba, W2):
    """
    Carga la matriz de puntuaciones del usuario para la prueba especificada.

    Args:
        usuario (str): Nombre del usuario.
        curso (int): Número del curso.
        prueba (int): Número de la prueba.
        W2 (np.array): Matriz de palabras de la prueba.

    Returns:
        np.array: Matriz de puntuaciones del usuario para la prueba.
    """
    # Ruta del archivo de puntuaciones del usuario
    usuario_scores_path = f"matrices/scores/Curso/{curso}/{usuario}/{prueba}/{usuario}_{prueba}_niño.npy"

    # Si el archivo de puntuaciones del usuario existe, se carga directamente
    if os.path.exists(usuario_scores_path):
        M2_A = np.load(usuario_scores_path)
    # Si no existe, generar una nueva matriz de usuarios
    else:
        new_M2 = matriz_usuarios(str(usuario), int(curso), int(prueba))
        # Conversión a formato numpy y eliminación de columnas vacías
        new_M2 = np.array(new_M2)
        columnas_no_cero = [i for i in range(len(W2[0])) if any(isinstance(row[i], list) for row in W2)]
        # Se guarda la nueva matriz de puntuaciones en un archivo `.npy`.
        new_M2_filt = new_M2[:, columnas_no_cero]
        M2_A = new_M2_filt

    return M2_A


def cargar_matrices_words(prueba):
    """
    Carga la matriz de palabras para la prueba especificada.

    Args:
        prueba (int): Número de la prueba.

    Returns:
        np.array: Matriz de palabras para la prueba.
    """
    # Ruta del archivo de la prueba de palabras
    prueba_path = f"matrices/palabras/Prueba_{prueba}.npy"

    # Si el archivo de la prueba de palabras existe, se carga directamente
    if os.path.exists(prueba_path):
        W2 = np.load(prueba_path, allow_pickle=True)

    # Si el archivo no existe, se genera una nueva matriz de palabras
    else:
        new_W2 = matriz_palabras(prueba)
        # Conversión a formato numpy y eliminación de columnas vacías
        new_W2 = np.array(new_W2, dtype=object)
        columnas_no_cero = [i for i in range(len(new_W2[0])) if any(isinstance(row[i], list) for row in new_W2)]
        new_W2_filt = new_W2[:, columnas_no_cero]
        # Se guarda la nueva matriz de palabras en un archivo .npy
        np.save(f'matrices/palabras/Prueba_{prueba}.npy', new_W2_filt)
        W2 = new_W2_filt
    return W2


def ultima_respuesta(usuario, curso, prueba):
    """
    Obtiene la última respuesta del usuario en una prueba específica.

    Args:
        usuario (str): Nombre del usuario.
        curso (int): Número del curso.
        prueba (int): Número de la prueba.

    Returns:
        list: Lista que contiene la última respuesta del usuario.
    """
    # Carga del archivo CSV con los datos de los usuarios
    df_user_data = pd.read_csv('datos_modelo.csv')

    # Filtrado por usuario, curso y prueba
    df_filtrado = df_user_data[(df_user_data['Usuario'] == usuario) & (df_user_data['Curso'] == int(curso)) & (
                df_user_data['Prueba'] == int(prueba))]

    # Conversión de la columna 'Respuesta' a formato lista
    df_filtrado['Respuesta'] = df_filtrado['Respuesta'].apply(ast.literal_eval)

    return df_filtrado.iloc[-1]['Respuesta']

def dificultades_proceso_heuristico(lista_de_tuplas_filtrada, FIN_PRUEBA):
    """
    Genera una lista con las dificultades que se le van a mostrar al usuario en el proceso heurístico.

    Args:
        lista_de_tuplas_filtrada (list): Lista de tuplas con las dificultades para cada palabra.
        FIN_PRUEBA (int): Número de palabras que se van a mostrar al usuario.

    Returns:
        list: Lista con las tuplas (fila, columna) que representan las dificultades a mostrar.
    """

    # Número deseado de tuplas a seleccionar
    max_fila = max(tupla[0] for tupla in lista_de_tuplas_filtrada)
    max_columna = max(tupla[1] for tupla in lista_de_tuplas_filtrada)

    # Mezclar aleatoriamente la lista de tuplas
    random.shuffle(lista_de_tuplas_filtrada)

    # Inicializar lista resultante
    lista_resultante = []

    # Índice para recorrer las columnas en lista_de_tuplas_filtrada
    indice = 0
    columna = 1
    while indice <= FIN_PRUEBA:
        tuplas_encontradas = list(filter(lambda x: x[1] == columna, lista_de_tuplas_filtrada))
        if tuplas_encontradas:
            lista_resultante.append(tuplas_encontradas[0])
            lista_de_tuplas_filtrada.remove(tuplas_encontradas[0])
            indice += 1

        columna += 1
        if columna > max_columna:
            columna = 1
    return lista_resultante
def proceso_herustico(matriz_score, matriz_palabras, usuario, porcentaje):
    """
    Ejecuta un proceso heurístico para seleccionar palabras a mostrar al usuario.

    Args:
        matriz_score (np.array): Matriz de puntuaciones (dificultad) para las palabras.
        matriz_palabras (list): Matriz de palabras.
        usuario (Usuario): Objeto que representa al usuario actual.
        porcentaje (float): Porcentaje de palabras a mostrar en la prueba heurística.

    Returns:
        list: Lista de palabras a mostrar al usuario.
        list: Lista de posiciones de las palabras en las matrices.
    """
    matriz_score = np.array(matriz_score)
    # --- Parámetros ---
    FIN_PRUEBA = int(np.size(matriz_score) * porcentaje)  # Número de palabras que aparecen en la prueba heurística
    # --- Obtención de posiciones válidas ---
    posiciones_cumplen_condicion = [(fila, columna) for fila, fila_datos in enumerate(matriz_palabras) for
                                    columna, celda in enumerate(fila_datos) if isinstance(celda, list)]
    lista_de_tuplas_filtrada = [tupla for tupla in posiciones_cumplen_condicion if tupla[1] != 0]
    posiciones_usadas = dificultades_proceso_heuristico(lista_de_tuplas_filtrada, FIN_PRUEBA)

    # --- Bucle principal ---
    lista_palabras = []

    for i in range(FIN_PRUEBA):
        pos_word, dificultad = posiciones_usadas[i]
        # De todas las palabras posibles para esa dificultad escogemos una aleatoria
        word = random.choice(matriz_palabras[pos_word][dificultad])
        lista_palabras.append(word)

    return lista_palabras, posiciones_usadas

def procesar_respuesta(result, m_score, posiciones_usadas):
    G_Modulador = [1.1, 0.9]  # 1.1 = Fallo
                              # 0.9 = Acierto
    # Asignar score si ACIERTA
    pos_word = int(posiciones_usadas[0])
    dificultad = int(posiciones_usadas[1])
    m_score = np.array(m_score)
    if m_score[pos_word][dificultad] == 0:
        m_score[pos_word][dificultad] = dificultad * m_score[pos_word][0]

    if result.lower() == 'y':
        m_score[pos_word][dificultad] = np.around(m_score[pos_word][dificultad] * G_Modulador[1], 4)
    # Asignar score si FALLA
    elif result.lower() == 'n':
        m_score[pos_word][dificultad] = np.around(m_score[pos_word][dificultad] * G_Modulador[0], 4)

    return m_score


def init_score(respuesta, m_score):
    """
    Calcula la puntuación inicial de un usuario, basado en sus respuestas al conjunto de palabras semillas
    de la prueba.

    Args:
        respuesta (list): Lista de respuestas del usuario (1 para correcto, 0 para incorrecto).
        m_score (list): Matriz de dificultad, donde cada elemento m_score[i][0] contiene
                        la puntuación inicial de la i-ésima palabra.

    Returns:
        float: La puntuación inicial calculada para el usuario.
    """
    vector_media = [] # Lista para almacenar puntuaciones de las respuestas correctas
    # Itera a través de las respuestas del usuario, buscando aciertos
    for index, r in enumerate(respuesta):
        # Si la respuesta es correcta (1)
        if r == 1:
            vector_media.append(m_score[index][0]) # Añade la puntuación de la dificultad
    # Si no se encontraron respuestas correctas, asignamos puntuaciones según respuestas incorrectas
    if (len(vector_media) == 0):
        for index, r in enumerate(respuesta):
            if r == 0:  # Si la respuesta es incorrecta (0)
                vector_media.append(m_score[index][0])

    media = np.mean(vector_media) # Calculo de la media de las puntuaciones

    return media


def escoger_palabra(s_score, incremento, m_word, result, score):
    """
    Esta función busca una palabra para mostrar al usuario, teniendo en cuenta su puntuación actual.

    Args:
        s_score (list): Matriz de puntuación estándar.
        incremento (float): Incremento/decremento de la puntuación para la búsqueda.
        m_word (list): Matriz de palabras.
        result (str): Resultado del usuario en la última pregunta ('y' para acierto, 'n' para fallo).
        score (float): Puntuación actual del usuario.

    Returns:
        str: Palabra seleccionada para mostrar al usuario.
        tuple: Posición de la palabra en las matrices (i, j).
    """
    encontrado = True
    # --- Búsqueda de la siguiente palabra ---
    while encontrado:
        # Recorremos la matriz s_score buscando palabras con el score actual
        for i in range(0, len(s_score)):
            for j in range(0, len(s_score[i])):
                # Rango de busqueda [score + 15%, score - 15%]
                if round(score - incremento, 2) <= s_score[i][j] <= round(score + incremento, 2):
                    # Si hay palabras para ese score
                        if m_word[i][j] != "nan":
                        # Seleccionamos una palabra al azar
                            word = m_word[i][j]
                            # Buscamos una palabra que no haya aparecido
                            if word in app.L_MOSTRADAS:
                                break
                            app.L_MOSTRADAS.append(word)
                            encontrado = False
                            print(s_score[i][j])
                            pos = (i, j)
                            return word, pos
            # Si se ha mostrado una palabra pasamos al siguiente score
            if not encontrado:
                break
        # Si no se ha encontrado ninguna palabra, aumentamos/disminuimos el score
        if encontrado:
            if score >= app.maximo:
                result = 'n'
            elif score <= app.minimo:
                result = 'y'
            if result.lower() == 'y':
                score = round(score + incremento, 3)
            else:
                score = round(score - incremento, 3)


def actualizar_m2(result, m_score, s_score, score, pos):
    """
    Actualiza las puntuaciones del aprendizaje M2 en función del resultado del usuario (si acertó o falló).

    Args:
        result (str): Respuesta del usuario: 'y' para acierto, 'n' para fallo.
        m_score (list): Matriz de dificultad del aprendizaje M2.
        s_score (list): Matriz de puntuación estándar.
        score (float): Puntuación actual del usuario.
        pos (list): Posición de la palabra y dificultad (pos[0] = palabra, pos[1] = dificultad).

    Returns:
        list: Matriz de dificultad actualizada (m_score).
        float: Puntuación actualizada del usuario (score).
    """

    # --- Variables ---
    Modulador = [1.3, 0.7]  # Factores usados para ajustar la puntuación
                            # 1.3 sube la puntuación, 0.7 la baja

    pos_word = int(pos[0])     # Extrae la posición de la palabra
    dificultad = int(pos[1])   # Extrae la dificultad

    # --- Actualización según acierto/fallo ---
    # --- ACIERTO ---
    if result.lower() == 'y':
        # Reinicia racha de fallos e incrementa racha de aciertos
        app.tendencia_negativa = 0
        app.tendencia_positiva += 1
        app.L_RESULTADOS.append(0)
        app.L_SCORES.append(score)
        # Modificador si hay 6 o más aciertos seguidos
        if (app.tendencia_positiva >= 6):
            # Reduce el score de la palabra
            m_score[pos_word][dificultad] = s_score[pos_word][dificultad] * (Modulador[1]*0.9)
            # Aumenta el score de la siguiente palabra que le tiene que aparecer
            score = round(score + (app.incremento*4), 3)
        # Modificador estándar
        else:
            # Reduce el score de la palabra
            if (s_score[pos_word][dificultad] * (Modulador[1]) >= 0.1):
                m_score[pos_word][dificultad] = s_score[pos_word][dificultad] * Modulador[1]
            else:
                m_score[pos_word][dificultad] = 0.1
            score = round(score + (app.incremento*2), 3)

        # Limitador de puntuación máxima
        if score > app.maximo:
            score == app.maximo

    # --- FALLO ---
    elif result.lower() == 'n':
        # Incrementa racha de fallos y reinicia la racha de aciertos
        app.tendencia_negativa += 1
        app.tendencia_positiva = 0
        app.L_RESULTADOS.append(1)
        app.L_SCORES.append(score)
        # Modificador si hay 6 o más fallos seguidos
        if(app.tendencia_negativa >= 6):
            # Aumenta el score de la palabra
            m_score[pos_word][dificultad] = s_score[pos_word][dificultad] * (Modulador[0]*1.2)
            # Disminuye el score de la siguiente palabra que le tiene que aparecer
            score = round(score - (app.incremento*2), 3)
        # Modificador estándar
        else:
            # Aumenta el score de la palabra
            m_score[pos_word][dificultad] = s_score[pos_word][dificultad] * Modulador[0]
            # Disminuye el score de la siguiente palabra que le tiene que aparecer
            score = round(score - app.incremento, 3)

        # Limitador de puntuación mínima
        if score < app.minimo:
            score = app.minimo

    return m_score, score
def resize_matrices_nan(matriz):
  """
  Esta función toma una matriz de listas y rellena las listas de cada columna
  hasta que todas las listas de la columna tengan la misma longitud.

  Parámetros:
    matriz: Una matriz de listas.

  Retorno:
    Una matriz de listas con las listas de igual longitud.
  """
  num_filas = len(matriz)
  num_columnas = len(matriz[0]) # Asumimos que todas las filas tienen igual longitud

  # Encontrar la longitud máxima de cada columna
  columnas_max = [0] * num_columnas
  for i in range(num_filas):
     for j in range(num_columnas):
         if isinstance(matriz[i][j], list):
             columnas_max[j] = max(columnas_max[j], len(matriz[i][j]))
         else:
            matriz[i][j] = [matriz[i][j]]

  # Rellenar las listas con None
  for i in range(num_filas):
    for j in range(num_columnas):
        if isinstance(matriz[i][j], list):
            if len(matriz[i][j]) < columnas_max[j]:
                matriz[i][j] += [np.nan] * (columnas_max[j] - len(matriz[i][j]))

  return matriz

def matrices_precisas(m_score, m_word):
    """
    Esta función toma las matrices M2 y W2 (score y palabras) y les da un formato
    regular con dimensiones iguales, asegurando que todas las palabras se tengan en cuenta.

    Args:
        m_score (list): Matriz de dificultad del aprendizaje M2.
        m_word (list): Matriz de palabras.

    Returns:
        np.array: Matriz de dificultad del aprendizaje M2 actualizada.
        np.array: Matriz de palabras actualizada.
    """
    l_M2 = []  # Lista temporal para construir las filas de M2
    l_W2 = []  # Lista temporal para construir las filas de W2
    palabras_encontradas = set()  # Para evitar palabras repetidas
    M2 = []  # Resultado final: Matriz M2
    W2 = []  # Resultado final: Matriz W2
    max_length = 0  # Longitud máxima de las filas
    # --- Recorrido de las matrices originales ---
    #m_word = resize_matrices_nan(m_word)

    for _M2, _W2 in zip(m_score, m_word):
        for s, w in zip(_M2, _W2):
            # Caso: 'w' es una lista (varias palabras con el mismo score)
            if isinstance(w, list):
                for _w in w:
                    l_M2.append(s)   # Agrega el score a la lista temporal
                    if _w not in palabras_encontradas:
                        l_W2.append(_w) # Agrega la palabra si es única
                        palabras_encontradas.add(_w)
                    else:
                        l_W2.append(np.nan) # Si está repetida, agrega NaN
            # Caso: 'w' es una sola palabra
            else:
                l_M2.append(s)
                l_W2.append(w)

        # Determina la longitud máxima (para rellenar posteriormente)
        max_length = max(max_length, len(l_M2))

        # Agrega las filas construidas a las matrices resultado
        M2.append(l_M2)
        W2.append(l_W2)
        # Reinicia variables temporales para la siguiente fila
        l_M2 = []
        l_W2 = []
    # --- Relleno (padding) para igualar dimensiones ---
    M2 = np.array([np.pad(row, (0, max_length - len(row)), constant_values=np.nan) for row in M2])
    W2 = np.array([np.pad(row, (0, max_length - len(row)), constant_values=np.nan) for row in W2])

    columnas_no_cero = [i for i, columna in enumerate(W2.T) if all(elemento == "nan" for elemento in columna)]
    new_W2_filt = np.delete(W2, columnas_no_cero, axis=1)
    new_M2_filt = np.delete(M2, columnas_no_cero, axis=1)

    return new_M2_filt, new_W2_filt


@app.get("/",response_class=HTMLResponse)
def home(request: Request):
    return templateJinja.TemplateResponse("home.html",{
        "request": request
    })

@app.post("/guardar_datos")
def guardar_datos(request: Request, usuario: str = Form('usuario'), prueba: str = Form('prueba'), curso: str = Form('curso')):
    # Cargar las matrices usando las funciones
    app.usuario = usuario
    app.curso = curso
    app.prueba = prueba
    app.m_word = cargar_matrices_words(prueba)
    app.m_score = cargar_matrices_scores(usuario, curso, prueba, app.m_word)

    app.respuesta = ultima_respuesta(usuario, curso, prueba)
    app.score = init_score(app.respuesta, app.m_score)  # Score de partida
    app.incremento = app.score*0.15                 # El incremento de dificultad es del 15% del score inicial
    if contar_elementos_no_cero(app.m_score) <= 12:
        l_word, l_posiciones = proceso_herustico(app.m_score, app.m_word, str(usuario), 0.2)
        return templateJinja.TemplateResponse("procesoheuristico.html",{
            "request": request,
            "word_list": l_word,
            "l_posiciones": l_posiciones
        })
    else:
        # Pensar otra manera de hacerlo
        _,app.m_word = matrices_precisas(app.m_score,app.m_word)
        app.s_score = rellenar_matriz(app.m_score)
        app.maximo = np.nanmax(app.s_score)
        app.minimo = np.nanmin(app.s_score)
        print(app.m_score)
        return templateJinja.TemplateResponse("recomendacion.html", {
            "request": request
        })

@app.post("/proceso_heuristico")
def proceso_heuristico(request: Request,
    button_value: str = Form(..., description="Value of the button"),
    pos_list_value: str = Form(..., description="Value of posList"),
    is_last_element: str = Form(..., description="Value of isLastElement"),  # Nueva variable booleana
):
    try:
        # Aquí puedes hacer lo que quieras con los valores recibidos
        pos_list_value= tuple(map(float, pos_list_value.split(',')))
        app.m_score = procesar_respuesta(button_value, app.m_score, pos_list_value) # DEVUELVE UNA MATRIZ

    except Exception as e:
        return {"error": str(e)}

    if is_last_element == str(1):
        # Creo el directorio
        m_s = app.m_score
        w_s = app.m_word
        app.m_score, app.m_word = matrices_precisas(m_s,w_s)
        s_s = rellenar_matriz(m_s)
        app.s_score,_ = matrices_precisas(s_s,w_s)
        app.maximo = np.nanmax(app.s_score)
        app.minimo = np.nanmin(app.s_score)
        Path('matrices/scores/Curso/' + str(app.curso) + '/' + str(app.usuario) + '/' + str(app.prueba)).mkdir(parents=True,exist_ok=True)
        np.save('matrices/scores/Curso/' + str(app.curso) + '/' + str(app.usuario) + '/' + str(app.prueba) + '/' + str(app.usuario) + '_' + app.prueba + '_niño' + '.npy', app.m_score)

        return JSONResponse(content={"redirect_url": "/recomendacion"}, status_code=200)

@app.get("/recomendacion",response_class=HTMLResponse)
def recomendacion_template(request: Request):
    return templateJinja.TemplateResponse("recomendacion.html",{
        "request": request
    })

@app.post('/obtener_palabra')
def obtener_palabra(request: Request,button_value: str = Form(..., description="Value of the button")):
    word, pos = escoger_palabra(app.s_score, app.incremento, app.m_word, button_value, app.score)
    if app.n_palabras < G_NUM_P_INTERVENCION:
        # Crear una respuesta JSON usando el objeto JSONResponse
        print(app.n_palabras)
        data = {"palabra": word,"posicion": pos, "n_palabras": app.n_palabras}
        response = JSONResponse(content=data)
        return response
    else:
        app.n_palabras = 0
        # Parte común del path
        common_path = f'matrices/scores/Curso/{app.curso}/{app.usuario}/{app.prueba}'
        usuario_scores_path = f'matrices/scores/Curso/{app.curso}/{app.usuario}/{app.prueba}/{app.usuario}_{app.prueba}_niño.npy'
        sgd_scores_path = f'matrices/scores/Curso/{app.curso}/{app.usuario}/{app.prueba}/SGD_{app.usuario}_{app.prueba}_niño.npy'

        # Guardar los cambios que se han hecho en las dificultades de las palabras demostradas
        guardar_resultados(common_path, f'cambiosniños', np.array(app.L_MOSTRADAS))

        # Guardar las respuestas de los sujetos
        guardar_resultados(common_path, f'resultadosniños', np.array(app.L_RESULTADOS))

        # Guardar los scores de los sujetos
        guardar_resultados(common_path, f'scoresniños', np.array(app.L_SCORES))
        usuario_scores_path = f'matrices/scores/Curso/{app.curso}/{app.usuario}/{app.prueba}/{app.usuario}_{app.prueba}_niño.npy'
        np.save(usuario_scores_path, app.m_score)
        app.L_RESULTADOS = []
        app.L_SCORES = []
        app.L_MOSTRADAS = []

        return JSONResponse(content={"redirect_url": "/fin_prueba"}, status_code=200)

@app.post('/actualizar_SR')
def actualizar_SR(button_value: str = Form(..., description="Value of the button"),pos_list: str = Form(..., description="Value of posList"),):
    pos_list = tuple(map(float, pos_list.strip('"').split(',')))
    app.m_score, app.score = actualizar_m2(button_value, app.m_score, app.s_score, app.score, pos_list)
    app.n_palabras += 1

@app.get("/fin_prueba",response_class=HTMLResponse)
def fin_prueba_template(request: Request):
    return templateJinja.TemplateResponse("fin_prueba.html",{
        "request": request
    })
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
