"""
@author: jimateo
"""
import numpy as np
import os
import pandas as pd
import math
from fcmeans import FCM
import joblib
import pickle
import ast
import warnings

from pathlib import Path
import math
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.metrics import mean_squared_error
from generadorpalabras import *
from niñosvirtuales import *

# Disable FutureWarning for regex parameter
warnings.filterwarnings("ignore")

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

    G_PSEUDO = 1     # 0 = Se intentan proponer palabras que sí que aparecen en el Corco
                     # 1 = No se hacen búsquedas en el Corco (alta probabilidad de que se genere una pseudo-palabra)
                     
    df_cribado = pd.read_excel('actividades_cribado.xlsx')
    # Convert the strings representing lists to actual lists:
    word_prueba = df_cribado.loc[df_cribado['Prueba'] == G_PRUEBA].drop(df_cribado.columns[0], axis=1).dropna(axis=1).values

    # Matriz W2
    for word in word_prueba[0]:
        for i in range(COLUMNAS_MATRIZ+1):
            if i in [0]:
                p_generadas = genera_palabras(word, i, i, i ,G_PSEUDO)
                if(len(p_generadas)>0):
                    G_Palabras_Generadas.append(p_generadas)
                else:
                    G_Palabras_Generadas.append(np.nan)
                            
            elif i in [3,4,6,7]:
                p_generadas = genera_palabras(word, i, i, i ,G_PSEUDO)
                if(len(p_generadas)>0):
                    G_Palabras_Generadas.append(p_generadas)
                else:
                    G_Palabras_Generadas.append(np.nan)
            else:
                if i in [1,2]:
                    for j in range(3,0,-1):
                        p_generadas = genera_palabras(word, i, j, j ,G_PSEUDO)
                        if(len(p_generadas)>0):
                            G_Palabras_Generadas.append(p_generadas)
                        else:
                            G_Palabras_Generadas.append(np.nan)
                else:
                    for j in range(13,0,-1):
                        p_generadas = genera_palabras(word, i, j, j ,G_PSEUDO)
                        if(len(p_generadas)>0):
                            G_Palabras_Generadas.append(p_generadas)
                        else:
                            G_Palabras_Generadas.append(np.nan)


        G_Palabras_Generadas = [sublista for sublista in G_Palabras_Generadas]

        G_MatrizM2_Palabras.append(G_Palabras_Generadas)
        G_Palabras_Generadas = []

    return G_MatrizM2_Palabras

        
def matriz_usuarios(user,curso,prueba):
    """
    Genera la matriz de usuarios para el usuario, curso y prueba indicados.
    
    Args:
        user (str): Nombre del usuario.
        curso (int): Número del curso.
        prueba (int): Número de la prueba.
    
    Returns:
        list: Lista de listas con las dificultades para cada palabra.
    """
    L_Modulador = [1.1, 0.9]
    COLUMNAS_MATRIZ = 24
    G_Score_Dificil = 0.8
    G_Score_Intermedio = 0.5
    G_Score_Facil = 0.2
    G_USER = user
    G_CURSO = curso
    G_MatrizM2 = []
    G_PRUEBA = prueba
    G_Dificultad_Cambio = []

    G_PSEUDO = 1     # 0 = Se intentan proponer palabras que sí que aparecen en el Corco
                     # 1 = No se hacen búsquedas en el Corco (alta probabilidad de que se genere una pseudo-palabra)

    df = pd.read_csv('dificultad_palabras.csv')
    df_user_data = pd.read_csv('datos_modelo.csv')
    df_cribado = pd.read_excel('actividades_cribado.xlsx')
    
    # Convertir las cadenas que representan listas a listas reales:
    df_user_data['Respuesta'] = df_user_data['Respuesta'].replace(['NaN', 'nan', 'Nan'], '[]')
    df_user_data['Respuesta'] = df_user_data['Respuesta'].apply(ast.literal_eval)

    word_prueba = df_cribado[df_cribado['Prueba'] == G_PRUEBA].values

    usuarios_unicos = df_user_data['Usuario'].drop_duplicates()
    # Utiliza reset_index() para restablecer el índice del DataFrame resultante
    usuarios_unicos = usuarios_unicos.reset_index(drop=True)

    prob_base = []

    for index, row in df.iterrows():
        word = row['Palabras']
        dificultad = row['Dificultad']
        if word in word_prueba:
            if dificultad == "FACIL":
                prob_base.append(G_Score_Facil)
            elif dificultad == "INTERMEDIA":
                prob_base.append(G_Score_Intermedio)
            elif dificultad == "DIFICIL":
                prob_base.append(G_Score_Dificil)

    # Asegurarse de que prob_base tenga al menos 12 elementos
    prob_base += [G_Score_Dificil] * (12 - len(prob_base))

    # Filtrar datos para la prueba, usuario y curso específicos
    df_filtrado = df_user_data[(df_user_data['Usuario'] == G_USER) & (df_user_data['Curso'] == G_CURSO) & (df_user_data['Prueba'] == G_PRUEBA)]

    # Modificar prob_base según las respuestas del usuario
    for index, row in df_filtrado.iterrows():
        answer = row['Respuesta']
        for i, _answer in enumerate(answer):
            # Ha fallado
            if _answer == 1:
                prob_base[i] = round(prob_base[i] * L_Modulador[0], 4)
            # Ha acertado
            elif _answer == 0:
                prob_base[i] = round(prob_base[i] * L_Modulador[1], 4)

    # Matriz M2
    for score in prob_base:
        for i in range(COLUMNAS_MATRIZ+1):
            if i == 0:
                G_Dificultad_Cambio.append(round(score,4))
            else:
                G_Dificultad_Cambio.append(0)

        G_MatrizM2.append(G_Dificultad_Cambio)
        G_Dificultad_Cambio = []

    return G_MatrizM2

# Verificar si el modelo FCM ya está ajustado y guardado
def asignar_cluster(nuevo_usuario):
    """
    Asigna un cluster al nuevo usuario utilizando un modelo FCM preajustado.

    Args:
        nuevo_usuario (np.array): Vector de características del nuevo usuario.

    Returns:
        np.array: Vector de pertenencia del usuario a los clusters.
    """
    # Cargar el modelo FCM
    with open('modelo_fcm.pkl', 'rb') as archivo:
        fcm = pickle.load(archivo)

    # Calcular la pertenencia del nuevo usuario a los clusters existentes
    membresia = fcm.predict(nuevo_usuario)

    print(f"El nuevo usuario pertenece al Cluster {membresia}")
    return membresia


def cargar_matrices(usuario, curso, prueba):
    # Ruta del archivo de la prueba de palabras
    prueba_path = f"matrices/palabras/Prueba_{prueba}.npy"
    # Ruta del archivo de puntuaciones del usuario
    usuario_scores_path = f"matrices/scores/Curso/{curso}/{usuario}/{prueba}/{usuario}_{prueba}_niño.npy"

    # Si el archivo de la prueba de palabras existe, se carga directamente
    if os.path.exists(prueba_path):
        W2 = np.load(prueba_path, allow_pickle=True)
    else:
        # Si el archivo no existe, se genera una nueva matriz de palabras
        new_W2 = matriz_palabras(prueba)
        # Conversión a formato numpy y eliminación de columnas vacías
        new_W2 = np.array(new_W2, dtype=object)
        columnas_no_cero = [i for i in range(len(new_W2[0])) if any(isinstance(row[i], list) for row in new_W2)]
        new_W2_filt = new_W2[:, columnas_no_cero]
        # Se guarda la nueva matriz de palabras en un archivo .npy
        np.save(f'matrices/palabras/Prueba_{prueba}.npy', new_W2_filt)
        W2 = new_W2_filt

    # Si el archivo de puntuaciones del usuario existe, se carga directamente
    if os.path.exists(usuario_scores_path):
        M2_A = np.load(usuario_scores_path)
        W2_A = np.load(f"matrices/palabras/Prueba_{prueba}_ampliada.npy", allow_pickle=True)
        # SGD
        S2_A = SGD_inference(M2_A)
        
    # Si no existe, generar una nueva matriz de usuarios
    else:
        new_M2 = matriz_usuarios(str(usuario), int(curso), int(prueba))
        # Conversión a formato numpy y eliminación de columnas vacías
        new_M2 = np.array(new_M2)
        columnas_no_cero = [i for i in range(len(W2[0])) if any(isinstance(row[i], list) for row in W2)]
        new_M2_filt = new_M2[:, columnas_no_cero]
        # Llamar a la función proceso_heuristico
        M2, W2 = proceso_herustico(new_M2_filt, W2, str(usuario), 0.2, prueba, curso)
        # Obtener matrices precisas
        M2_A, W2_A = matrices_precisas(M2, W2)
        # Intervención
        S2 = SGD_inference(M2)
        S2_A, _ = matrices_precisas(S2, W2)
        # Guardar la matriz de palabras ampliada
        np.save(f'matrices/palabras/Prueba_{prueba}_ampliada.npy', W2_A)

    return S2_A, W2_A, M2_A


def guardar_resultados(carpeta, nombre_archivo, mi_array):
    """
    Guarda un array de NumPy en un archivo .npy en la carpeta especificada.

    Args:
        carpeta (str): Ruta a la carpeta donde se guardará el archivo.
        nombre_archivo (str): Nombre del archivo sin extensión.
        mi_array (np.array): Array de NumPy que se desea guardar.

    Returns:
        None.
    """
    # Verificar si el archivo ya existe en la carpeta
    contador = 0
    if os.path.exists(os.path.join(carpeta, nombre_archivo+"_0.npy")):
        # Si el archivo ya existe, cambia el nombre
        nuevo_nombre_archivo = nombre_archivo+f'_{contador}.npy'
        
        while os.path.exists(os.path.join(carpeta, nuevo_nombre_archivo)):
            # Agrega un sufijo numérico para diferenciar
            contador += 1
            nuevo_nombre_archivo = nombre_archivo+f'_{contador}.npy'
    
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
    return mean_squared_error(actual_ratings[non_zero_indices], predictions[non_zero_indices])

def SGD_inference(M2):
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

def last_user_answer(usuario, curso, prueba):
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

def cluster_level(usuario,curso):
    df_user_data = pd.read_csv('datos_modelo.csv')
    
    df_user_data = df_user_data[df_user_data['Prueba'].isin([69,71,72])]
    
    df_filtrado = df_user_data[(df_user_data['Usuario'] == usuario) & (df_user_data['Curso'] == int(curso))]
    df_filtrado = df_filtrado.drop_duplicates(subset="Velocidad", keep="last")
    

    velocidad = np.array(df_filtrado['Velocidad'].tolist())
    velocidad = velocidad[:-1]
    cluster = asignar_cluster(velocidad)
    
    return cluster

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
    m_word = resize_matrices_nan(m_word)

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

def dificultades_proceso_heuristico(lista_de_tuplas_filtrada, FIN_PRUEBA):
    """
    Genera una lista con las dificultades que se le van a mostrar al usuario en el proceso heurístico.

    Args:
        lista_de_tuplas_filtrada (list): Lista de tuplas con las dificultades para cada palabra.
        FIN_PRUEBA (int): Número de palabras que se van a mostrar al usuario.

    Returns:
        list: Lista con las tuplas (fila, columna) que representan las dificultades a mostrar.
    """
    # Obtener el número total de tuplas en lista_de_tuplas_filtrada
    total_tuplas = len(lista_de_tuplas_filtrada)

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

# Proceso heurístico: proceso para recoger datos iniciales de los usuarios
def proceso_herustico(matriz_score,matriz_palabras,usuario,porcentaje,prueba,curso):
    G_Modulador = [1.1,0.9]
    matriz_score = np.array(matriz_score)
    
    G_FIN_PRUEBA = int(np.size(matriz_score)*porcentaje)    # Número de palabras que aparecen en la prueba heurística

    # Vector respuesta aleatorio
    r_heuristico = np.random.choice([0, 1], size=G_FIN_PRUEBA, p=[1 -  0.6,  0.6])
    
    n_p_semilla = matriz_score.shape[0]     # Número de palabras semillas

    mostradas = []                          # Palabras que se le han mostrado
    fin_heuristico = 1                      # 1 = Se ejecuta el proceso heurístico
                                            # 0 = Se acaba el proceso heurístico
    
    fin_prueba = 0                          # Contador de palabras

    # Obtengo las posiciones en las que se han generado palabras
    posiciones_cumplen_condicion = [(fila, columna) for fila, fila_datos in enumerate(matriz_palabras) for columna, celda in enumerate(fila_datos) if isinstance(celda, list)]
    lista_de_tuplas_filtrada = [tupla for tupla in posiciones_cumplen_condicion if tupla[1] != 0]
    
    
    posiciones_usadas = dificultades_proceso_heuristico(lista_de_tuplas_filtrada, G_FIN_PRUEBA)
    
    for i, _r in enumerate(r_heuristico):
        pos_word, dificultad = posiciones_usadas[i]
        if matriz_score[pos_word][dificultad] == 0:
            matriz_score[pos_word][dificultad] = dificultad * matriz_score[pos_word][0]

        # Asignar score si ACIERTA
        if _r == 0:
            matriz_score[pos_word][dificultad] = round(matriz_score[pos_word][dificultad] * G_Modulador[1],4)
        # Asignar score si FALLA
        elif _r == 1:
            matriz_score[pos_word][dificultad] = round(matriz_score[pos_word][dificultad] * G_Modulador[0],4)

    # Creo el directorio
    Path('matrices/scores/Curso/'+str(curso)+'/'+str(usuario)+'/'+str(prueba)).mkdir(parents=True, exist_ok=True)

    # Guardo la matriz que se ha generado con el proceso heurístico
    np_score = np.array(matriz_score)
    np.save('matrices/scores/Curso/'+str(curso)+'/'+str(usuario)+'/'+str(prueba)+'/'+str(usuario)+'_'+prueba+'_niño'+'.npy', np_score)

    return matriz_score, matriz_palabras


def intervencion(s_score, m_score,m_word,usuario,respuesta,cluster_level, prueba,curso):
    """
    Ejecuta la intervención para el usuario. Consta de tres fases:
        - Inicio-Facil = 20%
        - Intermedio-Incrementa la dificultad = 60%
        - Final-Facil = 20%
    Args:
        s_score (np.array): Matriz de scores (dificultad) para las palabras.
        m_score (np.array): Matriz de scores actualizada por el proceso heurístico.
        m_word (list): Matriz de palabras.
        usuario (str): Nombre del usuario.
        respuesta (list): Respuesta del usuario al proceso heurístico.
        cluster_level (int): Nivel del usuario (0: bajo, 1: alto).
        prueba (int): Número de la prueba.
        resultados_children (list): Resultados del usuario en la prueba de niños.
        curso (int): Número del curso.
        index (int): Índice para diferenciar entre diferentes ejecuciones.

    Returns:
        np.array: Matriz de scores actualizada con la intervención.
        list: Lista de palabras mostradas al usuario.
        list: Lista de resultados del usuario en la intervención.
        list: Lista de cambios realizados en la matriz de scores.
    """
    NUM_P_INTERVENCION = 100      # Número de palabras que se van a usar en la prueba de intervención
    
    # Variables
    L_PALABRAS_MOSTRADAS = []       # Lista para almacenar las palabras que se le han mostrado al sujeto
    L_RESULTADOS = []               # Lista para guardar los resultados del sujeto
    L_SCORES_MOSTRADOS = []
    L_CAMBIOS_MOSTRADOS = []
    
    print("Comienza el SR")
    print()
    
    Modulador = [1.3,0.9]    # Modificadores de puntuación para acierto y fallo
    
    tendencia_positiva = 0          
    tendencia_negativa = 0
        
    score_inicial = init_score(respuesta,m_score)   # Score de partida
    incremento = score_inicial * 0.15    # El incremento de dificultad es del 15% del score inicial
    
    # posicion vector respuesta
    r = 0
    
    # i = 0 fase incial
    # i = 1 fase intermedia
    
    # Encontrar el valor máximo y mínimo de la matriz
    maximo = np.nanmax(s_score)
    minimo = np.nanmin(s_score)
    
    # result = 'n': Si no encuentra ninguna palabra dentro de este rango buscamos una palabra más fácil a ese score inicial
    # result = 'n'
    for i in range(0,2):

        if i == 0:
            score = score_inicial
            # FASE 1
            #print("[*] Fase Inicial")
            #print()
            porcentaje = 0.2    # Porcantaje de palabras de NUM_P_INTERVENCION que aparecen en cada prueba
            
        elif i == 1:
            porcentaje= 0.6     
            # FASE 2
            #print("[*] Fase Intermedia")
            #print()
            if cluster_level == 1:     # Si el sujeto tiene una tendencia positiva se le aumenta el nivel en la 2º Fase
                score = score - 1                     
            elif cluster_level == 0:   # Si el sujeto tiene una tendencia negativa se le disminuye el nivel en la 2º Fase
                score = 10
            else:
                score = 14                  
            
        for w in range(0,int(NUM_P_INTERVENCION*porcentaje)):
            encontrado = 1
            
            # Bucle que no para hasta que encuentre la siguiente palabra
            while(encontrado == 1):     
                # Recorro la matriz s_score hasta encontrar las palabras que tienen score_inicial
                for i in range(0,len(s_score)):
                    for j in range(0,len(s_score[i])):
                        # Rango de busqueda [score_inicial + 15%, score_inicial - 15%]
                        if round(score - incremento,2)<=s_score[i][j] <= round(score + incremento,2):
                        # Compruebo que se ha generado palabras para ese score
                            if m_word[i][j] != "nan":
                                # Escogemos una palabra al azar de la lista
                                word = m_word[i][j]
                                '''
                                # Si el niño lleva una racha negativa le mostramos una palabra que haya acertado para motivarlo
                                if(tendencia_negativa >= 8):
                                    faciles = []
                                    for index, r in enumerate(L_RESULTADOS):
                                        if r == 0:
                                            faciles.append(L_PALABRAS_MOSTRADAS[index])
                                    if len(faciles):
                                        word = random.choice(faciles)
                                '''
                                if word in L_PALABRAS_MOSTRADAS:
                                    break

                                print(f'   Score: {score} - Palabra: {word}')
                                result = input('   Ha acertado? [y/n]:  ')
                                while result.lower() != 'y' and result.lower() != 'n':
                                    result = input('  No has introducido un valor válido [y/n]:  ')
                                print()

                                L_CAMBIOS_MOSTRADOS.append(j)
                                L_PALABRAS_MOSTRADAS.append(word)
                                encontrado = 0
                                L_SCORES_MOSTRADOS.append(s_score[i][j])
                                # Asignar score si ACIERTA
                                if result == 'y':
                                    tendencia_negativa = 0
                                    tendencia_positiva += 1
                                    L_RESULTADOS.append(0)                                          
                                    
                                    if(tendencia_positiva >= 6):
                                        # Establecemos un nuevo score
                                        score = round(score + (incremento*4), 3)
                                        if(s_score[i][j]*(Modulador[1]) >= 0.1):
                                            m_score[i][j] = round(s_score[i][j]*(Modulador[1]),3)
                                        else:
                                            m_score[i][j] = 0.1
                                        #print(f'   Racha: {"+"}')
                                        #print()
                                        
                                    else:
                                        # Establecemos un nuevo score
                                        if(s_score[i][j]*(Modulador[1]) >= 0.1):
                                            m_score[i][j] = s_score[i][j]*Modulador[1]
                                        else:
                                            m_score[i][j] = 0.1
                                        score = round(score + (incremento),3)
                                        
                                    if score > maximo:
                                        score = maximo
                                # Asignar score si FALLA
                                elif result == 'n':
                                    
                                    tendencia_negativa += 1
                                    tendencia_positiva = 0
                                    
                                    tendencia_negativa += 1
                                    tendencia_positiva = 0
                                    
                                    L_RESULTADOS.append(1)
                                    # Establecemos un nuevo score
                                    
                                    if(tendencia_negativa >= 6):
                                        m_score[i][j] =  round(s_score[i][j]*(Modulador[0]),3)
                                        score = round(score - (incremento * 2),3)
                                        #print(f'   Racha: {"-"}')
                                        #print()
                                        
                                    else:
                                        m_score[i][j] =  s_score[i][j]*Modulador[0]
                                        score = round(score - incremento,3)
                                    
                                    if score < minimo:
                                        score = minimo
                                break
                    # Si se ha mostrado una palabra pasamos al siguiente score
                    if (encontrado == 0):
                        break
                # Si no se ha encontrado ninguna palabra aumentamos o disminuimos el siguiente score hasta encontrar una palabra
                if encontrado == 1:
                    if score >= maximo:
                        result = 1
                    elif score <= minimo:
                        result = 0
                    if result == 0:
                        score = round(score + incremento,3)
                    else:
                        score = round(score - incremento,3)
    # FASE 3
    print("[*] Fase final")
    print()
    p_mostradas = 0
    l_aciertos = []
    
    
    for _r,_w in zip(L_RESULTADOS, L_PALABRAS_MOSTRADAS):
        if _r == 0:
            l_aciertos.append(_w)
    
    
    while(p_mostradas < NUM_P_INTERVENCION*0.2):
        w = random.choice(l_aciertos)
        print(f'   Palabra: {w}')
        result = input('   Ha acertado? [y/n]:  ')
        while result.lower() != 'y' and result.lower() != 'n':
            result = input('  No has introducido un valor válido [y/n]:  ')
        print()
        p_mostradas += 1
        # Asignar score si ACIERTA
        if result.lower() == 'y':
             if(tendencia_positiva >= 5):
                 print(f'   Racha: {"+"}')
                 print()
         # Asignar score si FALLA
        elif result.lower() == 'n':
             
             if(tendencia_negativa >= 5):
                 print(f'   Racha: {"-"}')
       
    # Parte común del path
    common_path = f'matrices/scores/Curso/{curso}/{usuario}/{prueba}'
    usuario_scores_path = f'matrices/scores/Curso/{curso}/{usuario}/{prueba}/{usuario}_{prueba}_niño.npy'
    sgd_scores_path = f'matrices/scores/Curso/{curso}/{usuario}/{prueba}/SGD_{usuario}_{prueba}_niño.npy'

    # Guardar los cambios que se han hecho en las dificultades de las palabras demostradas
    guardar_resultados(common_path, f'cambiosniños', np.array(L_CAMBIOS_MOSTRADOS))
    
    # Guardar las respuestas de los sujetos
    guardar_resultados(common_path, f'resultadosniños', np.array(L_RESULTADOS))
    
    # Guardar los scores de los sujetos
    guardar_resultados(common_path, f'scoresniños', np.array(L_SCORES_MOSTRADOS))

    
    # Guardar la matriz de scores del usuario
    np.save(usuario_scores_path, m_score)
    np.save(sgd_scores_path, s_score)

    
    print("Has terminado la prueba!!!!")
    
# ---------------  MAIN --------------------- #
# ------------------------------------------- #

usuario = "6950474"
curso = "6"
prueba = "74"

# level = cluster_level(usuario, curso, prueba)
level = 1
# Cargo las matrices m_score, m_word
S2_A, W2_A, M2_A = cargar_matrices(usuario,curso,prueba) # DEVUELVE TRES MATRICES MATRICES (s2_ampliado, m_word_ampliado, m2_ampliado)

respuesta = last_user_answer(usuario,curso,prueba)

#resultados_children = np.load("matrices/scores/Curso/"+"6"+"/"+"6591910"+"/"+"69"+"/resultados_4"+".npy")
#sultados_children = simulated_results_binary
intervencion(S2_A, M2_A, W2_A ,usuario, respuesta, level, prueba, curso)
