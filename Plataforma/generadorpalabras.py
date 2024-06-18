import random
import re
import silabeador
import math
import pyphen
import warnings
import pandas as pd

from unidecode import unidecode
from fonemas import Transcription

warnings.filterwarnings("ignore", category=FutureWarning)   # Disable FutureWarning for regex parameter

# --------------- #
# -- Funciones -- #
# --------------- #

def find_vowel_distance(vocal,G_DistanciaMaximaVocal,G_DistanciaFonemasVocales):
    '''''
    vocal -> vocal que se va a cambiar
    G_DistanciaMaximaVocal -> Límite de cambio
    G_DistanciaFonemasVocales -> Matriz distancias entre fonemas
    '''''
    letras_candidatas = []
    # Recorre el diccionario para obtener las distancias entre los fonemas
    for key in G_DistanciaFonemasVocales[vocal].keys():
        if vocal != key:
            # Si la distancia entre los fonemas es menor o igual a la variable 'G_DistanciaMaximaVocal' se guarda la vocal en la lista de candidatas
            if G_DistanciaFonemasVocales[vocal][key] == G_DistanciaMaximaVocal:
                letras_candidatas.append(key)
    return letras_candidatas


def matriz_distancias(matriz):  # Matriz generada al principio del programa
    '''''
    matriz -> G_DistanciaFonemasConsonantes
    '''''

    letras_unicas = []
    # Recorro la matriz para obtener una lista de fonemas que posteriormente van a ser las keys de un diccionario
    for fila in matriz:
        for elemento in fila:
            if isinstance(elemento, str) and elemento not in letras_unicas:
                letras_unicas.append(elemento)

    distancias_letras = {} # Diccionario en el que guardo un diccionario

    # Inicializo el diccionario
    for letra in letras_unicas:
        distancias_letras[letra] = {}

    # Calculo la distancia de todos los fonemas
    for i in range(len(matriz)):
        for j in range(len(matriz[0])):
            if isinstance(matriz[i][j], str):
                fonema_actual = matriz[i][j]             # Fonema al que le voy a calcular la distancia con respecto a los otros fonemas
                for x in range(len(matriz)):
                    for y in range(len(matriz[0])):
                        if isinstance(matriz[x][y], str) and (x, y) != (i, j):  # Condición para que no calcule la distancia asi misma
                            fonema_vecino = matriz[x][y]
                            distancia = math.sqrt((x - i) ** 2 + (y - j) ** 2)  # Calcula la distancia euclidiana entre los dos fonemas
                            distancias_letras[fonema_actual][fonema_vecino] = distancia

    return distancias_letras


def getDistanciaConsonante(silaba_referencia,silaba_modificada,dificultad,distancias_consonantes,G_ListaFonemasConsonantes):
    '''''
    silaba_referencia -> Silaba de la palabra original
    silaba_modificada -> Silaba de que se le ha aplicado un cambio
    dificultad -> G_DistanciaMaximaConsonante_Rango
    distancias_consonantes -> C_DistanciasEntreConsonantes
    G_ListaFonemasConsonantes -> lista de fonemas consonánticos    
    '''''

    pos = 0
    # Obtengo los fonemas de las sílabas y le quito caracteres que no son letras
    fonema_referencia = Transcription(silaba_referencia).phonology.words
    fonema_referencia = [pho.replace('ˈ', '') for pho in fonema_referencia]

    fonema_modificada = Transcription(silaba_modificada).phonology.words
    fonema_modificada = [pho.replace('ˈ', '') for pho in fonema_modificada]

    silaba_candidata = False
    distancia_menor = 0

    # Comprueba si tienen el mismo número de fonemas, si no es asi, si tienen que igualar

    for s_r, s_m in zip(fonema_referencia,fonema_modificada):
        for index, (f_r, f_m) in enumerate(zip(s_r, s_m)):
            if f_r != f_m:
                pos = index
                break

    for s_r, s_m in zip(fonema_referencia,fonema_modificada):
        # ------------ Iguala silabas fonemas --------- #
        if len(s_r) < len(s_m):
            s_r = s_r[:pos] + s_r[pos] + s_r[pos:]
        elif len(s_r) > len(s_m):
            s_m = s_m[:pos] + s_m[pos] + s_m[pos:]
        # --------------------------------------------- #
        # Calcula la distancia de cada fonema
        for l_r, l_m in zip(s_r, s_m):
            if l_r != l_m:
                if l_m in distancias_consonantes:
                    if l_r in G_ListaFonemasConsonantes and l_m in G_ListaFonemasConsonantes:
                        distancia = distancias_consonantes[l_r][l_m]
                        distancia_menor += distancia

        # Si la distancia que obtengo está en el rango de dificultad (G_DistanciaMaximaConsonante_Rango) significa que es un cambio válido
        if distancia_menor == dificultad[1]:
            silaba_candidata = True

    return silaba_candidata


def getDistanciaTotal(palabra_referencia,palabra_modificada,distancias_consonantes,distancias_vocales,G_ListaFonemasConsonantes,G_ListaVocales):
    '''''
    palabra_referencia -> INPUT_PALABRA
    palabra_modificada -> Palabra que queremos calcular la distancia
    distancias_consonantes -> Matriz de las distancias entre los fonemas consonanticos
    distancias_vocales -> Matriz de las distancias entre los fonemas vocálico
    G_ListaFonemasConsonantes -> Lista de los fonemas consonanticos
    G_ListaVocales -> Lista de los fonemas vocálico
    '''''

    pos = 0
    # Obtengo los fonemas de las sílabas y le quito caracteres que no son letras
    fonema_referencia = Transcription(palabra_referencia).phonology.words
    fonema_referencia = [pho.replace('ˈ', '') for pho in fonema_referencia]

    fonema_modificada = Transcription(palabra_modificada).phonology.words
    fonema_modificada = [pho.replace('ˈ', '') for pho in fonema_modificada]

    distancia_menor = 0

    # Comprueba si tienen el mismo número de fonemas, si no es asi, si tienen que igualar

    for s_r, s_m in zip(fonema_referencia,fonema_modificada):
        for index, (f_r, f_m) in enumerate(zip(s_r, s_m)):
            if f_r != f_m:
                pos = index
                break

    for s_r, s_m in zip(fonema_referencia,fonema_modificada):
        # ------------ Iguala silabas fonemas --------- #
        if len(s_r) < len(s_m):
            s_r = s_r[:pos] + s_r[pos] + s_r[pos:]
        elif len(s_r) > len(s_m):
            s_m = s_m[:pos] + s_m[pos] + s_m[pos:]
        # --------------------------------------------- #
        # Calcula la distancia de cada fonema
        for l_r, l_m in zip(s_r, s_m):
            if l_r != l_m:
                if l_m in distancias_consonantes:
                    if l_r in G_ListaFonemasConsonantes and l_m in G_ListaFonemasConsonantes:
                        distancia = distancias_consonantes[l_r][l_m]
                        distancia_menor += distancia
                if l_m in distancias_vocales:

                    if l_r in G_ListaVocales and l_m in G_ListaVocales:
                        distancia = distancias_vocales[l_r][l_m]
                        distancia_menor += distancia

    return distancia_menor


def palabrasCandidatas(lista_palabras,G_Palabras_Corco):
    '''''
    lista_palabras -> Lista de palabras que queremos comprobar si existen en el corco
    G_Palabras_Corco -> Lista de palabras que aparecen en el corco
    '''''

    palabras_corco = []

    for palabra in lista_palabras:
        if palabra in G_Palabras_Corco:
            palabras_corco.append(palabra)

    return palabras_corco


def ordeno_foneticamente(lista_ordenar,palabra_original,C_DistanciasEntreConsonantes,G_DistanciaFonemasVocales,G_ListaFonemasConsonantes,G_ListaVocales):
    '''''
    lista_ordenar -> Lista de palabras que queremos ordenar
    palabra_original -> INPUT_PALABRA
    C_DistanciasEntreConsonantes -> Matriz distancias fonemas consonanticos
    G_DistanciaFonemasVocales -> Matriz distancias fonemas vocálicos
    G_ListaFonemasConsonantes -> Lista fonemas vocálicos
    G_ListaVocales -> Lista fonemas vocálicos
    Utiliza la función getDistanciaTotal como clave para ordenar la lista.
    '''''

    lista_ordenada = sorted(lista_ordenar, key=lambda palabra:getDistanciaTotal(palabra_original,palabra,C_DistanciasEntreConsonantes,G_DistanciaFonemasVocales,G_ListaFonemasConsonantes,G_ListaVocales))

    return lista_ordenada

def buscar_palabra_acento(G_Palabras_sin_Tilde,G_Palabras_Corco, G_Acento_Palabra,palabra, palabra_semilla,G_DIC_TILDE,tilde_original):

    # Busca la palabra en G_Palabras_sin_tilde y obtiene su índice si se encuentra
    try:
        # En el Corco hay palabras que aparecen con tílde y sin tílde. Siempre la palabra sin tílde va delante de la que tiene tílde
        if tilde_original:
            if G_Palabras_sin_Tilde.count(palabra) > 1:
                indice = G_Palabras_sin_Tilde.index(palabra) + 1
            else:
                indice = G_Palabras_sin_Tilde.index(palabra)
                # Si la palabra está en el Corco compruebo que tiene el acento en la misma posición que la palabra original
                if G_Acento_Palabra[G_Palabras_sin_Tilde.index(palabra_semilla)] != G_Acento_Palabra[indice]:
                    return palabra
        else:
            # Si la palabra original no tiene tilde la busco directamente en el Corco
            indice = G_Palabras_sin_Tilde.index(palabra)
            # Si la palabra está en el Corco compruebo que tiene el acento en la misma posición que la palabra original
            if G_Acento_Palabra[G_Palabras_sin_Tilde.index(palabra_semilla)] != G_Acento_Palabra[indice]:
                return palabra

        return G_Palabras_Corco[indice]
    except :
        try:
            # En el Corco hay palabras que aparecen con tílde y sin tílde. Siempre la palabra sin tílde va delante de la que tiene tílde
            if tilde_original:
                if G_Palabras_sin_Tilde.count(palabra_semilla) > 1:
                    indice = G_Palabras_sin_Tilde.index(palabra_semilla) + 1
                else:
                    indice = G_Palabras_sin_Tilde.index(palabra_semilla)
            else:
                indice = G_Palabras_sin_Tilde.index(palabra_semilla)
            # Si la palabra semilla aparece en G_Palabras_sin_Tilde y no en G_Palabras_Corco quiere decir que la palabra lleva tilde.
            # Si la palabra semilla aparece en G_Palabras_sin_Tilde y en G_Palabras_Corco y tilde_original = True significa que esa palabra aparece en el Corco con tílde y sin ella.
            if (palabra_semilla in G_Palabras_sin_Tilde and palabra_semilla not in G_Palabras_Corco) or (palabra_semilla in G_Palabras_sin_Tilde and palabra_semilla in G_Palabras_Corco and tilde_original):
                # Obtengo en que sílaba está el acento
                acento = G_Acento_Palabra[indice]
                pos_acento = re.search(r'T', acento).start()
                # Separo la palabra en sílaba
                silaba = silabeador.Syllabification(palabra).syllables
                new_silaba = []
                # Compruebo que el número de sílabas es mayor o igual a la posición en la que está el acento
                if len(silaba)-1 >= pos_acento:
                    for index, _silaba in enumerate(silaba):
                        # Recorro la palabra hasta llegar a la posición en la que tiene que estar el acento
                        if index == pos_acento:
                            for index2, letra in enumerate(_silaba):
                                # Si letra es una vocal, le añado una tílde
                                if letra in ['a','e','i','o','u']:
                                    tilde = G_DIC_TILDE[letra]
                                    _silaba = list(_silaba)
                                    _silaba[index2] = tilde
                                    _silaba = "".join(_silaba)
                                    break
                        new_silaba.append(_silaba)
                    return "".join(new_silaba)
                else:
                    return palabra
            else:
                return palabra

        except :
            # Si la palabra no aparece el corco no le añado acento
            return palabra  # La palabra no se encontró en G_palabras_sin_tilde


def add_accent(G_Palabras_sin_Tilde, G_Palabras_Corco, G_Acento_Palabra,listado_candidatas, INPUT_PALABRA,G_DIC_TILDE,tilde_original):

    l_accent = []
    l_original = listado_candidatas
    l_sin_modificadas = []
    for word in listado_candidatas:
        l_accent.append(buscar_palabra_acento(G_Palabras_sin_Tilde,G_Palabras_Corco, G_Acento_Palabra,word, INPUT_PALABRA,G_DIC_TILDE,tilde_original))

    if tilde_original:
        for l_a, l_o in zip(l_accent,l_original):
            if l_a == l_o:
                l_sin_modificadas.append(l_o)

        if len(l_sin_modificadas) > 0:
            print()
            print(f'\tNo se ha podido añadir el acento a las siguientes palabras: {l_sin_modificadas}')
            print()

    return l_accent

def genera_palabras(INPUT_PALABRA, INPUT_TIPO_CAMBIO, G_DistanciaMaximaVocal, G_DistanciaMaximaConsonante,G_PSEUDO):
    '''''
    INPUT_PALABRA -> Palabra generadora
    INPUT_TIPO_CAMBIO -> Cambio que se le va a aplicar a la palabra generadora
    '''''


    # ------------------------ #
    # -- Entorno de trabajo -- #
    # ------------------------ #

    dic = pyphen.Pyphen(lang='es')

    # ------------------------ #
    # -- Variables globales -- #
    # ------------------------ #

    G_LOG = 0  # 0 = No se muestran tantos mensajes por pantalla...
    # 1 = Se muestran muchos mensajes por pantalla...

    # 0 = Se intentan proponer palabras que sí que aparecen en el Corco
    # 1 = No se hacen búsquedas en el Corco (alta probabilidad de que se genere una pseudo-palabra)

    G_DistanciaMaximaConsonante_Rango = [0, G_DistanciaMaximaConsonante]

    G_ListaVocales = ['a', 'e', 'i', 'o', 'u']
    #G_ListaFonemasVocales = ['a', 'e', 'i', 'o', 'u']
    G_ListaConsonantes = ['b', 'c', 'd', 'f', 'g', 'h', 'j', 'k', 'l', 'm', 'n', 'ñ', 'p', 'q', 'r', 's', 't', 'v', 'w',
                          'x', 'y', 'z']
    G_ListaConsonantesNoQ = ['b', 'c', 'd', 'f', 'g', 'h', 'j', 'k', 'l', 'm', 'n', 'ñ', 'p', 'r', 's', 't', 'v',
                             'w', 'x', 'y', 'z']
    G_ListaFonemasConsonantes = ['p', 'b', 't', 'd', 'k', 'g', 'f', 'θ', 's', 'ʝ', 'x', 'ʧ', 'm', 'n', 'ɲ', 'l', 'ʎ',
                                 'ɾ', 'r']  # No están ordenados

    G_DIC_TILDE = { 'a': 'á',
                    'e': 'é',
                    'i': 'í',
                    'o': 'ó',
                    'u': 'ú'
    }

    #G_hMuda = ['h']
    #G_FonemashMuda = ['α']

    G_REGLA_ConsonantesAlFinal = ['l', 'r', 's', 'x','z']  # Duda / No se deberia de añadir la n? Ejemplo Carbol, Carbon
    G_REGLA_ConsonantesDobles = ['bl', 'br', 'cl', 'cr', 'dr', 'fl', 'fr', 'gl', 'gr', 'll', 'pl', 'pr', 'tr']
    #G_REGLA_ConsonantesDoblesR = ['rr']
    G_REGLA_SilabasQueQui = ['que', 'quel', 'quen', 'ques', 'qui', 'quil', 'quin', 'quien', 'quis']

    G_Patrones_Consonantes = [r'[aeiou]',
                              r'[aeiou][aeiou]',
                              r'[bcdfghjklmnpqrstvwxyz][aeiou]',
                              r'[aeiou][bcdfghjklmnpqrstvwxyz]',
                              r'[aeiou][aeiou][bcdfghjklmnpqrstvwxyz]',
                              r'[bcdfghjklmnpqrstvwxyz][aeiou][aeiou]',
                              r'[bcdfghjklmnpqrstvwxyz][aeiou][bcdfghjklmnpqrstvwxyz]',
                              r'[bcdfghjklmnpqrstvwxyz][bcdfghjklmnpqrstvwxyz][aeiou]',
                              r'[bcdfghjklmnpqrstvwxyz][aeiou][aeiou][bcdfghjklmnpqrstvwxyz]',
                              r'[bcdfghjklmnpqrstvwxyz][bcdfghjklmnpqrstvwxyz][aeiou][aeiou]',
                              r'[bcdfghjklmnpqrstvwxyz][bcdfghjklmnpqrstvwxyz][aeiou][bcdfghjklmnpqrstvwxyz]',
                              r'[bcdfghjklmnpqrstvwxyz][bcdfghjklmnpqrstvwxyz][aeiou][aeiou][bcdfghjklmnpqrstvwxyz]',
                              r'[gq][u][i][e][n]',
                              r'[bcdfghjklmnpqrstvwxyz][bcdfghjklmnpqrstvwxyz][aeiou][bcdfghjklmnpqrstvwxyz][bcdfghjklmnpqrstvwxyz]'
                              ]
    # --------------------------------------------------- #

    G_Patrones_Consonantes = G_Patrones_Consonantes[::-1]
    if G_LOG:
        print()
        print(
            "[Nota]: No se están considerando terminaciones latinas como Curriculu[m], ni catalanismos/arabismos/judeísmos como Fue[t], Raba[t] ó Arara[t].")
        print()

    G_DistanciaFonemasVocales = {
        'a': {'a': 0, 'e': 1, 'i': 2, 'o': 1, 'u': 2},
        'e': {'a': 1, 'e': 0, 'i': 1, 'o': 2, 'u': 3},
        'i': {'a': 2, 'e': 1, 'i': 0, 'o': 3, 'u': 2},
        'o': {'a': 1, 'e': 2, 'i': 3, 'o': 0, 'u': 1},
        'u': {'a': 2, 'e': 3, 'i': 2, 'o': 1, 'u': 0}}

    G_DistanciaFonemasConsonantes = [[0] * 14 for _ in range(7)]
    G_DistanciaFonemasConsonantes[0][0] = 'p'
    G_DistanciaFonemasConsonantes[0][1] = 'b'
    G_DistanciaFonemasConsonantes[0][6] = 't'
    G_DistanciaFonemasConsonantes[0][7] = 'd'
    G_DistanciaFonemasConsonantes[0][12] = 'k'
    G_DistanciaFonemasConsonantes[0][13] = 'g'
    G_DistanciaFonemasConsonantes[1][2] = 'f'
    G_DistanciaFonemasConsonantes[1][4] = 'θ'
    G_DistanciaFonemasConsonantes[1][8] = 's'
    G_DistanciaFonemasConsonantes[1][11] = 'ʝ'
    G_DistanciaFonemasConsonantes[1][12] = 'x'
    G_DistanciaFonemasConsonantes[2][10] = 'ʧ'
    G_DistanciaFonemasConsonantes[3][1] = 'm'
    G_DistanciaFonemasConsonantes[3][9] = 'n'
    G_DistanciaFonemasConsonantes[3][11] = 'ɲ'
    G_DistanciaFonemasConsonantes[4][9] = 'l'
    G_DistanciaFonemasConsonantes[4][11] = 'ʎ'
    G_DistanciaFonemasConsonantes[5][9] = 'ɾ'
    G_DistanciaFonemasConsonantes[6][9] = 'r'

    # --------------- #
    # ---- Corco ---- #
    # --------------- #

    df_corco = pd.read_csv('new_corco.csv')

    G_Palabras_Corco = df_corco['Lista de ítem'].values.tolist()
    G_Palabras_sin_Tilde = df_corco['Lista de ítems sin tildes'].values.tolist()
    G_Acento_Palabra = df_corco['Posición Acento'].values.tolist()

    C_DistanciasEntreConsonantes = matriz_distancias(G_DistanciaFonemasConsonantes)

    # ==================================================== #
    # ================ PROGRAMA PRINCIPAL ================ #
    # ==================================================== #
    if G_LOG:
        print()
        print(f"==== ESCENARIO {INPUT_TIPO_CAMBIO} ====")
        print()
        if (INPUT_TIPO_CAMBIO == 0):
            print('MODO PRUEBA')
        elif (INPUT_TIPO_CAMBIO == 1):
            print('SUSTITUCIÓN DE UNA VOCAL EN CUALQUIERA DE LOS EXTREMOS DE LA PALABRA')
        elif (INPUT_TIPO_CAMBIO == 2):
            print('SUSTITUCIÓN DE AMBAS VOCALES EN LOS EXTREMOS DE LA PALABRA')
        elif (INPUT_TIPO_CAMBIO == 3):
            print('ELIMINAR VOCALES')
        elif (INPUT_TIPO_CAMBIO == 4):
            print('AÑADIR VOCALES')
        elif (INPUT_TIPO_CAMBIO == 5):
            print('MODIFICAR UNA CONSONANTE')
        elif (INPUT_TIPO_CAMBIO == 6):
            print('ELIMINAR CONSONANTES')
        elif (INPUT_TIPO_CAMBIO == 7):
            print('AÑADIR CONSONANTES')
        print()

        print(f"\t[*] Palabra de entrada: {INPUT_PALABRA}")
    # Quito acento a la palabra
    transformacion_palabra = unidecode(INPUT_PALABRA)
    if INPUT_PALABRA == transformacion_palabra:
        tilde_original = False
    else:
        INPUT_PALABRA = transformacion_palabra
        tilde_original = True
    # [1] Determinamos las sílabas
    syllables = silabeador.Syllabification(INPUT_PALABRA).syllables
    OUTPUT_PALABRA = syllables.copy()
    if G_LOG:
        print(f"\t[*] Descomponemos la palabra de entrada en {len(syllables)} sílabas: {syllables}")

    # [2] Por cada sílaba, determinamos su tipo:
    tipo_silaba_segun_posicion = []
    for _syllable_index, _syllable in enumerate(syllables):
        for _patron_index, _patron in enumerate(G_Patrones_Consonantes):
            if bool(re.search(_patron, _syllable)):
                tipo_silaba_segun_posicion.append(_patron_index)
                break
    if G_LOG:
        print(f"\t[*] Tipos de las sílabas encontradas: {tipo_silaba_segun_posicion}.")

    #################################################################
    #### Se evalúa el tipo  de sílaba para hacer la modificación ####
    #
    # 13.   [0]         #
    # 12.   [00]        #
    # 11.   [c0]        #
    # 10.   [0c]        #
    # 9.    [00c]       #
    # 8.    [c00]       #
    # 7.    [c0c]       #
    # 6.    [cc0]       #
    # 5.    [c00c]      #
    # 4.    [cc00]      #
    # 3.    [cc0c]      #
    # 2.    [cc00c]     #
    # 1.    [c000c]     #
    # 0.    [cc0cc]     #
    #
    #################################################################
    #################################################################

    # [3] Evaluamos cada caso por separado:

    # [3.0] MODO PRUEBA
    # -----------------

    if(INPUT_TIPO_CAMBIO == 0):
        listado_candidatas = []
        pass

    # [3.1] SUSTITUCIÓN DE UNA VOCAL EN CUALQUIERA DE LOS EXTREMOS DE LA PALABRA
    # --------------------------------------------------------------------------

    elif(INPUT_TIPO_CAMBIO == 1):

        _instrucciones_modificacion = {}
        _silabas_candidatas = []

        for id_tipo,val_tipo in enumerate(tipo_silaba_segun_posicion):
            if(val_tipo == 13 or val_tipo == 10):    # [0] [0c]
                _instrucciones_modificacion[id_tipo] = {'pos_vocal_1': 0,
                                                        'pos_vocal_2': 0,
                                                        'listado_candidatas_vocal_1': find_vowel_distance(OUTPUT_PALABRA[id_tipo][0],G_DistanciaMaximaVocal,G_DistanciaFonemasVocales),
                                                        'listado_candidatas_vocal_2': find_vowel_distance(OUTPUT_PALABRA[id_tipo][0],G_DistanciaMaximaVocal,G_DistanciaFonemasVocales)}
                _silabas_candidatas.append(id_tipo)
            elif(val_tipo == 12 or val_tipo == 9):  # [00] [00c]
                _instrucciones_modificacion[id_tipo] = {'pos_vocal_1': 0,
                                                        'pos_vocal_2': 1,
                                                        'listado_candidatas_vocal_1': find_vowel_distance(OUTPUT_PALABRA[id_tipo][0],G_DistanciaMaximaVocal,G_DistanciaFonemasVocales),
                                                        'listado_candidatas_vocal_2': find_vowel_distance(OUTPUT_PALABRA[id_tipo][1],G_DistanciaMaximaVocal,G_DistanciaFonemasVocales)}
                _silabas_candidatas.append(id_tipo)
            elif(val_tipo == 11 or val_tipo == 7):   # [c0] [c0c]
                _instrucciones_modificacion[id_tipo] = {'pos_vocal_1': 1,
                                                        'pos_vocal_2': 1,
                                                        'listado_candidatas_vocal_1': find_vowel_distance(OUTPUT_PALABRA[id_tipo][1],G_DistanciaMaximaVocal,G_DistanciaFonemasVocales),
                                                        'listado_candidatas_vocal_2': find_vowel_distance(OUTPUT_PALABRA[id_tipo][1],G_DistanciaMaximaVocal,G_DistanciaFonemasVocales)}
                _silabas_candidatas.append(id_tipo)
            elif(val_tipo == 8 or val_tipo == 5):   # [c00] [c00c]
                listado_candidatas_vocal_1 = find_vowel_distance(OUTPUT_PALABRA[id_tipo][1],G_DistanciaMaximaVocal,G_DistanciaFonemasVocales)
                listado_candidatas_vocal_2 = find_vowel_distance(OUTPUT_PALABRA[id_tipo][2],G_DistanciaMaximaVocal,G_DistanciaFonemasVocales)
                if OUTPUT_PALABRA[id_tipo][2] in listado_candidatas_vocal_1:
                    listado_candidatas_vocal_1.remove(OUTPUT_PALABRA[id_tipo][2])
                if OUTPUT_PALABRA[id_tipo][1] in listado_candidatas_vocal_2:
                    listado_candidatas_vocal_2.remove(OUTPUT_PALABRA[id_tipo][1])

                _instrucciones_modificacion[id_tipo] = {'pos_vocal_1': 1,
                                                        'pos_vocal_2': 2,
                                                        'listado_candidatas_vocal_1': listado_candidatas_vocal_1,
                                                        'listado_candidatas_vocal_2': listado_candidatas_vocal_2}
                _silabas_candidatas.append(id_tipo)
            elif(val_tipo == 6 or val_tipo == 3 or val_tipo == 0):   # [cc0] [cc0c] [cc0cc]
                _instrucciones_modificacion[id_tipo] = {'pos_vocal_1': 2,
                                                        'pos_vocal_2': 2,
                                                        'listado_candidatas_vocal_1': find_vowel_distance(OUTPUT_PALABRA[id_tipo][2],G_DistanciaMaximaVocal,G_DistanciaFonemasVocales),
                                                        'listado_candidatas_vocal_2': find_vowel_distance(OUTPUT_PALABRA[id_tipo][2],G_DistanciaMaximaVocal,G_DistanciaFonemasVocales)}
                _silabas_candidatas.append(id_tipo)
            elif(val_tipo == 4 or val_tipo == 2):    # [cc00] [cc00c]
                _instrucciones_modificacion[id_tipo] = {'pos_vocal_1': 2,
                                                        'pos_vocal_2': 3,
                                                        'listado_candidatas_vocal_1': find_vowel_distance(OUTPUT_PALABRA[id_tipo][2],G_DistanciaMaximaVocal,G_DistanciaFonemasVocales),
                                                        'listado_candidatas_vocal_2': find_vowel_distance(OUTPUT_PALABRA[id_tipo][3],G_DistanciaMaximaVocal,G_DistanciaFonemasVocales)}
                _silabas_candidatas.append(id_tipo)
            elif(val_tipo == 1):    # [c000c] Nuevo
                # Si la sílaba es Quien/Guien solo se puede sustituir la última vocal
                lista = find_vowel_distance( OUTPUT_PALABRA[id_tipo][3], G_DistanciaMaximaVocal,G_DistanciaFonemasVocales)
                lista.remove(OUTPUT_PALABRA[id_tipo][2])

                _instrucciones_modificacion[id_tipo] = {'pos_vocal_1': 3,
                                                        'pos_vocal_2': 3,
                                                        'listado_candidatas_vocal_1': lista,
                                                        'listado_candidatas_vocal_2': lista}
                _silabas_candidatas.append(id_tipo)

            else:
                print(f"\t[WARNING]: La sílaba {OUTPUT_PALABRA[id_tipo]} tiene una estructura inválida...")

        # -- Determinamos lista de palabras modificadas -- #

        listado_candidatas = []

        for _id_silaba in [0,len(OUTPUT_PALABRA)-1]:

            if(_id_silaba == 0):
                _posicion_reemplazo = _instrucciones_modificacion[_id_silaba]['pos_vocal_1']
                _alternativas_reemplazo = _instrucciones_modificacion[_id_silaba]['listado_candidatas_vocal_1']
            else:
                _posicion_reemplazo = _instrucciones_modificacion[_id_silaba]['pos_vocal_2']
                _alternativas_reemplazo = _instrucciones_modificacion[_id_silaba]['listado_candidatas_vocal_2']

            for _c in _alternativas_reemplazo:

                _palabra_candidata = OUTPUT_PALABRA.copy()
                _silaba_reemplazo = list(_palabra_candidata[_id_silaba])
                _silaba_reemplazo[_posicion_reemplazo] = _c
                _palabra_candidata[_id_silaba] = ''.join(_silaba_reemplazo)
                _palabra_candidata = "".join(_palabra_candidata)

                listado_candidatas.append(_palabra_candidata)


    # [3.2] SUSTITUCIÓN DE AMBAS VOCALES EN LOS EXTREMOS DE LA PALABRA
    # ----------------------------------------------------------------

    elif(INPUT_TIPO_CAMBIO == 2):

        _instrucciones_modificacion = {}
        _silabas_candidatas = []

        for id_tipo,val_tipo in enumerate(tipo_silaba_segun_posicion):
            if(val_tipo == 13 or val_tipo == 10):    # [0] [0c]
                _instrucciones_modificacion[id_tipo] = {'pos_vocal_1': 0,
                                                        'pos_vocal_2': 0,
                                                        'listado_candidatas_vocal_1': find_vowel_distance(OUTPUT_PALABRA[id_tipo][0],G_DistanciaMaximaVocal,G_DistanciaFonemasVocales),
                                                        'listado_candidatas_vocal_2': find_vowel_distance(OUTPUT_PALABRA[id_tipo][0],G_DistanciaMaximaVocal,G_DistanciaFonemasVocales)}
                _silabas_candidatas.append(id_tipo)
            elif(val_tipo == 12 or val_tipo == 9):  # [00] [00c]
                _instrucciones_modificacion[id_tipo] = {'pos_vocal_1': 0,
                                                        'pos_vocal_2': 1,
                                                        'listado_candidatas_vocal_1': find_vowel_distance(OUTPUT_PALABRA[id_tipo][0],G_DistanciaMaximaVocal,G_DistanciaFonemasVocales),
                                                        'listado_candidatas_vocal_2': find_vowel_distance(OUTPUT_PALABRA[id_tipo][1],G_DistanciaMaximaVocal,G_DistanciaFonemasVocales)}
                _silabas_candidatas.append(id_tipo)
            elif(val_tipo == 11 or val_tipo == 7):   # [c0] [c0c]
                _instrucciones_modificacion[id_tipo] = {'pos_vocal_1': 1,
                                                        'pos_vocal_2': 1,
                                                        'listado_candidatas_vocal_1': find_vowel_distance(OUTPUT_PALABRA[id_tipo][1],G_DistanciaMaximaVocal,G_DistanciaFonemasVocales),
                                                        'listado_candidatas_vocal_2': find_vowel_distance(OUTPUT_PALABRA[id_tipo][1],G_DistanciaMaximaVocal,G_DistanciaFonemasVocales)}
                _silabas_candidatas.append(id_tipo)
            elif(val_tipo == 8 or val_tipo == 5):   # [c00] [c00c]
                _instrucciones_modificacion[id_tipo] = {'pos_vocal_1': 1,
                                                        'pos_vocal_2': 2,
                                                        'listado_candidatas_vocal_1': find_vowel_distance(OUTPUT_PALABRA[id_tipo][1],G_DistanciaMaximaVocal,G_DistanciaFonemasVocales),
                                                        'listado_candidatas_vocal_2': find_vowel_distance(OUTPUT_PALABRA[id_tipo][2],G_DistanciaMaximaVocal,G_DistanciaFonemasVocales)}
                _silabas_candidatas.append(id_tipo)
            elif(val_tipo == 6 or val_tipo == 3 or val_tipo == 0):   # [cc0] [cc0c] [cc0cc]
                _instrucciones_modificacion[id_tipo] = {'pos_vocal_1': 2,
                                                        'pos_vocal_2': 2,
                                                        'listado_candidatas_vocal_1': find_vowel_distance(OUTPUT_PALABRA[id_tipo][2],G_DistanciaMaximaVocal,G_DistanciaFonemasVocales),
                                                        'listado_candidatas_vocal_2': find_vowel_distance(OUTPUT_PALABRA[id_tipo][2],G_DistanciaMaximaVocal,G_DistanciaFonemasVocales)}
                _silabas_candidatas.append(id_tipo)
            elif(val_tipo == 4 or val_tipo == 2):    # [cc00] [cc00c]
                _instrucciones_modificacion[id_tipo] = {'pos_vocal_1': 2,
                                                        'pos_vocal_2': 3,
                                                        'listado_candidatas_vocal_1': find_vowel_distance(OUTPUT_PALABRA[id_tipo][2],G_DistanciaMaximaVocal,G_DistanciaFonemasVocales),
                                                        'listado_candidatas_vocal_2': find_vowel_distance(OUTPUT_PALABRA[id_tipo][3],G_DistanciaMaximaVocal,G_DistanciaFonemasVocales)}
                _silabas_candidatas.append(id_tipo)
            elif(val_tipo == 1):                     # [c000c]
                # Si la sílaba es Quien/Guien solo se puede sustituir la última vocal
                lista = find_vowel_distance( OUTPUT_PALABRA[id_tipo][3], G_DistanciaMaximaVocal,G_DistanciaFonemasVocales)
                lista.remove(OUTPUT_PALABRA[id_tipo][2])

                _instrucciones_modificacion[id_tipo] = {'pos_vocal_1': 3,
                                                        'pos_vocal_2': 3,
                                                        'listado_candidatas_vocal_1': lista,
                                                        'listado_candidatas_vocal_2': lista}
                _silabas_candidatas.append(id_tipo)
                _silabas_candidatas.append(id_tipo)
            else:
                print(f"\t[WARNING]: La sílaba {OUTPUT_PALABRA[id_tipo]} tiene una estructura inválida...")

        # -- Determinamos lista de palabras modificadas -- #

        listado_candidatas = []

        _id_silaba_1a = 0
        _posicion_reemplazo_1a = _instrucciones_modificacion[_id_silaba_1a]['pos_vocal_1']
        _alternativas_reemplazo_1a = _instrucciones_modificacion[_id_silaba_1a]['listado_candidatas_vocal_1']

        _id_silaba_2a = len(OUTPUT_PALABRA)-1
        _posicion_reemplazo_2a = _instrucciones_modificacion[_id_silaba_2a]['pos_vocal_2']
        _alternativas_reemplazo_2a = _instrucciones_modificacion[_id_silaba_2a]['listado_candidatas_vocal_2']

        for _c1 in _alternativas_reemplazo_1a:
            for _c2 in _alternativas_reemplazo_2a:
                _palabra_candidata = OUTPUT_PALABRA.copy()
                _silaba_reemplazo_1a = list(_palabra_candidata[_id_silaba_1a])

                _silaba_reemplazo_1a[_posicion_reemplazo_1a] = _c1
                _palabra_candidata[_id_silaba_1a] = ''.join(_silaba_reemplazo_1a)
                _silaba_reemplazo_2a = list(_palabra_candidata[_id_silaba_2a])
                _silaba_reemplazo_2a[_posicion_reemplazo_2a] = _c2
                _palabra_candidata[_id_silaba_2a] = ''.join(_silaba_reemplazo_2a)

                _palabra_candidata = "".join(_palabra_candidata)

                listado_candidatas.append(_palabra_candidata)

    # [3.3] ELIMINAR VOCALES
    # ----------------------

    elif(INPUT_TIPO_CAMBIO == 3):

        _instrucciones_modificacion = {}
        _silabas_candidatas = []

        for id_tipo,val_tipo in enumerate(tipo_silaba_segun_posicion):

            if(val_tipo in [13]):
                # Siempre se puede eliminar alguna vocal
                _instrucciones_modificacion[id_tipo] = [0]
                _silabas_candidatas.append(id_tipo)
            elif(val_tipo == 12 or val_tipo == 9):
                # Siempre se puede eliminar alguna vocal
                _instrucciones_modificacion[id_tipo] = [0,1]
                _silabas_candidatas.append(id_tipo)
            elif(val_tipo == 11):
                if (id_tipo < len(OUTPUT_PALABRA)-1 and tipo_silaba_segun_posicion[id_tipo+1] in [13,12,10,9]):
                    # Solo se puede eliminar si la siguiente sílaba empieza por vocal
                    _instrucciones_modificacion[id_tipo] = [1]
                    _silabas_candidatas.append(id_tipo)
                elif (id_tipo == len(OUTPUT_PALABRA)-1 and tipo_silaba_segun_posicion[id_tipo-1] in [11]):
                    # Solo se puede eliminar si la anterior sílaba tiene la estructura [c0]
                    _instrucciones_modificacion[id_tipo] = [1]
                    _silabas_candidatas.append(id_tipo)
            elif(val_tipo == 10 and id_tipo > 0 and tipo_silaba_segun_posicion[id_tipo-1] in [13,12,11,8,6,4]):
                # Solo se puede eliminar si la anterior sílaba termina por vocal
                _instrucciones_modificacion[id_tipo] = [0]
                _silabas_candidatas.append(id_tipo)
            elif(val_tipo == 8 or val_tipo == 5):
                # Siempre se puede eliminar alguna vocal
                _instrucciones_modificacion[id_tipo] = [1,2]
                _silabas_candidatas.append(id_tipo)
            elif(val_tipo == 6 and id_tipo < len(OUTPUT_PALABRA)-1 and tipo_silaba_segun_posicion[id_tipo+1] in [13,12,10,9]):
                # Solo se puede eliminar si la siguiente sílaba empieza por vocal
                _instrucciones_modificacion[id_tipo] = [2]
                _silabas_candidatas.append(id_tipo)
            elif(val_tipo == 4 or val_tipo == 2):
                # Siempre se puede eliminar alguna vocal
                _instrucciones_modificacion[id_tipo] = [2,3]
                _silabas_candidatas.append(id_tipo)
            # ----------------- Nuevo --------------------- #
            elif(val_tipo == 1):
                # Solo se puede eliminar la vocal i y la e
                _instrucciones_modificacion[id_tipo] = [2,3]
                _silabas_candidatas.append(id_tipo)
            # --------------------------------------------- #
            else:
                # No se puede eliminar ninguna vocal... (Ej.: Caso 1 y Caso 5)
                pass

        # -- Determinamos lista de palabras modificadas -- #

        listado_candidatas = []

        for _id_silaba_candidata in _silabas_candidatas:
            for _pos_vocal in _instrucciones_modificacion[_id_silaba_candidata]:

                _palabra_candidata = OUTPUT_PALABRA.copy()

                _aux_silaba = list(_palabra_candidata[_id_silaba_candidata])
                _aux_silaba[_pos_vocal] = ''

                _palabra_candidata[_id_silaba_candidata] = ''.join(_aux_silaba)
                _palabra_candidata = "".join(_palabra_candidata)

                listado_candidatas.append(_palabra_candidata)

    # [3.4] AÑADIR VOCALES
    # --------------------

    elif(INPUT_TIPO_CAMBIO == 4):

        _instrucciones_modificacion = {}
        _silabas_candidatas = []

        _Fin = len(tipo_silaba_segun_posicion)-1

        for id_tipo,val_tipo in enumerate(tipo_silaba_segun_posicion):

            _posiciones_candidatas = []
            _vocales_candidatas = []


            if(val_tipo == 13):     # [0]
                # Condición: 1, Inserta en: 0, Estructura: [INI]+x+[0]+[c**|FIN]
                # Condición: 2, Inserta en: 0, Estructura: [**c]+x+[0]+[c**|FIN]
                # Condición: 3, Inserta en: 1, Estructura: [INI]+[0]+x+[c**|FIN]
                # Condición: 4, Inserta en: 1, Estructura: [**c]+[0]+x+[c**|FIN]
                if(((id_tipo == 0) or (id_tipo>0 and tipo_silaba_segun_posicion[id_tipo-1] in [10,9,7,5,3,2])) and
                   ((id_tipo == _Fin) or (id_tipo < _Fin and tipo_silaba_segun_posicion[id_tipo+1] in [11,8,7,6,5,4,3,2]))):
                    _vocal_noRep = [vowel for vowel in G_ListaVocales if vowel not in OUTPUT_PALABRA[id_tipo][0]]
                    _posiciones_candidatas.extend([0,1])
                    _vocales_candidatas.extend([_vocal_noRep,_vocal_noRep])
            elif(val_tipo == 12):   # [00]
                pass
            elif(val_tipo == 11):    # [c0]
                # Condición: 1, Inserta en: 0, Estructura: [INI]+x+[c0]
                # Condición: 2, Inserta en: 0, Estructura: [**c]+x+[c0]
                if((id_tipo == 0) or
                   (id_tipo > 0 and tipo_silaba_segun_posicion[id_tipo-1] in [10,9,7,5,3,2])):
                    _posiciones_candidatas.extend([0])
                    _vocales_candidatas.extend([G_ListaVocales])
                # Condición: 3, Inserta en: 0, Estructura: [INI]+[0]+x+[c0]
                if(id_tipo == 1 and tipo_silaba_segun_posicion[id_tipo-1] == 12):
                    _vocal_noRep = [vowel for vowel in G_ListaVocales if vowel not in OUTPUT_PALABRA[id_tipo-1][0]]
                    _posiciones_candidatas.extend([0])
                    _vocales_candidatas.extend([_vocal_noRep])
                # Condición: 4, Inserta en: 0, Estructura: [*c0]+x+[c0]
                if(id_tipo > 0 and tipo_silaba_segun_posicion[id_tipo-1] in [11,6]):
                    _vocal_noRep = [vowel for vowel in G_ListaVocales if vowel not in OUTPUT_PALABRA[id_tipo-1][-1]]
                    _posiciones_candidatas.extend([0])
                    _vocales_candidatas.extend([_vocal_noRep])
                # Condición: 5, Inserta en: 1|2, Estructura: [c0]+x+[FIN]
                # Condición: 6, Inserta en: 1|2, Estructura: [c0]+x+[c**]
                if((id_tipo == _Fin) or
                   (id_tipo < _Fin and tipo_silaba_segun_posicion[id_tipo+1] in [11,8,7,6,5,4,3,2])):
                    _vocal_noRep = [vowel for vowel in G_ListaVocales if vowel not in OUTPUT_PALABRA[id_tipo][-1]]
                    _posiciones_candidatas.extend([1,2])
                    _vocales_candidatas.extend([_vocal_noRep,_vocal_noRep])
            elif(val_tipo == 10):    # [0c]
                # Condición: 1, Inserta en: 0|1, Estructura: [INI]+x+[0c]
                # Condición: 2, Inserta en: 0|1, Estructura: [**c]+x+[0c]
                if((id_tipo == 0) or
                   (id_tipo > 0 and tipo_silaba_segun_posicion[id_tipo-1] in [10,9,7,5,3,2])):
                    _vocal_noRep = [vowel for vowel in G_ListaVocales if vowel not in OUTPUT_PALABRA[id_tipo][0]]
                    _posiciones_candidatas.extend([0,1])
                    _vocales_candidatas.extend([_vocal_noRep,_vocal_noRep])
                # Condición: 3, Inserta en: 2, Estructura: [0c]+x+[FIN]
                # Condición: 4, Inserta en: 2, Estructura: [0c]+x+[c**]
                if((id_tipo == _Fin) or
                   (id_tipo < _Fin and tipo_silaba_segun_posicion[id_tipo+1] in [11,8,7,6,5,4,3,2])):
                    _posiciones_candidatas.extend([2])
                    _vocales_candidatas.extend([G_ListaVocales])
            elif(val_tipo == 9):    # [00c]
                # Condición: 1, Inserta en: 3, Estructura: [00c]+x+[FIN]
                # Condición: 2, Inserta en: 3, Estructura: [00c]+x+[c**]
                if((id_tipo == _Fin) or
                   (id_tipo < _Fin and tipo_silaba_segun_posicion[id_tipo+1] in [11,8,7,6,5,4,3,2])):
                    _posiciones_candidatas.extend([3])
                    _vocales_candidatas.extend([G_ListaVocales])
            elif(val_tipo == 8):    # [c00]
                # Condición: 1, Inserta en: 0, Estructura: [INI]+x+[c00]
                # Condición: 2, Inserta en: 0, Estructura: [**c]+x+[c00]
                if((id_tipo == 0) or
                   (id_tipo > 0 and tipo_silaba_segun_posicion[id_tipo-1] in [10,9,7,5,3,2])):
                    _posiciones_candidatas.extend([0])
                    _vocales_candidatas.extend([G_ListaVocales])
                # Condición: 3, Inserta en: 0, Estructura: [INI]+[0]+x+[c00]
                if(id_tipo == 1 and tipo_silaba_segun_posicion[id_tipo-1] == 10):
                    _vocal_noRep = [vowel for vowel in G_ListaVocales if vowel not in OUTPUT_PALABRA[id_tipo-1][0]]
                    _posiciones_candidatas.extend([0])
                    _vocales_candidatas.extend([_vocal_noRep])
                # Condición: 4, Inserta en: 0, Estructura: [*c0]+x+[c00]
                if(id_tipo > 0 and tipo_silaba_segun_posicion[id_tipo-1] in [11,6]):
                    # ------------ NUEVO --------------- #
                    # Condición para estructura [qui/e, gui/e]
                    if ("ui" in OUTPUT_PALABRA[id_tipo] or "ue" in OUTPUT_PALABRA[id_tipo]):
                        _vocal_noRep = [vowel for vowel in G_ListaVocales if vowel not in OUTPUT_PALABRA[id_tipo]]
                        _posiciones_candidatas.extend([3])
                        _vocales_candidatas.extend([_vocal_noRep])
                    # ---------------------------------- #
                    else:
                        _vocal_noRep = [vowel for vowel in G_ListaVocales if vowel not in OUTPUT_PALABRA[id_tipo-1][-1]]
                        _posiciones_candidatas.extend([0])
                        _vocales_candidatas.extend([_vocal_noRep])
                # ------------ NUEVO --------------- #
                # Condición: 5, Inserta en 3, Estructura: [qui/e, gui/e]
                if ("ui" in OUTPUT_PALABRA[0] or "ue" in OUTPUT_PALABRA[0]):
                    _vocal_noRep = [vowel for vowel in G_ListaVocales if vowel not in OUTPUT_PALABRA[id_tipo]]
                    _posiciones_candidatas.extend([3])
                    _vocales_candidatas.extend([_vocal_noRep])
                # ---------------------------------- #
            elif(val_tipo == 7):    # [c0c]
                # Condición: 1, Inserta en: 0, Estructura: [INI]+x+[c0c]
                # Condición: 2, Inserta en: 0, Estructura: [**c]+x+[c0c]
                if((id_tipo == 0) or
                   (id_tipo > 0 and tipo_silaba_segun_posicion[id_tipo-1] in [10,9,7,5,3,2])):
                    _posiciones_candidatas.extend([0])
                    _vocales_candidatas.extend([G_ListaVocales])
                # Condición: 3, Inserta en: 0, Estructura: [*c0]+x+[c0c]
                if(id_tipo > 0 and tipo_silaba_segun_posicion[id_tipo-1] in [11,6]):
                    _vocal_noRep = [vowel for vowel in G_ListaVocales if vowel not in OUTPUT_PALABRA[id_tipo-1][-1]]
                    _posiciones_candidatas.extend([0])
                    _vocales_candidatas.extend([_vocal_noRep])
                # Condición: 4, Inserta en: 0, Estructura: [INI]+[0]+x+[c0c]
                if(id_tipo == 1 and tipo_silaba_segun_posicion[id_tipo-1] == 12):
                    _vocal_noRep = [vowel for vowel in G_ListaVocales if vowel not in OUTPUT_PALABRA[id_tipo-1][0]]
                    _posiciones_candidatas.extend([0])
                    _vocales_candidatas.extend([_vocal_noRep])
                # Condición: 5, Inserta en: 1|2, Estructura: [cx0xc]
                _vocal_noRep = [vowel for vowel in G_ListaVocales if vowel not in OUTPUT_PALABRA[id_tipo][1]]
                _posiciones_candidatas.extend([1,2])
                _vocales_candidatas.extend([_vocal_noRep,_vocal_noRep])
                # Condición: 5, Inserta en: 3, Estructura: [c0c]+x+[FIN]
                # Condición: 6, Inserta en: 3, Estructura: [c0c]+x+[c**]
                if((id_tipo == _Fin) or
                   (id_tipo < _Fin and tipo_silaba_segun_posicion[id_tipo+1] in [11,8,7,6,5,4,3,2])):
                    _posiciones_candidatas.extend([3])
                    _vocales_candidatas.extend([G_ListaVocales])
            elif(val_tipo == 6):    # [cc0]
                # Condición: 1, Inserta en: 0, Estructura: [INI]+x+[cc0]
                # Condición: 2, Inserta en: 0, Estructura: [**c]+x+[cc0]
                if((id_tipo == 0) or
                   (id_tipo > 0 and tipo_silaba_segun_posicion[id_tipo-1] in [10,9,7,5,3,2])):
                    _posiciones_candidatas.extend([0])
                    _vocales_candidatas.extend([G_ListaVocales])
                # Condición: 3, Inserta en: 0, Estructura: [*c0]+x+[cc0]
                if(id_tipo > 0 and tipo_silaba_segun_posicion[id_tipo-1] in [11,6]):
                    _vocal_noRep = [vowel for vowel in G_ListaVocales if vowel not in OUTPUT_PALABRA[id_tipo-1][-1]]
                    _posiciones_candidatas.extend([0])
                    _vocales_candidatas.extend([_vocal_noRep])
                # Condición: 4, Inserta en: 0, Estructura: [INI]+[0]+x+[cc0]
                if(id_tipo == 1 and tipo_silaba_segun_posicion[id_tipo-1] == 12):
                    _vocal_noRep = [vowel for vowel in G_ListaVocales if vowel not in OUTPUT_PALABRA[id_tipo-1][0]]
                    _posiciones_candidatas.extend([0])
                    _vocales_candidatas.extend([_vocal_noRep])
                # Condición: 5, Inserta en: 1, Estructura: [c+x+c0] (Siempre cumple)
                _posiciones_candidatas.extend([1])
                _vocales_candidatas.extend([G_ListaVocales])
                # Condición: 6, Inserta en: 2|3, Estructura: [cc0]+x+[FIN]
                # Condición: 7, Inserta en: 2|3, Estructura: [cc0]+x+[c**]
                if((id_tipo == _Fin) or
                   (id_tipo < _Fin and tipo_silaba_segun_posicion[id_tipo+1] in [11,8,7,6,5,4,3,2])):
                    _vocal_noRep = [vowel for vowel in G_ListaVocales if vowel not in OUTPUT_PALABRA[id_tipo][-1]]
                    _posiciones_candidatas.extend([2,3])
                    _vocales_candidatas.extend([_vocal_noRep,_vocal_noRep])
            elif(val_tipo == 5):    # [c00c]
                # Condición: 1, Inserta en: 0, Estructura: [INI]+x+[c00c]
                # Condición: 2, Inserta en: 0, Estructura: [**c]+x+[c00c]
                if((id_tipo == 0) or
                   (id_tipo > 0 and tipo_silaba_segun_posicion[id_tipo-1] in [10,9,7,5,3,2])):
                    _posiciones_candidatas.extend([0])
                    _vocales_candidatas.extend([G_ListaVocales])
                # Condición: 3, Inserta en: 0, Estructura: [*c0]+x+[cc0]
                if(id_tipo > 0 and tipo_silaba_segun_posicion[id_tipo-1] in [11,6]):
                    _vocal_noRep = [vowel for vowel in G_ListaVocales if vowel not in OUTPUT_PALABRA[id_tipo-1][-1]]
                    _posiciones_candidatas.extend([0])
                    _vocales_candidatas.extend([_vocal_noRep])
                # Condición: 4, Inserta en: 0, Estructura: [INI]+[0]+x+[c00c]
                if(id_tipo == 1 and tipo_silaba_segun_posicion[id_tipo-1] == 12):
                    _vocal_noRep = [vowel for vowel in G_ListaVocales if vowel not in OUTPUT_PALABRA[id_tipo-1][0]]
                    _posiciones_candidatas.extend([0])
                    _vocales_candidatas.extend([_vocal_noRep])
                # Condición: 5, Inserta en: 4, Estructura: [c00c]+x+[FIN]
                # Condición: 6, Inserta en: 4, Estructura: [c00c]+x+[c**]
                if((id_tipo == _Fin) or
                   (id_tipo < _Fin and tipo_silaba_segun_posicion[id_tipo+1] in [11,8,7,6,5,4,3,2])):
                    _posiciones_candidatas.extend([4])
                    _vocales_candidatas.extend([G_ListaVocales])

                # Condición: 7, Estructura [que/qui/gui/gue + derivadas]
                if ("ui" in OUTPUT_PALABRA[id_tipo] or "ue" in OUTPUT_PALABRA[id_tipo]):
                    _vocal_noRep = [vowel for vowel in G_ListaVocales if vowel not in OUTPUT_PALABRA[id_tipo]]
                    _posiciones_candidatas.extend([3])
                    _vocales_candidatas.extend([_vocal_noRep])

            elif(val_tipo == 4):    # [cc00]
                # Condición: 1, Inserta en: 0, Estructura: [INI]+x+[cc00]
                # Condición: 2, Inserta en: 0, Estructura: [**c]+x+[cc00]
                if((id_tipo == 0) or
                   (id_tipo > 0 and tipo_silaba_segun_posicion[id_tipo-1] in [10,9,7,5,3,2])):
                    _posiciones_candidatas.extend([0])
                    _vocales_candidatas.extend([G_ListaVocales])
                # Condición: 3, Inserta en: 0, Estructura: [*c0]+x+[cc00]
                if((id_tipo > 0 and tipo_silaba_segun_posicion[id_tipo-1] in [11,6])):
                    _vocal_noRep = [vowel for vowel in G_ListaVocales if vowel not in OUTPUT_PALABRA[id_tipo-1][-1]]
                    _posiciones_candidatas.extend([0])
                    _vocales_candidatas.extend([_vocal_noRep])
                # Condición: 4, Inserta en: 0, Estructura: [INI]+[0]+x+[cc00]
                if(id_tipo == 1 and tipo_silaba_segun_posicion[id_tipo-1] == 12):
                    _vocal_noRep = [vowel for vowel in G_ListaVocales if vowel not in OUTPUT_PALABRA[id_tipo-1][0]]
                    _posiciones_candidatas.extend([0])
                    _vocales_candidatas.extend([_vocal_noRep])
                # Condición: 5, Inserta en: 1, Estructura: [c+x+c00] (Siempre cumple)
                _posiciones_candidatas.extend([1])
                _vocales_candidatas.extend([G_ListaVocales])
            elif(val_tipo == 3): # [cc0c]
                # Condición: 1, Inserta en: 0, Estructura: [INI]+x+[cc0c]
                # Condición: 2, Inserta en: 0, Estructura: [**c]+x+[cc0c]
                if((id_tipo == 0) or
                   (id_tipo > 0 and tipo_silaba_segun_posicion[id_tipo-1] in [10,9,7,5,3,2])):
                    _posiciones_candidatas.extend([0])
                    _vocales_candidatas.extend([G_ListaVocales])
                # Condición: 3, Inserta en: 0, Estructura: [*c0]+x+[cc0c]
                if((id_tipo > 0 and tipo_silaba_segun_posicion[id_tipo-1] in [11,6])):
                    _vocal_noRep = [vowel for vowel in G_ListaVocales if vowel not in OUTPUT_PALABRA[id_tipo-1][-1]]
                    _posiciones_candidatas.extend([0])
                    _vocales_candidatas.extend([_vocal_noRep])
                # Condición: 4, Inserta en: 0, Estructura: [INI]+[0]+x+[cc00]
                if(id_tipo == 1 and tipo_silaba_segun_posicion[id_tipo-1] == 12):
                    _vocal_noRep = [vowel for vowel in G_ListaVocales if vowel not in OUTPUT_PALABRA[id_tipo-1][0]]
                    _posiciones_candidatas.extend([0])
                    _vocales_candidatas.extend([_vocal_noRep])
                # Condición: 5, Inserta en: 1, Estructura: [c+x+c0c] (Siempre cumple)
                _posiciones_candidatas.extend([1])
                _vocales_candidatas.extend([G_ListaVocales])
                # Condición: 6, Inserta en: 2|3, Estructura: [cc+(x|0)+c]
                _vocal_noRep = [vowel for vowel in G_ListaVocales if vowel not in OUTPUT_PALABRA[id_tipo][2]]
                _posiciones_candidatas.extend([2,3])
                _vocales_candidatas.extend([_vocal_noRep,_vocal_noRep])
                # Condición: 7, Inserta en: 4, Estructura: [cc0c]+x+[END]
                # Condición: 8, Inserta en: 4, Estructura: [cc0c]+x+[c**]
                if((id_tipo == _Fin) or
                   (id_tipo < _Fin and tipo_silaba_segun_posicion[id_tipo+1] in [11,8,7,6,5,4,3,2])):
                    _posiciones_candidatas.extend([4])
                    _vocales_candidatas.extend([G_ListaVocales])
            elif(val_tipo == 2): # [cc00c]
                # Condición: 1, Inserta en: 0, Estructura: [INI]+x+[cc00c]
                # Condición: 2, Inserta en: 0, Estructura: [**c]+x+[cc00c]
                if((id_tipo == 0) or
                   (id_tipo > 0 and tipo_silaba_segun_posicion[id_tipo-1] in [10,9,7,5,3,2])):
                    _posiciones_candidatas.extend([0])
                    _vocales_candidatas.extend([G_ListaVocales])
                # Condición: 3, Inserta en: 0, Estructura: [*c0]+x+[cc0c]
                if((id_tipo > 0 and tipo_silaba_segun_posicion[id_tipo-1] in [11,6])):
                    _vocal_noRep = [vowel for vowel in G_ListaVocales if vowel not in OUTPUT_PALABRA[id_tipo-1][-1]]
                    _posiciones_candidatas.extend([0])
                    _vocales_candidatas.extend([_vocal_noRep])
                # Condición: 4, Inserta en: 0, Estructura: [INI]+[0]+x+[cc00]
                if(id_tipo == 1 and tipo_silaba_segun_posicion[id_tipo-1] == 12):
                    _vocal_noRep = [vowel for vowel in G_ListaVocales if vowel not in OUTPUT_PALABRA[id_tipo-1][0]]
                    _posiciones_candidatas.extend([0])
                    _vocales_candidatas.extend([_vocal_noRep])
                # Condición: 5, Inserta en: 1, Estructura: [c+x+c00c] (Siempre cumple)
                _posiciones_candidatas.extend([1])
                _vocales_candidatas.extend([G_ListaVocales])
                # Condición: 6, Inserta en: 5, Estructura: [cc00c]+x+[END]
                # Condición: 7, Inserta en: 5, Estructura: [cc00c]+x+[c**]
                if((id_tipo == _Fin) or
                   (id_tipo < _Fin and tipo_silaba_segun_posicion[id_tipo+1] in [11,8,7,6,5,4,3,2])):
                    _posiciones_candidatas.extend([5])
                    _vocales_candidatas.extend([G_ListaVocales])
            elif(val_tipo == 0):
                # Condición: 1, Inserta en: 0, Estructura: [INI]+x+[cc0c]
                # Condición: 2, Inserta en: 0, Estructura: [**c]+x+[cc0c]
                if((id_tipo == 0) or
                   (id_tipo > 0 and tipo_silaba_segun_posicion[id_tipo-1] in [10,9,7,5,3,2])):
                    _posiciones_candidatas.extend([0])
                    _vocales_candidatas.extend([G_ListaVocales])
                # Condición: 3, Inserta en: 0, Estructura: [*c0]+x+[cc0c]
                if((id_tipo > 0 and tipo_silaba_segun_posicion[id_tipo-1] in [11,6])):
                    _vocal_noRep = [vowel for vowel in G_ListaVocales if vowel not in OUTPUT_PALABRA[id_tipo-1][-1]]
                    _posiciones_candidatas.extend([0])
                    _vocales_candidatas.extend([_vocal_noRep])
                # Condición: 4, Inserta en: 0, Estructura: [INI]+[0]+x+[cc00]
                if(id_tipo == 1 and tipo_silaba_segun_posicion[id_tipo-1] == 12):
                    _vocal_noRep = [vowel for vowel in G_ListaVocales if vowel not in OUTPUT_PALABRA[id_tipo-1][0]]
                    _posiciones_candidatas.extend([0])
                    _vocales_candidatas.extend([_vocal_noRep])
                # Condición: 5, Inserta en: 1, Estructura: [c+x+c0c] (Siempre cumple)
                _posiciones_candidatas.extend([1])
                _vocales_candidatas.extend([G_ListaVocales])
                # Condición: 6, Inserta en: 2|3, Estructura: [cc+(x|0)+c]
                _vocal_noRep = [vowel for vowel in G_ListaVocales if vowel not in OUTPUT_PALABRA[id_tipo][2]]
                _posiciones_candidatas.extend([2,3])
                _vocales_candidatas.extend([_vocal_noRep,_vocal_noRep])
                # Condición: 7, Inserta en: 4, Estructura: [cc0c]+x+[END]
                # Condición: 8, Inserta en: 4, Estructura: [cc0c]+x+[c**]
                if((id_tipo == _Fin) or
                   (id_tipo < _Fin and tipo_silaba_segun_posicion[id_tipo+1] in [11,8,7,6,5,4,3,2])):
                    _posiciones_candidatas.extend([4])
                    _vocales_candidatas.extend([G_ListaVocales])

            else:                   # Casos no contemplados...
                pass

            if(len(_posiciones_candidatas)>0):
                _instrucciones_modificacion[id_tipo] = {'pos': _posiciones_candidatas,
                                                        'val': _vocales_candidatas}
                _silabas_candidatas.append(id_tipo)

        # -- Determinamos lista de palabras modificadas -- #

        listado_candidatas = []

        for _posicion_silaba in _silabas_candidatas:

            pos_list = _instrucciones_modificacion[_posicion_silaba]['pos']
            val_list = _instrucciones_modificacion[_posicion_silaba]['val']

            for pos, val in zip(pos_list, val_list):

                for letra in val:

                    _palabra_candidata = OUTPUT_PALABRA.copy()
                    _aux_silaba = list(_palabra_candidata[_posicion_silaba])

                    if(pos==0):
                        _aux_silaba = [letra]+_aux_silaba
                    elif(pos==len(_aux_silaba)):
                        _aux_silaba = _aux_silaba+[letra]
                    else:
                        _aux_silaba = _aux_silaba[:pos]+[letra]+_aux_silaba[pos:]

                    _palabra_candidata[_posicion_silaba] = ''.join(_aux_silaba)
                    _palabra_candidata = "".join(_palabra_candidata)

                    listado_candidatas.append(_palabra_candidata)

    # [3.5] MODIFICAR UNA CONSONANTE
    # ------------------------------

    elif(INPUT_TIPO_CAMBIO == 5):

        _instrucciones_modificacion = {}
        _silabas_candidatas = []

        _Fin = len(tipo_silaba_segun_posicion)-1

        for id_tipo,val_tipo in enumerate(tipo_silaba_segun_posicion):

            _posiciones_candidatas = []
            _consonantes_candidatas = []


            if(val_tipo == 13):                     # [0]
                pass
            elif(val_tipo == 12):                   # [00]
                pass
            elif(val_tipo == 11):                    # [c0]

                # if 'h' in OUTPUT_PALABRA[id_tipo]:
                #     fonetica = setHfonema(_syllable)
                # else:
                #     fonetica = Transcription(_syllable)
                #     fonetica = fonetica.phonology.syllables
                #     fonetica = [syl.replace('ˈ', '') for syl in fonetica]

                _posiciones_candidatas.extend([0])
                _consonantes_candidatas.extend(G_ListaConsonantesNoQ)
            elif(val_tipo == 10):                    # [0c]

                _posiciones_candidatas.extend([1])
                _consonantes_candidatas.extend(G_REGLA_ConsonantesAlFinal)
            elif(val_tipo == 9):                    # [00c]

                _posiciones_candidatas.extend([2])
                _consonantes_candidatas.extend(G_REGLA_ConsonantesAlFinal)
            elif(val_tipo == 8):                    # [c00]

                if(OUTPUT_PALABRA[id_tipo] not in G_REGLA_SilabasQueQui):
                    _posiciones_candidatas.extend([0])
                    _consonantes_candidatas.extend(G_ListaConsonantesNoQ)
            elif(val_tipo == 7):                    # [c0c]

                _posiciones_candidatas.extend([0,2])
                _consonantes_candidatas.extend([G_ListaConsonantesNoQ,G_REGLA_ConsonantesAlFinal])

            elif(val_tipo == 6 or val_tipo == 4):   # [cc0] [cc00]

                if(id_tipo > 0 and OUTPUT_PALABRA[id_tipo-1][-1:] in G_ListaVocales):
                    _posiciones_candidatas.extend(['d'])
                    _consonantes_candidatas.extend(G_REGLA_ConsonantesDobles + ['rr'])
                else:
                    _posiciones_candidatas.extend(['d'])
                    _consonantes_candidatas.extend(G_REGLA_ConsonantesDobles)
            elif(val_tipo == 5):                    # [c00c]

                if(OUTPUT_PALABRA[id_tipo] not in G_REGLA_SilabasQueQui):
                    _posiciones_candidatas.extend([0,3])
                    _consonantes_candidatas.extend([G_ListaConsonantesNoQ,G_REGLA_ConsonantesAlFinal])
                else:
                    _posiciones_candidatas.extend(['Q'])
                    _consonantes_candidatas.extend([word for word in G_REGLA_SilabasQueQui if len(word) == 4 and word != OUTPUT_PALABRA[id_tipo]])
            elif(val_tipo == 3):                    # [cc0c]

                if(id_tipo > 0 and OUTPUT_PALABRA[id_tipo-1][-1:] in G_ListaVocales):
                    _posiciones_candidatas.extend(['d',3])
                    _consonantes_candidatas.extend([G_REGLA_ConsonantesDobles + ['rr'],G_REGLA_ConsonantesAlFinal])
                else:
                    _posiciones_candidatas.extend(['d',3])
                    _consonantes_candidatas.extend([G_REGLA_ConsonantesDobles,G_REGLA_ConsonantesAlFinal])
            elif(val_tipo == 2):                    # [cc00c]

                if(id_tipo > 0 and OUTPUT_PALABRA[id_tipo-1][-1:] in G_ListaVocales):
                    _posiciones_candidatas.extend(['d',4])
                    _consonantes_candidatas.extend([G_REGLA_ConsonantesDobles + ['rr'],G_REGLA_ConsonantesAlFinal])
                else:
                    _posiciones_candidatas.extend(['d',4])
                    _consonantes_candidatas.extend([G_REGLA_ConsonantesDobles,G_REGLA_ConsonantesAlFinal])
            elif(val_tipo == 1):                    # [c000c]
                _posiciones_candidatas.extend([0, 4])
                _consonantes_candidatas.extend([G_ListaConsonantesNoQ, G_REGLA_ConsonantesAlFinal])
            elif(val_tipo == 0):                    # [cc0cc]

                if(id_tipo > 0 and OUTPUT_PALABRA[id_tipo-1][-1:] in G_ListaVocales):
                    _posiciones_candidatas.extend(['d',4])
                    _consonantes_candidatas.extend([G_REGLA_ConsonantesDobles + ['rr'],G_REGLA_ConsonantesAlFinal])
                else:
                    _posiciones_candidatas.extend(['d',4])
                    _consonantes_candidatas.extend([G_REGLA_ConsonantesDobles,G_REGLA_ConsonantesAlFinal])
            else:
                pass

            if(len(_posiciones_candidatas)>0):
                _instrucciones_modificacion[id_tipo] = {'pos': _posiciones_candidatas,
                                                        'val': _consonantes_candidatas}
                _silabas_candidatas.append(id_tipo)

        # -- Determinamos lista de palabras modificadas -- #

        listado_candidatas = []

        for _posicion_silaba in _silabas_candidatas:

            pos_list = _instrucciones_modificacion[_posicion_silaba]['pos']
            val_list = _instrucciones_modificacion[_posicion_silaba]['val']


            for id_pos,ival_pos in enumerate(pos_list):
                # -------------- Nuevo ---------------- #
                if len(syllables) > _posicion_silaba+1:
                    next_syllabes = syllables[_posicion_silaba+1]
                    syllable = syllables[_posicion_silaba]
                    if next_syllabes[0] in ['p','b'] and syllable[-1] == 'm':
                        val_list[id_pos].append('n')
                # ------------------------------------- #
                if(len(pos_list)>1):
                    val = val_list[id_pos]
                else:
                    val = val_list

                for letra in val:
                    _palabra_candidata = OUTPUT_PALABRA.copy()
                    # ---- NUEVO ----
                    _silaba = _palabra_candidata[_posicion_silaba]
                    _new_silaba = ""
                    # ---------------
                    _aux_silaba = list(_palabra_candidata[_posicion_silaba])

                    if(ival_pos == 'd'):
                        _aux_silaba = [letra]+_aux_silaba[2:]
                        _new_silaba = ''.join(_aux_silaba)

                    elif(ival_pos == 'Q'):
                        _aux_silaba = [letra]
                        _new_silaba = ''.join(_aux_silaba)

                    else:
                        # ----------- NUEVO --------------- #
                        if letra == 'q' and _silaba[ival_pos+1] != 'u':
                            _aux_silaba[ival_pos] = 'qu'
                        else:
                            _aux_silaba[ival_pos] = letra
                        _new_silaba = ''.join(_aux_silaba)
                        # --------------------------------- #

                    _palabra_candidata[_posicion_silaba] = ''.join(_aux_silaba)
                    _palabra_candidata = "".join(_palabra_candidata)
                    if(_palabra_candidata not in [INPUT_PALABRA]):
                        if (getDistanciaConsonante(_silaba,_new_silaba,G_DistanciaMaximaConsonante_Rango,C_DistanciasEntreConsonantes,G_ListaFonemasConsonantes)):
                            listado_candidatas.append(_palabra_candidata)

    # [3.6] ELIMINAR CONSONANTES
    # --------------------------

    elif(INPUT_TIPO_CAMBIO == 6):

        _instrucciones_modificacion = {}
        _silabas_candidatas = []

        _Fin = len(tipo_silaba_segun_posicion)-1

        for id_tipo,val_tipo in enumerate(tipo_silaba_segun_posicion):

            _posiciones_candidatas = []

            if(val_tipo == 13):                     # [0]
                pass
            elif(val_tipo == 12):                   # [00]
                pass
            elif(val_tipo == 11):                    # [c0]
                # No se permite cuando: [*00]+[_c_0]
                # No se permite cuando: [**0]+[_c_0]+[0**]
                # No se permite cuando: [**0]+[0]+[_c_0]
                if(id_tipo > 0 and tipo_silaba_segun_posicion[id_tipo-1] in [12,8,4]):
                    pass
                elif(id_tipo > 0 and tipo_silaba_segun_posicion[id_tipo-1] in [13,12,11,8,6,4] and id_tipo < _Fin and tipo_silaba_segun_posicion[id_tipo+1] in [13,12,10,9]):
                    pass
                elif(id_tipo > 1 and tipo_silaba_segun_posicion[id_tipo-1] in [12] and tipo_silaba_segun_posicion[id_tipo-2] in [13,12,11,8,6,4]):
                    pass
                elif(id_tipo > 0 and tipo_silaba_segun_posicion[id_tipo-1] in [11] and syllables[id_tipo-1][1] == syllables[id_tipo][1]):
                    pass
                else:
                    _posiciones_candidatas.extend([0])
            elif(val_tipo == 10):                    # [0c]
                # No se permite cuando: [*00]+[0_c_]                        (Ya se debe estar cumpliendo)
                # No se permite cuando: [**0]+[0_c_]+[0**]                  (Ya se debe estar cumpliendo)
                _posiciones_candidatas.extend([1])
            elif(val_tipo == 9):                    # [00c]
                # No se permite cuando: [00c]+[0*]                          (Ya se debe estar cumpliendo)
                _posiciones_candidatas.extend([2])
            elif(val_tipo == 8):                    # [c00]
                # No se permite cuando: [**0]+[_c_00]
                if(id_tipo > 0 and tipo_silaba_segun_posicion[id_tipo-1] in [13,12,11,8,6,4]):
                    pass
                else:
                    _posiciones_candidatas.extend([0])
            elif(val_tipo == 7):                    # [c0c]
                # 1ª Consonante - No se permite cuando: [*00]+[_c_0c]
                # 1ª Consonante - No se permite cuando: [*0]+[0]+[_c_0c]
                if(id_tipo > 0 and tipo_silaba_segun_posicion[id_tipo-1] in [12,8,4]):
                    pass
                elif(id_tipo > 1 and tipo_silaba_segun_posicion[id_tipo-1] in [13] and tipo_silaba_segun_posicion[id_tipo-2] in [13,12,11,8,6,4]):
                    pass
                else:
                    _posiciones_candidatas.extend([0])
                # 2ª Consonante - No se permite cuando: [c0_c_]+[00*]       (Ya se debe estar cumpliendo)
                _posiciones_candidatas.extend([2])
            elif(val_tipo == 6):                    # [cc0]
                # Siempre se puede eliminar una u otra consonante:
                _posiciones_candidatas.extend([0,1])
            elif(val_tipo == 5):                    # [c00c]
                # 1ª Consonante - No se permite cuando: [**0]+[_c_00c]
                if(id_tipo > 0 and tipo_silaba_segun_posicion[id_tipo-1] in [13,12,11,8,6,4]):
                    pass
                else:
                    _posiciones_candidatas.extend([0])
                # 2ª Consonante - No se permite cuando: [c00_c_]+[0**]
                if(id_tipo < _Fin and tipo_silaba_segun_posicion[id_tipo+1] in [13,12,10,9]):
                    pass
                else:
                    _posiciones_candidatas.extend([3])
            elif(val_tipo == 4):                    # [cc00]
                # Siempre se puede eliminar una u otra consonante:
                _posiciones_candidatas.extend([0,1])
            elif(val_tipo == 3):                    # [cc0c]
                # Siempre se puede eliminar 1ª ó 2ª consonante:
                _posiciones_candidatas.extend([0,1])
                # 3ª Consonante - No se permite cuando: [cc0_c_]+[00*]      (Ya se debe estar cumpliendo)
                _posiciones_candidatas.extend([3])
            elif(val_tipo == 2):                    # [cc00c]
                # Siempre se puede eliminar 1ª ó 2ª consonante:
                _posiciones_candidatas.extend([0,1])
                # 3ª Consonante - No se permite cuando: [cc00_c_]+[0*]      (Ya se debe estar cumpliendo)
                _posiciones_candidatas.extend([3])
            elif(val_tipo == 1):
                # Siempre se puede eliminar 1ª ó 5ª consonante:
                _posiciones_candidatas.extend([0,4])

            elif(val_tipo == 0):
                # Siempre se puede eliminar 1ª, 2ª, 3ª y 4ª consonante:
                _posiciones_candidatas.extend([0,1,3,4])
            else:
                pass

            if(len(_posiciones_candidatas)>0):
                _instrucciones_modificacion[id_tipo] = _posiciones_candidatas
                _silabas_candidatas.append(id_tipo)

        # -- Determinamos lista de palabras modificadas -- #

        listado_candidatas = []

        for _posicion_silaba in _silabas_candidatas:

            for ival_pos in _instrucciones_modificacion[_posicion_silaba]:

                _palabra_candidata = OUTPUT_PALABRA.copy()
                _aux_silaba = list(_palabra_candidata[_posicion_silaba])
                _aux_silaba[ival_pos] = ''
                _palabra_candidata[_posicion_silaba] = ''.join(_aux_silaba)
                _palabra_candidata = "".join(_palabra_candidata)

                listado_candidatas.append(_palabra_candidata)

    # [3.7] AÑADIR CONSONANTES
    # ------------------------

    elif(INPUT_TIPO_CAMBIO == 7):

        _instrucciones_modificacion = {}
        _silabas_candidatas = []

        _Fin = len(tipo_silaba_segun_posicion)-1

        for id_tipo,val_tipo in enumerate(tipo_silaba_segun_posicion):

            _posiciones_candidatas = []
            _consonantes_candidatas = []

            if(val_tipo == 13):                     # [0]
                # Siempre puedo añadir al principio:
                _posiciones_candidatas.extend([0])
                _consonantes_candidatas.extend([G_ListaConsonantesNoQ])
                # Siempre puedo añadir al final pero incluso más letras que G_REGLA_ConsonantesAlFinal si se cumple alguna condición adicional:
                if(id_tipo < _Fin and tipo_silaba_segun_posicion[id_tipo+1] in [11,8,7,5] and OUTPUT_PALABRA[id_tipo+1][0] in ['l']):
                    _posiciones_candidatas.extend([1])
                    _consonantes_candidatas.extend([G_REGLA_ConsonantesAlFinal+['b','c','f','g','l','p']])
                elif(id_tipo < _Fin and tipo_silaba_segun_posicion[id_tipo+1] in [11,8,7,5] and OUTPUT_PALABRA[id_tipo+1][0] in ['r']):
                    if(OUTPUT_PALABRA[id_tipo+1][1] not in ['r']): # Evitas problemas como: A-[rr]a-yan
                        _posiciones_candidatas.extend([1])
                        _consonantes_candidatas.extend([G_REGLA_ConsonantesAlFinal+['b','c','d','f','g','p','r','t']])
                else:
                    _posiciones_candidatas.extend([1])
                    _consonantes_candidatas.extend([G_REGLA_ConsonantesAlFinal])

            elif(val_tipo == 12):                   # [00]
                # Siempre puedo añadir al principio:
                if('q'+OUTPUT_PALABRA[id_tipo] in G_REGLA_SilabasQueQui):
                    _posiciones_candidatas.extend([0])
                    _consonantes_candidatas.extend([G_ListaConsonantes])
                else:
                    _posiciones_candidatas.extend([0])
                    _consonantes_candidatas.extend([G_ListaConsonantesNoQ])
                # Siempre puedo añadir al final pero incluso más letras que G_REGLA_ConsonantesAlFinal si se cumple alguna condición adicional:
                if(id_tipo < _Fin and tipo_silaba_segun_posicion[id_tipo+1] in [11,8,7,5] and OUTPUT_PALABRA[id_tipo+1][0] in ['l']):
                    _posiciones_candidatas.extend([2])
                    _consonantes_candidatas.extend([G_REGLA_ConsonantesAlFinal+['b','c','f','g','l','p']])
                elif(id_tipo < _Fin and tipo_silaba_segun_posicion[id_tipo+1] in [11,8,7,5] and OUTPUT_PALABRA[id_tipo+1][0] in ['r']):
                    if(OUTPUT_PALABRA[id_tipo+1][1] not in ['r']): # Evitas problemas como: A-[rr]a-yan
                        _posiciones_candidatas.extend([2])
                        _consonantes_candidatas.extend([G_REGLA_ConsonantesAlFinal+['b','c','d','f','g','p','r','t']])
                else:
                    _posiciones_candidatas.extend([2])
                    _consonantes_candidatas.extend([G_REGLA_ConsonantesAlFinal])

            elif(val_tipo == 11):                    # [c0]
                # Antes de 1ª consonante:
                if((id_tipo == 0) or (id_tipo > 0 and OUTPUT_PALABRA[id_tipo-1][-1] in G_ListaVocales)):
                    if(OUTPUT_PALABRA[id_tipo][0] in ['l']):
                        _posiciones_candidatas.extend([0])
                        _consonantes_candidatas.extend([['b','c','f','g','l','p']])
                    elif(OUTPUT_PALABRA[id_tipo][0] in ['r']):
                        if(id_tipo > 0):
                            _posiciones_candidatas.extend([0])
                            _consonantes_candidatas.extend([['b','c','d','f','g','p','t','r']]) # A-[rr]o-yar
                        else:
                            _posiciones_candidatas.extend([0])
                            _consonantes_candidatas.extend([['b','c','d','f','g','p','t']])
                # Después de 1ª consonante:
                if(OUTPUT_PALABRA[id_tipo][0] in ['c']):
                    _posiciones_candidatas.extend([1])
                    _consonantes_candidatas.extend([['h','l','r']])       # 'co' -> 'cho','clo','cro'
                elif(OUTPUT_PALABRA[id_tipo][0] in ['b','f','g','p']):
                    _posiciones_candidatas.extend([1])
                    _consonantes_candidatas.extend([['l','r']])           # 'bo' -> 'blo','bro'
                elif(OUTPUT_PALABRA[id_tipo][0] in ['d','t']):
                    _posiciones_candidatas.extend([1])
                    _consonantes_candidatas.extend([['r']])               # 'do' -> 'dro'
                elif(OUTPUT_PALABRA[id_tipo][0] in ['l']):
                    _posiciones_candidatas.extend([1])
                    _consonantes_candidatas.extend([['l']])               # 'lo' -> 'llo'
                elif((id_tipo > 0) and (OUTPUT_PALABRA[id_tipo-1][-1] in G_ListaVocales) and (OUTPUT_PALABRA[id_tipo][0] in ['r'])):
                    _posiciones_candidatas.extend([1])
                    _consonantes_candidatas.extend([['r']])               # [**0]+'ro' -> [**0]+'rro'
                # Siempre puedo añadir al final pero incluso más letras que G_REGLA_ConsonantesAlFinal si se cumple alguna condición adicional:
                if(id_tipo < _Fin and tipo_silaba_segun_posicion[id_tipo+1] in [11,8,7,5] and OUTPUT_PALABRA[id_tipo+1][0] in ['l']):
                    _posiciones_candidatas.extend([2])
                    _consonantes_candidatas.extend([G_REGLA_ConsonantesAlFinal+['b','c','f','g','l','p']])
                elif(id_tipo < _Fin and tipo_silaba_segun_posicion[id_tipo+1] in [11,8,7,5] and OUTPUT_PALABRA[id_tipo+1][0] in ['r']):
                    if(OUTPUT_PALABRA[id_tipo+1][1] not in ['r']): # Evitas problemas como: A-[rr]a-yan
                        _posiciones_candidatas.extend([2])
                        _consonantes_candidatas.extend([G_REGLA_ConsonantesAlFinal+['b','c','d','f','g','p','r','t']])
                else:
                    _posiciones_candidatas.extend([2])
                    _consonantes_candidatas.extend([G_REGLA_ConsonantesAlFinal])
                # NUEVO - Crea la estructura CH
                if('h' in OUTPUT_PALABRA[id_tipo]):
                    _posiciones_candidatas.extend([OUTPUT_PALABRA[id_tipo].find('h')])
                    _consonantes_candidatas.extend([['c']])     #Albahaca -> Albachaca

            elif(val_tipo == 10):                    # [0c]
                # Siempre puedo añadir al principio:
                _posiciones_candidatas.extend([0])
                _consonantes_candidatas.extend([G_ListaConsonantesNoQ])

            elif(val_tipo == 9):                    # [00c]
                # Caso particular: {"Que", "Qui"}
                if('q'+OUTPUT_PALABRA[id_tipo] in G_REGLA_SilabasQueQui):
                    _posiciones_candidatas.extend([0])
                    _consonantes_candidatas.extend([G_ListaConsonantes])
                else:
                    _posiciones_candidatas.extend([0])
                    _consonantes_candidatas.extend([G_ListaConsonantesNoQ])

            elif(val_tipo == 8):                    # [c00]
                # Antes de 1ª consonante:
                if((id_tipo == 0) or (id_tipo > 0 and OUTPUT_PALABRA[id_tipo-1][-1] in G_ListaVocales)):
                    if(OUTPUT_PALABRA[id_tipo][0] in ['l']):
                        _posiciones_candidatas.extend([0])
                        _consonantes_candidatas.extend([['b','c','f','g','l','p']])
                    elif(OUTPUT_PALABRA[id_tipo][0] in ['r']):
                        if(id_tipo > 0):
                            _posiciones_candidatas.extend([0])
                            _consonantes_candidatas.extend([['b','c','d','f','g','p','t','r']]) # A-[rr]o-yar
                        else:
                            _posiciones_candidatas.extend([0])
                            _consonantes_candidatas.extend([['b','c','d','f','g','p','t']])
                # Después de 1ª consonante:
                if(OUTPUT_PALABRA[id_tipo][0] in ['c']):
                    _posiciones_candidatas.extend([1])
                    _consonantes_candidatas.extend([['h','l','r']])       # 'co' -> 'cho','clo','cro'
                elif(OUTPUT_PALABRA[id_tipo][0] in ['b','f','g','p']):
                    _posiciones_candidatas.extend([1])
                    _consonantes_candidatas.extend([['l','r']])           # 'bo' -> 'blo','bro'
                elif(OUTPUT_PALABRA[id_tipo][0] in ['d','t']):
                    _posiciones_candidatas.extend([1])
                    _consonantes_candidatas.extend([['r']])               # 'do' -> 'dro'
                elif(OUTPUT_PALABRA[id_tipo][0] in ['l']):
                    _posiciones_candidatas.extend([1])
                    _consonantes_candidatas.extend([['l']])               # 'lo' -> 'llo'
                elif((id_tipo > 0) and (OUTPUT_PALABRA[id_tipo-1][-1] in G_ListaVocales) and (OUTPUT_PALABRA[id_tipo][0] in ['r'])):
                    _posiciones_candidatas.extend([1])
                    _consonantes_candidatas.extend([['r']])               # [**0]+'ro' -> [**0]+'rro'
                # Siempre puedo añadir al final pero incluso más letras que G_REGLA_ConsonantesAlFinal si se cumple alguna condición adicional:
                if(id_tipo < _Fin and tipo_silaba_segun_posicion[id_tipo+1] in [11,8,7,5] and OUTPUT_PALABRA[id_tipo+1][0] in ['l']):
                    _posiciones_candidatas.extend([3])
                    _consonantes_candidatas.extend([G_REGLA_ConsonantesAlFinal+['b','c','f','g','l','p']])
                elif(id_tipo < _Fin and tipo_silaba_segun_posicion[id_tipo+1] in [11,8,7,5] and OUTPUT_PALABRA[id_tipo+1][0] in ['r']):
                    if(OUTPUT_PALABRA[id_tipo+1][1] not in ['r']): # Evitas problemas como: A-[rr]a-yan
                        _posiciones_candidatas.extend([3])
                        _consonantes_candidatas.extend([G_REGLA_ConsonantesAlFinal+['b','c','d','f','g','p','r','t']])
                else:
                    _posiciones_candidatas.extend([3])
                    _consonantes_candidatas.extend([G_REGLA_ConsonantesAlFinal])

            elif(val_tipo == 7):                    # [c0c]
                # Antes de 1ª consonante:
                if((id_tipo == 0) or (id_tipo > 0 and OUTPUT_PALABRA[id_tipo-1][-1] in G_ListaVocales)):
                    if(OUTPUT_PALABRA[id_tipo][0] in ['l']):
                        _posiciones_candidatas.extend([0])
                        _consonantes_candidatas.extend([['b','c','f','g','l','p']])
                    elif(OUTPUT_PALABRA[id_tipo][0] in ['r']):
                        if(id_tipo > 0):
                            _posiciones_candidatas.extend([0])
                            _consonantes_candidatas.extend([['b','c','d','f','g','p','t','r']]) # A-[rr]o-yar
                        else:
                            _posiciones_candidatas.extend([0])
                            _consonantes_candidatas.extend([['b','c','d','f','g','p','t']])
                # Después de 1ª consonante:
                if(OUTPUT_PALABRA[id_tipo][0] in ['c']):
                    _posiciones_candidatas.extend([1])
                    _consonantes_candidatas.extend([['h','l','r']])       # 'co' -> 'cho','clo','cro'
                elif(OUTPUT_PALABRA[id_tipo][0] in ['b','f','g','p']):
                    _posiciones_candidatas.extend([1])
                    _consonantes_candidatas.extend([['l','r']])           # 'bo' -> 'blo','bro'
                elif(OUTPUT_PALABRA[id_tipo][0] in ['d','t']):
                    _posiciones_candidatas.extend([1])
                    _consonantes_candidatas.extend([['r']])               # 'do' -> 'dro'
                elif(OUTPUT_PALABRA[id_tipo][0] in ['l']):
                    _posiciones_candidatas.extend([1])
                    _consonantes_candidatas.extend([['l']])               # 'lo' -> 'llo'
                elif((id_tipo > 0) and (OUTPUT_PALABRA[id_tipo-1][-1] in G_ListaVocales) and (OUTPUT_PALABRA[id_tipo][0] in ['r'])):
                    _posiciones_candidatas.extend([1])
                    _consonantes_candidatas.extend([['r']])               # [**0]+'ro' -> [**0]+'rro'
                # ------------------- NUEVO -------------------- #
                # Crea la estructura CH
                if('h' in OUTPUT_PALABRA[id_tipo]):
                    pos = OUTPUT_PALABRA[id_tipo].find('h')
                    _posiciones_candidatas.extend([OUTPUT_PALABRA[id_tipo].find('h')])
                    _consonantes_candidatas.extend([['c']])               # 'Alhambra' -> 'Alchambra'
                # ---------------------------------------------- #

            elif(val_tipo == 6):                    # [cc0]
                # Siempre puedo añadir al final pero incluso más letras que G_REGLA_ConsonantesAlFinal si se cumple alguna condición adicional:
                if(id_tipo < _Fin and tipo_silaba_segun_posicion[id_tipo+1] in [11,8,7,5] and OUTPUT_PALABRA[id_tipo+1][0] in ['l']):
                    _posiciones_candidatas.extend([3])
                    _consonantes_candidatas.extend([G_REGLA_ConsonantesAlFinal+['b','c','f','g','l','p']])
                elif(id_tipo < _Fin and tipo_silaba_segun_posicion[id_tipo+1] in [11,8,7,5] and OUTPUT_PALABRA[id_tipo+1][0] in ['r']):
                    if(OUTPUT_PALABRA[id_tipo+1][1] not in ['r']): # Evitas problemas como: A-[rr]a-yan
                        _posiciones_candidatas.extend([3])
                        _consonantes_candidatas.extend([G_REGLA_ConsonantesAlFinal+['b','c','d','f','g','p','r','t']])
                else:
                    _posiciones_candidatas.extend([3])
                    _consonantes_candidatas.extend([G_REGLA_ConsonantesAlFinal])

            elif(val_tipo == 5):                    # [c00c]
                # Antes de 1ª consonante:
                if((id_tipo == 0) or (id_tipo > 0 and OUTPUT_PALABRA[id_tipo-1][-1] in G_ListaVocales)):
                    if(OUTPUT_PALABRA[id_tipo][0] in ['l']):
                        _posiciones_candidatas.extend([0])
                        _consonantes_candidatas.extend([['b','c','f','g','l','p']])
                    elif(OUTPUT_PALABRA[id_tipo][0] in ['r']):
                        if(id_tipo > 0):
                            _posiciones_candidatas.extend([0])
                            _consonantes_candidatas.extend([['b','c','d','f','g','p','t','r']]) # A-[rr]o-yar
                        else:
                            _posiciones_candidatas.extend([0])
                            _consonantes_candidatas.extend([['b','c','d','f','g','p','t']])
                # Después de 1ª consonante:
                if(OUTPUT_PALABRA[id_tipo][0] in ['c']):
                    _posiciones_candidatas.extend([1])
                    _consonantes_candidatas.extend([['h','l','r']])       # 'co' -> 'cho','clo','cro'
                elif(OUTPUT_PALABRA[id_tipo][0] in ['b','f','g','p']):
                    _posiciones_candidatas.extend([1])
                    _consonantes_candidatas.extend([['l','r']])           # 'bo' -> 'blo','bro'
                elif(OUTPUT_PALABRA[id_tipo][0] in ['d','t']):
                    _posiciones_candidatas.extend([1])
                    _consonantes_candidatas.extend([['r']])               # 'do' -> 'dro'
                elif(OUTPUT_PALABRA[id_tipo][0] in ['l']):
                    _posiciones_candidatas.extend([1])
                    _consonantes_candidatas.extend([['l']])               # 'lo' -> 'llo'
                elif((id_tipo > 0) and (OUTPUT_PALABRA[id_tipo-1][-1] in G_ListaVocales) and (OUTPUT_PALABRA[id_tipo][0] in ['r'])):
                    _posiciones_candidatas.extend([1])
                    _consonantes_candidatas.extend([['r']])               # [**0]+'ro' -> [**0]+'rro'
                # Después de la vocal:
                _posiciones_candidatas.extend([2])
                _consonantes_candidatas.extend([G_ListaConsonantesNoQ])

            elif(val_tipo == 4):                    # [cc00]
                # Después de la 1ª vocal:
                _posiciones_candidatas.extend([3])
                _consonantes_candidatas.extend([G_ListaConsonantesNoQ])
                # Siempre puedo añadir al final pero incluso más letras que G_REGLA_ConsonantesAlFinal si se cumple alguna condición adicional:
                if(id_tipo < _Fin and tipo_silaba_segun_posicion[id_tipo+1] in [11,8,7,5] and OUTPUT_PALABRA[id_tipo+1][0] in ['l']):
                    _posiciones_candidatas.extend([4])
                    _consonantes_candidatas.extend([G_REGLA_ConsonantesAlFinal+['b','c','f','g','l','p']])
                elif(id_tipo < _Fin and tipo_silaba_segun_posicion[id_tipo+1] in [11,8,7,5] and OUTPUT_PALABRA[id_tipo+1][0] in ['r']):
                    if(OUTPUT_PALABRA[id_tipo+1][1] not in ['r']): # Evitas problemas como: A-[rr]a-yan
                        _posiciones_candidatas.extend([4])
                        _consonantes_candidatas.extend([G_REGLA_ConsonantesAlFinal+['b','c','d','f','g','p','r','t']])
                else:
                    _posiciones_candidatas.extend([4])
                    _consonantes_candidatas.extend([G_REGLA_ConsonantesAlFinal])

            elif(val_tipo == 3):                    # [cc0c]
                if ('h' in OUTPUT_PALABRA[id_tipo]):
                    pos = OUTPUT_PALABRA[id_tipo].find('h')
                    _posiciones_candidatas.extend([OUTPUT_PALABRA[id_tipo].find('h')])
                    _consonantes_candidatas.extend([['c']])  # 'Prohibir' -> 'Prochibir'

            elif(val_tipo == 2):                    # [cc00c]
                # Después de la 1ª vocal:
                _posiciones_candidatas.extend([3])
                _consonantes_candidatas.extend([G_ListaConsonantesNoQ])

            elif(val_tipo == 1): # [c000c]
                # Condición: 1, Inserta en 2, Estructura: [c0]+x+[END]
                _posiciones_candidatas.extend([2])
                _consonantes_candidatas.extend([G_ListaConsonantesNoQ])
                # Condición: 2, Inserta en 3, Estructura: [c00]+x+[END]
                _posiciones_candidatas.extend([3])
                _consonantes_candidatas.extend([G_ListaConsonantesNoQ])

            else:
                pass

            if(len(_posiciones_candidatas)>0):
                _instrucciones_modificacion[id_tipo] = {'pos': _posiciones_candidatas,
                                                        'val': _consonantes_candidatas}
                _silabas_candidatas.append(id_tipo)

        # -- Determinamos lista de palabras modificadas -- #

        listado_candidatas = []

        for _posicion_silaba in _silabas_candidatas:

            pos_list = _instrucciones_modificacion[_posicion_silaba]['pos']
            val_list = _instrucciones_modificacion[_posicion_silaba]['val']

            for pos, val in zip(pos_list, val_list):

                for letra in val:

                    _palabra_candidata = OUTPUT_PALABRA.copy()
                    _silaba = _palabra_candidata[_posicion_silaba]
                    _aux_silaba = list(_palabra_candidata[_posicion_silaba])

                    if(pos==0):
                        _aux_silaba = [letra]+_aux_silaba
                    elif(pos==len(_aux_silaba)):
                        _aux_silaba = _aux_silaba+[letra]
                    else:
                        _aux_silaba = _aux_silaba[:pos]+[letra]+_aux_silaba[pos:]
                    # ----------- NUEVO --------------- #
                    if len(_silaba)-1 != pos:
                        if letra == 'q' and _silaba[pos + 1] != 'u':
                            _aux_silaba[pos] = 'qu'
                    else:
                        _aux_silaba[pos] = letra
                    _new_silaba = ''.join(_aux_silaba)
                    # --------------------------------- #
                    _palabra_candidata[_posicion_silaba] = ''.join(_aux_silaba)
                    _palabra_candidata = "".join(_palabra_candidata)

                    listado_candidatas.append(_palabra_candidata)

    # --------------------------------------------------------------------------
    # --------------------------------------------------------------------------
    # --------------------------------------------------------------------------
    # --------------------------------------------------------------------------
    
    # [4] Devolvemos una palabra al azar:

    listado_candidatas = list(set(listado_candidatas))
    # Añado acento a la palabra
    listado_candidatas_acento = add_accent(G_Palabras_sin_Tilde, G_Palabras_Corco, G_Acento_Palabra,listado_candidatas,INPUT_PALABRA,G_DIC_TILDE,tilde_original)
    if G_LOG:
        print(f"\t[*] Posibles alternativas: {listado_candidatas_acento}")

    if (len(listado_candidatas) > 0):
        if G_LOG != 0:
            if(G_PSEUDO == 0):
                if palabrasCandidatas(listado_candidatas,G_Palabras_Corco):
                    listado_candidatas_final = palabrasCandidatas(listado_candidatas_acento,G_Palabras_Corco)
                    print()
                    print('Lista de palabras genereadas que aparecen en el corco: ', listado_candidatas_final)
                    print()
                else:
                    listado_candidatas_final = listado_candidatas_acento
                    print()
                    print('No se ha generado ninguna palabra que aparezca en el corco, en su lugar, se le va a mostrar una pseudopalabra')
                    print()
            else:

                listado_candidatas_final = listado_candidatas

            if INPUT_TIPO_CAMBIO not in [3,4,6,7]:
                listado_candidatas_final = ordeno_foneticamente(listado_candidatas_final, INPUT_PALABRA,C_DistanciasEntreConsonantes,G_DistanciaFonemasVocales,G_ListaFonemasConsonantes,G_ListaVocales)
                listado_candidatas_acento = add_accent(G_Palabras_sin_Tilde, G_Palabras_Corco, G_Acento_Palabra,listado_candidatas_final,INPUT_PALABRA,G_DIC_TILDE,tilde_original)
                print(f"\t[*] Lista ordenada fonéticamente : {listado_candidatas_acento}")
                OUTPUT_PALABRA = listado_candidatas_acento
                print(f"\t[*] Palabra de salida : {OUTPUT_PALABRA}")

            else:
                OUTPUT_PALABRA = listado_candidatas_acento
                print()
                print(f"\t[*] Palabra aleatoria de salida : {OUTPUT_PALABRA}")
                print()
        else:
            if (G_PSEUDO == 0):
                if palabrasCandidatas(listado_candidatas, G_Palabras_Corco):
                    listado_candidatas_final = palabrasCandidatas(listado_candidatas_acento, G_Palabras_Corco)
                else:
                    listado_candidatas_final = listado_candidatas_acento
            else:

                listado_candidatas_final = listado_candidatas_acento

            if INPUT_TIPO_CAMBIO not in [3, 4, 6, 7]:
                
                #listado_candidatas_final = ordeno_foneticamente(listado_candidatas_final, INPUT_PALABRA,C_DistanciasEntreConsonantes,G_DistanciaFonemasVocales,G_ListaFonemasConsonantes,G_ListaVocales)
                #listado_candidatas_acento = add_accent(G_Palabras_sin_Tilde, G_Palabras_Corco, G_Acento_Palabra,listado_candidatas_final,INPUT_PALABRA,G_DIC_TILDE,tilde_original)
                OUTPUT_PALABRA = listado_candidatas_acento

            else:
                OUTPUT_PALABRA = listado_candidatas_acento
    else:
        if INPUT_TIPO_CAMBIO != 0:
            OUTPUT_PALABRA = []
        else:
            OUTPUT_PALABRA = [INPUT_PALABRA]
        

    return OUTPUT_PALABRA