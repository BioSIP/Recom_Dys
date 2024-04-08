from generadorpalabras import *

INPUT_PALABRA = "".join(['paso'])

INPUT_TIPO_CAMBIO = 1   # ---------- #
                        # [0] = No se modifica la palabra
                        # ----- Vocales ----- #
                        # [1] = Se altera una vocal al azar al principio y al final
                        # [2] = Se alteran ambas vocales al principio y al final
                        # [3] = Se quitan vocales si es posible
                        # [4] = Se añaden vocales
                        # ----- Consonantes ----- #
                        # [5] = Se altera una consonante al azar
                        # [6] = Se elimina una consonante
                        # [7] = Se añade una consonante si es posible
                        # ---------- #

G_PSEUDO = 0     # 0 = Se intentan proponer palabras que sí que aparecen en el Corco
                 # 1 = No se hacen búsquedas en el Corco (alta probabilidad de que se genere una pseudo-palabra)

G_LOG = 0        # 0 = No se muestran tantos mensajes por pantalla...
                 # 1 = Se muestran muchos mensajes por pantalla...

G_DistanciaMaximaVocal = 2  # Distancia máxima entre vocales.
G_DistanciaMaximaConsonante = 1  # Distancia máxima entre consonantes.

output = genera_palabras(INPUT_PALABRA, INPUT_TIPO_CAMBIO, G_DistanciaMaximaVocal, G_DistanciaMaximaConsonante, G_PSEUDO, G_LOG)
