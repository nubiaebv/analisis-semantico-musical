"""
Clase: path

Objetivo: Py con funciones para utilidades a nivel de codigo en rutas de archivos

Cambios:
    1. Creacion de la funcion obtener_ruta_app con el fin de mapear la ruta raiz del proyecto
    pmarin 14-02-2026
"""
import glob
import os
import logging
import pandas as pd
# Configurar logger para esta clase
logger = logging.getLogger(__name__)

def obtener_ruta_local(nombre_objetivo="analisis-semantico-musical"):
    """
    Busca la ruta hasta la carpeta con nombre `nombre_objetivo` en entorno local
    """
    try:
        ruta_actual = os.path.dirname(__file__)
    except NameError:
        # Para entornos donde __file__ no existe, como Jupyter
        ruta_actual = os.getcwd()
    except Exception as e:
        logger.error(f"No se pudo determinar la ruta base: {e}")
        return None

    try:
        iteraciones = 0
        max_iteraciones = 10  # Prevenir bucles infinitos

        while iteraciones < max_iteraciones:
            if os.path.basename(ruta_actual) == nombre_objetivo:
                logger.info(f"Carpeta '{nombre_objetivo}' encontrada en: {ruta_actual}")
                return ruta_actual

            ruta_padre = os.path.dirname(ruta_actual)

            if ruta_padre == ruta_actual:  # Llegamos a la raíz del sistema
                logger.warning(f"No se encontró la carpeta '{nombre_objetivo}' - llegamos a la raíz")
                break

            ruta_actual = ruta_padre
            iteraciones += 1

        logger.warning(f"No se encontró la carpeta '{nombre_objetivo}' después de {max_iteraciones} iteraciones")
        return None

    except Exception as e:
        logger.error(f"Error al buscar la carpeta '{nombre_objetivo}': {e}")
        return None


def obtener_ruta_app(nombre_objetivo="analisis-semantico-musical"):
    """
    Versión simplificada y robusta para Azure App Service y entornos locales

    Args:
        nombre_objetivo (str): Nombre de la carpeta objetivo a buscar (solo para entornos locales)

    Returns:
        str: Ruta completa válida (nunca None)
    """
    try:
        directorio_actual = os.getcwd()
        logger.info(f"Directorio actual: {directorio_actual}")

        # Verificar que el directorio actual contenga los archivos principales
        archivos_principales = ['main.py', 'src', 'app','dashboard']
        archivos_encontrados = []

        for archivo in archivos_principales:
            ruta_archivo = os.path.join(directorio_actual, archivo)
            if os.path.exists(ruta_archivo):
                archivos_encontrados.append(archivo)

        logger.info(f"Archivos principales encontrados: {archivos_encontrados}")

        # Si encontramos al menos 2 archivos principales, usar directorio actual
        if len(archivos_encontrados) >= 2:
            logger.info(
                f"Usando directorio actual (tiene {len(archivos_encontrados)} archivos principales): {directorio_actual}")
            return directorio_actual


        # Para entornos locales, intentar buscar la carpeta objetivo
        ruta_local = obtener_ruta_local(nombre_objetivo)
        if ruta_local:
            logger.info(f"Encontrada carpeta objetivo en entorno local: {ruta_local}")
            return ruta_local

        # Fallback: usar directorio actual siempre
        logger.info(f"Usando directorio actual como fallback: {directorio_actual}")
        return directorio_actual

    except Exception as e:
        logger.error(f"Error en obtener_ruta_app: {e}")
        # Último recurso: directorio de trabajo actual
        try:
            fallback = os.getcwd()
            logger.info(f"Usando último recurso: {fallback}")
            return fallback
        except:
            logger.error("No se pudo obtener ni siquiera el directorio actual")
            return "."


def validar_ruta_app(ruta):
    """
    Valida que la ruta sea válida y accesible

    Args:
        ruta (str): Ruta a validar

    Returns:
        bool: True si la ruta es válida, False en caso contrario
    """
    try:
        if not ruta:
            return False

        if not isinstance(ruta, (str, bytes, os.PathLike)):
            return False

        if not os.path.exists(ruta):
            logger.warning(f"La ruta no existe: {ruta}")
            return False

        if not os.path.isdir(ruta):
            logger.warning(f"La ruta no es un directorio: {ruta}")
            return False

        # Verificar que sea accesible
        try:
            os.listdir(ruta)
            return True
        except PermissionError:
            logger.warning(f"Sin permisos para acceder a la ruta: {ruta}")
            return False

    except Exception as e:
        logger.error(f"Error al validar ruta: {e}")
        return False


def abrir_archivo(archivo = "corpus_canciones.csv"):
    try:
        directorio_proyecto = obtener_ruta_local()
        df = pd.read_csv(directorio_proyecto + '\\data\\processed\\'+archivo, delimiter=',', decimal=".",
                         encoding='utf-8')

        if df.empty:
            print("Advertencia: El archivo está vacío.")

        return df

    except FileNotFoundError as e:
        print(f"Error de ubicación: {e}")
    except pd.errors.EmptyDataError:
        print("Error: El archivo CSV está vacío.")
    except pd.errors.ParserError:
        print("Error: El archivo tiene un formato incorrecto (posible error de delimitadores).")
    except Exception as e:
        print(f"Ocurrió un error inesperado: {e}")

    return None