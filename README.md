# NLP aplicado al screening de Páncreas Patológico

Este repositorio contiene scripts y herramientas para el análisis y procesamiento de informes de imagenología.
A continuación se detallan los pasos para configurar el entorno y ejecutar los scripts disponibles.

## Configuración del Entorno

1. Crear un entorno virtual:
   python -m venv mi_entorno

2. En Windows:
   mi_entorno\Scripts\activate
   En macOS y Linux:
   source mi_entorno/bin/activate

3. Instalar los requerimientos necesarios:
   pip install -r requirements.txt

## Ejecución de Scripts

## Instrucciones para la interfaz Gradio
gradiointerfaz.py: Proporciona una interfaz gráfica para interactuar con el sistema. Esta interfaz utiliza funciones del archivo funciones.py para evaluar calidad, estadísticas, realizar búsquedas y realizar el screening de páncreas patológico.
Para utilizar la interfaz, insertar el archivo DatasetPrueba.xls.

## Importante sobre DatasetPrueba.xls
El archivo DatasetPrueba incluido en este repositorio es solo para fines de prueba. No contiene información confidencial ni sensible y está diseñado únicamente como un ejemplo para mostrar cómo funcionan los scripts y la interfaz. Los resultados generados a partir de este dataset son ficticios y no deben considerarse como datos reales o utilizables en un contexto clínico o de investigación.

## Otros archivos
Clasificacion.xlsx : Este archivo se realizó para corregir palabras y construir una base de datos a partir de palabras extraídas de 30,000 informes.
glosario.xlsx : En este archivo se extrajeron siglas de 30,000 informes para construir un glosario con sus respectivos significados y contextos.
Palabrasycategorias.xlsx : Contiene palabras correctas y sus respectivas categorías, lo cual es útil para el análisis y clasificación de términos.
Diccionarioespañol1.txt : Diccionario en idioma español.
