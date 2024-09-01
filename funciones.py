# -*- coding: utf-8 -*-
"""
Created on Sun Sep  1 14:35:41 2024

@author: emili
"""

import pandas as pd
import matplotlib.pyplot as plt
import unicodedata
from unidecode import unidecode
import re
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize

# Carga de datos
palabrasycategorias = pd.read_excel('Palabrasycategorias.xlsx')
glosario = pd.read_excel('glosario.xlsx')

# Sets para palabras del glosario y diccionario
glosario_words = {unidecode(word.lower()) for word in glosario['Termino'] if isinstance(word, str)}
dicmed_words = {unidecode(word.lower()) for word in palabrasycategorias['Palabra'] if isinstance(word, str)}

with open('Diccionarioespañol1.txt', 'r', encoding='utf-8') as file:
    content_words = set(re.findall(r'\b\w+\b', file.read().lower()))

# Columnas requeridas para el archivo a cargar
columnas_requeridas = ['Fecha_Ing', 'Tipo_OS', 'Nro_OS', 'Nro_Acto', 'Acto', 'Edad', 'Sexo', 'Nro_Medico', 'Dato_Clinico', 'Informe']

def verificar_y_cargar_archivo(file):
    """ Verifica y carga un archivo Excel, realizando validaciones sobre sus columnas. """
    global data
    try:
        todas_filas = pd.read_excel(file, header=None)
        comienzo = todas_filas[todas_filas.apply(lambda row: row.astype(str).str.contains('Fecha_Ing').any(), axis=1)].index[0]
        data = pd.read_excel(file, skiprows=comienzo)
    except Exception as e:
        return None, f"Error al leer el archivo: {e}", "error"

    columnas_faltantes = [col for col in columnas_requeridas if col not in data.columns]
    
    if columnas_faltantes:
        return None, f"El archivo no tiene los siguientes campos: {', '.join(columnas_faltantes)}", "error"

    # Convertir tipos de datos y realizar filtrado
    data['Fecha_Ing'] = pd.to_datetime(data['Fecha_Ing'])
    data['Edad'] = pd.to_numeric(data['Edad'], errors='coerce')
    
    return data, "El archivo se cargó correctamente.", "success"

def normalizar(texto):
    #Normaliza texto eliminando acentos y caracteres especiales
    if isinstance(texto, str):
        texto = texto.lower().strip()
        texto = unicodedata.normalize('NFKD', texto)
        texto = ''.join([c for c in texto if not unicodedata.combining(c)])
        texto = re.sub(r'\s+', ' ', texto)
    return texto

# FUNCIONES ESTADÍSTICAS

def periodo_tiempo():
    #Calcula el período de tiempo entre la fecha mínima y máxima
    fecha_minima = pd.to_datetime(data['Fecha_Ing']).min()
    fecha_maxima = pd.to_datetime(data['Fecha_Ing']).max()
    periodo_tiempo = fecha_maxima - fecha_minima
    
    def format_periodo_tiempo(periodo):
        dias = periodo.days
        años = dias // 365
        meses = (dias % 365) // 30
        dias_restantes = (dias % 365) % 30
        return f"{años} años, {meses} meses, {dias_restantes} días"

    return format_periodo_tiempo(periodo_tiempo)

def distribucion_genero():
    #Devuelve una tabla con la distribución de género 
    conteo_sexo = data['Sexo'].value_counts()
    total_registros = len(data)
    
    return pd.DataFrame({
        "Género": ["Femenino", "Masculino"],
        "Cantidad": [conteo_sexo.get('F', 0), conteo_sexo.get('M', 0)],
        "Porcentaje": [round((conteo_sexo.get('F', 0) / total_registros) * 100, 1), 
                       round((conteo_sexo.get('M', 0) / total_registros) * 100, 1)]
    })

def histograma_edad_genero():
    #Genera y guarda un histograma de distribución de edad por género
    plt.figure()
    data[data['Sexo'] == 'F']['Edad'].hist(bins=20, alpha=0.7, label='Mujeres', color='#ffa3a5')
    data[data['Sexo'] == 'M']['Edad'].hist(bins=20, alpha=0.7, label='Hombres', color='#61a5c2')
    plt.title('Distribución de Edad por Género')
    plt.xlabel('Edad')
    plt.ylabel('Frecuencia')
    plt.legend()
    plt.grid(False)
    plt.savefig('histograma_edad_genero.png')
    return 'histograma_edad_genero.png'

def genero_por_tipo_de_acto():
    #Devuelve una tabla con la distribución de género por tipo de acto
    genero_tipoacto = data.groupby(['Acto', 'Sexo']).size().unstack(fill_value=0)
    genero_tipoacto['Total'] = genero_tipoacto.sum(axis=1)
    genero_tipoacto['Porcentaje_F'] = round((genero_tipoacto['F'] / genero_tipoacto['Total']) * 100, 1)
    genero_tipoacto['Porcentaje_M'] = round((genero_tipoacto['M'] / genero_tipoacto['Total']) * 100, 1)
    return genero_tipoacto.reset_index().sort_values(by='Total', ascending=False)

def grafico_genero_por_tipo_de_acto():
    #Genera y guarda gráfico de barra para la distribución de género por los 10 tipos de estudios más frecuentes
    genero_tipoacto = data.groupby(['Acto', 'Sexo']).size().unstack(fill_value=0)
    genero_tipoacto['Total'] = genero_tipoacto.sum(axis=1)
    top10_estudios = genero_tipoacto.sort_values(by='Total', ascending=False).head(10).index
    genero_top10 = genero_tipoacto.loc[top10_estudios]
    genero_top10['Porcentaje_F'] = round((genero_top10['F'] / genero_top10['Total']) * 100, 1)
    genero_top10['Porcentaje_M'] = round((genero_top10['M'] / genero_top10['Total']) * 100, 1)
    ax = genero_top10[['Porcentaje_F', 'Porcentaje_M']].plot(kind='bar', stacked=True, figsize=(12, 8), color=['#ffa3a5', '#61a5c2'])

    plt.title('Distribución de Género por los 10 Tipos de Estudios más Frecuentes')
    plt.xlabel('Tipo de Estudio')
    plt.ylabel('Porcentaje')
    plt.legend(['Femenino', 'Masculino'])
    plt.xticks(rotation=45, ha='right')
    
    for container in ax.containers:
        ax.bar_label(container, label_type='center')

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.2)
    plt.savefig('grafico_genero_top10_estudios.png')
    plt.close()
    return 'grafico_genero_top10_estudios.png'

def plot_genero_distribucion():
    #Genera y guarda gráfico de distribución de género por tipo de estudio para los 10 tipos más frecuentes
    genero_tipoacto = data.groupby(['Acto', 'Sexo']).size().unstack(fill_value=0)
    genero_tipoacto['Total'] = genero_tipoacto.sum(axis=1)
    top10_estudios = genero_tipoacto.nlargest(10, 'Total').index
    genero_top10 = genero_tipoacto.loc[top10_estudios]
    plt.figure(figsize=(12, 8))
    genero_top10 = genero_top10.rename(columns={'F': 'Femenino', 'M': 'Masculino'})
    genero_top10[['Femenino', 'Masculino']].plot(kind='bar', stacked=True, ax=plt.gca(), color=['#ffa3a5','#61a5c2'])

    plt.xlabel('Tipo de Estudio')
    plt.ylabel('Cantidad')
    plt.title('Distribución por Género de los 10 Tipos de Estudios más Frecuentes')

    for i, (index, row) in enumerate(genero_top10.iterrows()):
        plt.text(i, row['Total'] + 5, str(row['Total']), ha='center')

    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('grafico_genero2_top10_estudios.png')
    return 'grafico_genero2_top10_estudios.png'

def estadisticas_edad_tipo_de_acto():
    #Calcula estadísticas de edad por tipo de acto
    return data.groupby('Acto')['Edad'].agg(
        Promedio_Edad=lambda x: round(x.mean(), 1),
        Cantidad='count',
        Desviacion_Estandar=lambda x: round(x.std(), 1)
    ).sort_values(by='Cantidad', ascending=False).reset_index()

def grafico_top5_estudios():
    genero_tipoacto = data.groupby(['Acto', 'Sexo']).size().unstack(fill_value=0)
    genero_tipoacto['Total'] = genero_tipoacto.sum(axis=1)
    top5_estudios = genero_tipoacto.sort_values(by='Total', ascending=False).head(5).index
    
    plt.figure(figsize=(15, 10))
    for acto in top5_estudios:
        plt.hist(data[data['Acto'] == acto]['Edad'], bins=20, alpha=0.5, label=acto)
    plt.title('Distribución de Edad por los 5 Tipos de Estudios más Frecuentes')
    plt.xlabel('Edad')
    plt.ylabel('Frecuencia')
    plt.legend()
    plt.savefig('grafico_top5_estudios.png')
    return 'grafico_top5_estudios.png'


def grafico_edad_acto():
    #Genera y guarda gráfico de edad por tipo de acto
    resumen = estadisticas_edad_tipo_de_acto()
    resumen_top10 = resumen.head(10)
    
    plt.figure(figsize=(12, 8))
    plt.barh(resumen_top10['Acto'], resumen_top10['Promedio_Edad'], color='skyblue', xerr=resumen_top10['Desviacion_Estandar'])
    plt.xlabel('Promedio Edad')
    plt.ylabel('Tipo de Acto')
    plt.title('Promedio de Edad por Tipo de Acto')
    plt.tight_layout()
    plt.savefig('grafico_edad_tipo_de_acto.png')
    plt.close()
    return 'grafico_edad_tipo_de_acto.png'


#BÚSQUEDAS

def buscar_por_texto(texto, columna, es_frase=False):
    #Buscar palabras o frases en una columna especificada
    if data is None:
        return "No hay datos cargados."
    
    columnas_req = ['Fecha_Ing', 'Tipo_OS', 'Nro_OS', 'Nro_Acto', 'Dato_Clinico']
    # Normalizar el contenido de la columna
    data[f'{columna}_Normalizado'] = data[columna].apply(normalizar)
    # Normalizar el texto de búsqueda
    texto_normalizado = normalizar(texto)
    
    # Filtrado según sea búsqueda de palabras o frase
    if es_frase:
        filtrado = data[data[f'{columna}_Normalizado'].str.contains(texto_normalizado, case=False, na=False)]
    else:
        palabras = texto_normalizado.split()
        filtrado = data
        for palabra in palabras:
            filtrado = filtrado[filtrado[f'{columna}_Normalizado'].str.contains(palabra, case=False, na=False)]
    
    data_unique = filtrado.drop_duplicates(subset=['Nro_OS'])
    return data_unique[columnas_req]

def buscar_por_palabras(palabras):
    #Busca palabras en la columna 'Informe'
    return buscar_por_texto(palabras, 'Informe', es_frase=False)

def buscar_por_palabrasdc(palabras):
    #Busca palabras en la columna 'Dato_Clinico'
    return buscar_por_texto(palabras, 'Dato_Clinico', es_frase=False)

def buscar_por_frase(frase):
    #Busca una frase en la columna 'Informe'
    return buscar_por_texto(frase, 'Informe', es_frase=True)

def buscar_por_frasedc(frase):
    #Busca una frase en la columna 'Dato_Clinico'
    return buscar_por_texto(frase, 'Dato_Clinico', es_frase=True)

# POSIBLE PANCREAS PATOLOGICO

def palabras_cercanas(texto, palabras, max_distancia=10):
   
    palabras = [palabra.lower() for palabra in palabras]
    frases = sent_tokenize(texto)
    for frase in frases:
        tokens = word_tokenize(frase)
        indices = [i for i, token in enumerate(tokens) if token in palabras]
        if len(indices) >= len(palabras):
            for i in range(len(indices) - 1):
                if indices[i + 1] - indices[i] <= max_distancia:
                    return True
    return False

def posibles_anomalias_pancreas():
    if data is None:
        return "No hay datos cargados."

    palabras = palabrasycategorias['Palabra']
    categorias = palabrasycategorias['Categoria']
    palabras_patologias = palabras[categorias == 'Patologia']

    palabras_relacionadas_pancreas = [
        "adenocarcinoma pancreatico",
        "pancreatitis",
        "tumor neuroendocrino pancreatico",
        "cancer de pancreas",
        "pancreatoblastoma",
        "neoplasia quistica mucinosa",
        "neoplasia quistica serosa",
        "insulinoma",
        "glucagonoma",
        "gastrinoma",
        "somatostatinoma",
        "carcinoma de celulas acinares",
        "metastasis pancreaticas",
        "cistadenoma seroso",
        "cistadenocarcinoma mucinoso",
        "pancreatoduodenectomia (procedimiento de whipple)",
        "icpn (intracystic papillary mucinous neoplasm)",
        "ductal adenocarcinoma",
        "pseudocisto pancreatico",
        "sindrome de von hippel-lindau",
        "sindrome de zollinger-ellison",
        "cirugia de whipple",
        "cpre (colangiopancreatografia retrograda endoscopica)",
        "eus (endoscopic ultrasound)",
        "ca 19-9 (marcador tumoral)",
        "mutacion kras",
        "bilirrubina elevada",
        "obstruccion biliar",
        "ictericia",
        "dolor abdominal",
        "perdida de peso",
        "diabetes",
        "conducto de wirsung",
        "retropancreatico",
        "periduodenopancreatica",
        "pancreatectomia",
        "peripancreatico",
        "intrapancreatica",
        "peripancreaticas",
        "paripancreaticos",
        "cefalopancreatico","pancreatica",
        "lesion tumoral pancreatica", "cabeza de pancreas", "pancreas atrófico", "cuerpo y cola de pancreas", "lesion pancreatica"
    ]
    
    palabras_relacionadas_pancreas_escaped = [re.escape(palabra) for palabra in palabras_relacionadas_pancreas]
    
    frases_exclusion = [
        "pancreas de forma y tamaño normal",
        "pancreas de forma y tamaño habitual",
        "pancreas de forma, tamaño y densidad habitual",
        "pancreas de morfologia, tamaño y densidad habitual",
        "pancreas y glandulas suprarrenales sin alteraciones",
        "pancreas, bazo y glandulas suprarrenales sin alteraciones",
        "pancreas sin lesiones",
        "Pancreas de morfologia y tamaño habitual",
        "bazo, pancreas y suprarrenales sin alteraciones",
        "pancreas y glandulas suprarrenales sin alteraciones"
    ]

    data['Informe_Normalizado'] = data['Informe'].apply(normalizar)
    columnas_req = ['Fecha_Ing', 'Tipo_OS', 'Nro_OS', 'Nro_Acto', 'Dato_Clinico']
    
    # Filtrado por patologías y palabras relacionadas con el páncreas
    informes_patologias_pancreas = data[
        data['Informe_Normalizado'].str.contains('|'.join(palabras_patologias), case=False, na=False) & 
        data['Informe_Normalizado'].str.contains('|'.join(palabras_relacionadas_pancreas_escaped), case=False, na=False)
    ]
    
    # Excluir frases 
    informes_patologias_pancreas = informes_patologias_pancreas[
        ~informes_patologias_pancreas['Informe_Normalizado'].str.contains('|'.join(frases_exclusion), case=False, na=False)
    ]

    # nltk para comprobar cercanía de palabras clave
    # informes_patologias_pancreas['Cercania'] = informes_patologias_pancreas['Informe_Normalizado'].apply(
    #     lambda x: palabras_cercanas(x, palabras_relacionadas_pancreas)
    # )
    # informes_patologias_pancreas = informes_patologias_pancreas[informes_patologias_pancreas['Cercania']]
    
    data_unique2 = informes_patologias_pancreas.drop_duplicates(subset=['Nro_OS'])

    return data_unique2[columnas_req]


#CALIDAD

# buscar palabras faltantes en un texto
def busco_faltantes(text):
    if isinstance(text, str):
        text_normalized = unidecode(text.lower())

        # Reemplaza los caracteres de puntuación y números por espacios, excepto '/'
        text_normalized = re.sub(r'[,:.\d;]+', ' ', text_normalized)

        # Divide el texto en partes basadas en espacios y limpia cada parte
        words = []
        for segment in text_normalized.split():
            parts = segment.split('-')
            for part in parts:
                cleaned_part = re.sub(r'^[^\w]+|[^\w]+$', '', part)

                # filtro especifico y palabras que contienen números
                if cleaned_part and cleaned_part != 'i/v' and not re.search(r'\d', cleaned_part):
                    if '/' not in cleaned_part:
                        words.append(cleaned_part)
        words = {unidecode(word) for word in words}

        # encuentro faltantes
        faltantes = [word for word in words if word not in content_words and word not in dicmed_words and word not in glosario_words]
        return faltantes

    return []

# Calculo y graficar la cantidad de columnas completas 
def cantidad_completos():
    columnas = ['Fecha_Ing', 'Tipo_OS', 'Nro_OS', 'Nro_Acto', 'Acto', 'Edad', 'Sexo', 'Dato_Clinico', 'Informe', 'Nro_Medico']
    completos = [(data[col].notna().sum()) for col in columnas]

    plt.figure()
    plt.bar(columnas, completos, color='blue', alpha=0.5, label='Completos')
    plt.xlabel('Columnas')
    plt.ylabel('Cantidad')
    plt.legend()
    for i, valor in enumerate(completos):
        plt.text(i, valor + 0.5, str(valor), ha='center', va='bottom')

    plt.xticks(rotation=45)
    plt.ylim(0, max(completos) * 1.1)
    plt.tight_layout()
    plt.savefig('Cantidad de columnas completas.png')
    plt.close()
    return 'Cantidad de columnas completas.png'

# Detectar errores en los informes
def deteccion_errores():
    data_unique = data.drop_duplicates(subset=['Nro_OS'])
    data_informe = data_unique.apply(normalizar, axis=1)
    data_informe['Palabras mal escritas'] = data_informe['Informe'].apply(busco_faltantes)
    data_informe['Cant. palabras mal escritas'] = data_informe['Palabras mal escritas'].apply(len)
    data_informe_ordenado = data_informe[['Fecha_Ing', 'Tipo_OS', 'Nro_OS', 'Nro_Medico', 'Cant. palabras mal escritas', 'Palabras mal escritas']]
    data_informe_ordenado = data_informe_ordenado.sort_values(by='Cant. palabras mal escritas', ascending=False)
    data_informe_ordenado['Informe resaltado'] = data_informe.apply(lambda row: resaltar_errores(row['Informe'], row['Palabras mal escritas']), axis=1)
    return data_informe_ordenado

# Mostrar el informe resaltado
def mostrar_informe_resaltado(nro_os):
    df = deteccion_errores()
    try:
        nro_os = int(nro_os)  
        if nro_os in df['Nro_OS'].values:
            informe = df[df['Nro_OS'] == nro_os]['Informe resaltado'].values[0]
            informe = informe.replace('\n', '<br>')
            leyendas = """
            <div style="background-color: #f2f2f2; border: 1px solid #ccc; padding: 10px; margin-bottom: 10px; max-width: 400px; border-radius: 10px;">
    <h4>Errores:</h4>
    <div style="display: flex; align-items: center;">
        <div style="background-color: lightcoral; width: 20px; height: 20px; margin-right: 5px; border-radius: 5px;"></div>
        <span style="margin-right: 10px;">Palabras mal escritas</span>
        <div style="background-color: #fdfd96; width: 20px; height: 20px; margin-right: 5px; border-radius: 5px;"></div>
        <span>Errores codificación</span>
    </div>
</div>
            """

            return informe, leyendas
        else:
            return "Número de OS no encontrado."
    except ValueError:
        return "El número de OS debe ser un número entero."

# Clasificar errores
def clasificar_errores(faltantes):
    caracteres_especiales = re.compile(r'[^\w\s]')
    codificacion = [word for word in faltantes if caracteres_especiales.search(word)]
    resto = [word for word in faltantes if not caracteres_especiales.search(word)]
    return codificacion, resto

# Resaltar errores en el texto
def resaltar_errores(text, faltantes):
    codificacion, resto = clasificar_errores(faltantes)
    for word in codificacion:
        text = re.sub(f"({re.escape(word)})", r'<span style="background-color: yellow;">\1</span>', text, flags=re.IGNORECASE)
    for word in resto:
        text = re.sub(f"({re.escape(word)})", r'<span style="background-color: lightcoral;">\1</span>', text, flags=re.IGNORECASE)
    return text

# Calcular errores por médico
def calcular_errores_por_medico():
    data_informe = deteccion_errores()
    data_informe['Errores codificación'], data_informe['Errores'] = zip(*data_informe['Palabras mal escritas'].apply(clasificar_errores))
    
    errores_por_medico = data_informe.groupby('Nro_Medico').apply(lambda df: {
        'Cant. de informes': df.shape[0],
        'Promedio errores por informe': round(df['Cant. palabras mal escritas'].mean(), 1),
        'Desviacion errores por informe': round(df['Cant. palabras mal escritas'].std(), 1),
        'Cantidad informes con errores': len(df[df['Cant. palabras mal escritas'] > 0]),
        'Cant. de errores total': df['Cant. palabras mal escritas'].sum(),
        'Errores codificación': sum(len(e) for e in df['Errores codificación']),
        'Errores palabras': sum(len(e) for e in df['Errores']),
        'Errores': ', '.join(set(word for faltantes in df['Palabras mal escritas'] for word in faltantes))
    }).apply(pd.Series).reset_index()

    errores_por_medico = errores_por_medico[errores_por_medico['Cantidad informes con errores'] > 0]
    errores_por_medico['Porcentaje informes con errores'] = (errores_por_medico['Cantidad informes con errores'] / errores_por_medico['Cant. de informes']) * 100
    errores_por_medico['Porcentaje informes con errores'] = errores_por_medico['Porcentaje informes con errores'].apply(lambda x: round(x, 1))
    errores_por_medico = errores_por_medico.sort_values(by='Errores palabras', ascending=False)
    return errores_por_medico

# Filtrar datos por médico
def filtrar_por_medico(medicos):
    data_informe_ordenado = deteccion_errores()
    data_medico = data_informe_ordenado[data_informe_ordenado['Nro_Medico'] == int(medicos)]
    errores_por_medico = calcular_errores_por_medico()
    df_filtrado = errores_por_medico[errores_por_medico['Nro_Medico'] == int(medicos)]
    
    columnas_requeridas = ['Fecha_Ing', 'Tipo_OS', 'Nro_OS', 'Nro_Medico', 'Cant. palabras mal escritas', 'Palabras mal escritas']
    columnas_requeridas2 = ['Nro_Medico', 'Cant. de informes', 'Promedio errores por informe', 'Desviacion errores por informe', 'Cantidad informes con errores', 'Porcentaje informes con errores', 'Cant. de errores total', 'Errores codificación', 'Errores palabras', 'Errores']
    
    return df_filtrado[columnas_requeridas2], data_medico[columnas_requeridas]

# Graficar la cantidad de informes con errores por médico
def plot_errores_por_medico():
    data5 = calcular_errores_por_medico()
    data5 = data5[data5['Errores palabras'] > 0]
    
    data5 = data5.sort_values(by='Cantidad informes con errores', ascending=False)
    
    plt.figure(figsize=(12, 8))
    bars = plt.bar(data5['Nro_Medico'].astype(str), data5['Cantidad informes con errores'], color='#d9594c')
    plt.xlabel('Nro_Medico')
    plt.ylabel('Cantidad de informes con errores')
    plt.xticks(rotation=45)
    
    for bar in bars:
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), str(int(bar.get_height())), ha='center', va='bottom')
    
    plt.title('Cantidad de informes con errores por médico')
    plt.tight_layout()
    plt.savefig('Cantidad de informes con errores por médico.png')
    plt.close()
    return 'Cantidad de informes con errores por médico.png'


def graficar_errores_por_medico():
    df_errores = calcular_errores_por_medico()
    
    df_errores_sorted = df_errores.sort_values(by='Porcentaje informes con errores', ascending=False)
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(df_errores_sorted['Nro_Medico'].astype(str), df_errores_sorted['Porcentaje informes con errores'], color='#d9594c')
    plt.xlabel('Número de Médico')
    plt.ylabel('Porcentaje de Informes con Errores')
    plt.title('Porcentaje de Informes con Errores por Médico')
    plt.xticks(rotation=90)
    
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval, f'{yval:.2f}%', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('porcentaje_informes_con_errores.png')
    return 'porcentaje_informes_con_errores.png'


def crear_histograma_errores():
    errores_por_medico = calcular_errores_por_medico()
    errores_por_medico = errores_por_medico[errores_por_medico['Errores palabras'] > 0]
    
    plt.figure(figsize=(14, 7))
    
    # Datos de errores de codificación
    errores_con_vocal = errores_por_medico[['Nro_Medico', 'Errores codificación']].copy()
    errores_con_vocal = errores_con_vocal.fillna(0)  
    plt.subplot(1, 2, 2)
    width = 0.6  
    x = range(len(errores_con_vocal))  
    bars = plt.bar(x, errores_con_vocal['Errores codificación'], color='#f9df74', width=width, align='center')
    plt.xlabel('Nro_Medico')
    plt.ylabel('Cantidad')
    plt.title('Errores codificación')
    plt.xticks(x, errores_con_vocal['Nro_Medico'].astype(str), rotation=90)
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval, str(int(yval)), ha='center', va='bottom')

    # Datos de errores de palabras
    plt.subplot(1, 2, 1)
    x = range(len(errores_por_medico)) 
    bars = plt.bar(x, errores_por_medico['Errores palabras'], color='#d9594c', align='center') 
    plt.xlabel('Nro_Medico')
    plt.ylabel('Cantidad')
    plt.title('Errores palabras')
    plt.xticks(x, errores_por_medico['Nro_Medico'].astype(str), rotation=90)
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval, str(int(yval)), ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig('errores_por_medico.png')
    return 'errores_por_medico.png'

def errores_general():
    data_informe_ordenado = deteccion_errores()
    errores_por_medico = calcular_errores_por_medico()
    
    nan_informe = data['Informe'].isna().sum()
    total_filas= len(data)
    total_informes_informe = total_filas - nan_informe
    mas_de_5_errores_informe = len(data_informe_ordenado[data_informe_ordenado['Cant. palabras mal escritas'] > 5])
    menos_o_igual_5_errores_informe = total_informes_informe - mas_de_5_errores_informe
    porcentaje_mas_de_5_informe = (mas_de_5_errores_informe / total_informes_informe) * 100
    porcentaje_menos_o_igual_5_informe = (menos_o_igual_5_errores_informe / total_informes_informe) * 100
    cantidad_total_errores_informe = data_informe_ordenado['Cant. palabras mal escritas'].sum()
    cantidad_total_errores_codificacion = errores_por_medico['Errores codificación'].sum()
    cantidad_total_errores_palabras = errores_por_medico['Errores palabras'].sum()

    resumen_errores_informe = (
        f"Total de informes: {total_informes_informe}\n"
        f"Cantidad total de errores: {cantidad_total_errores_informe}\n"
        f"Cantidad total de errores de codificación: {cantidad_total_errores_codificacion}\n"
        f"Cantidad total de errores de palabras: {cantidad_total_errores_palabras}\n"
        f"Informes con más de 5 errores:\n"
        f"  Cantidad: {mas_de_5_errores_informe}\n"
        f"  Porcentaje: {porcentaje_mas_de_5_informe:.2f}%\n"
        f"Informes con 5 o menos errores:\n"
        f"  Cantidad: {menos_o_igual_5_errores_informe}\n"
        f"  Porcentaje: {porcentaje_menos_o_igual_5_informe:.2f}%"
    )

    return resumen_errores_informe
