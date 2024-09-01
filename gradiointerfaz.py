# -*- coding: utf-8 -*-
"""
Created on Sun Sep  1 14:58:09 2024

@author: emili
"""

import gradio as gr
from funciones import *

estilos = """

# /* para botones dentro de las pestañas */
# .tabs > div > div > button {
#     color: black;
#     padding: 5px 6px;  
#     cursor: pointer;
#     width: 80px; 
# }

# /* para pestañas principales */
# button {
#     background-color: #5cb8d1; 
#     color: white;
#     padding: 5px 12px;  
#     cursor: pointer;
#     margin: 5px 0;
# }


# /* color al pasar el mouse botones pestañas principales */
# button:hover {
#     background-color: #247287;
# }

.calidad {
    color: black;
    padding: 5px 6px;  
    cursor: pointer;  
    width: 90px; /* Un poco más ancho */
    height: 30px; /* Un poco más alto */
    border-radius: 4px; /* Bordes no tan redondeados */
    font-size: 14px; /* Tamaño de la letra más pequeño */
}

/* descargar como excel*/
.textbox{
    width: 300px;  
    height: 60px; 
    box-sizing: border-box;
}


.input{
    width: 465px;  
    height: 80px;  
    box-sizing: border-box;
    padding: 5px;
    margin: 5px;
}

.output{
    width: 465px;  
    height: 80px;  
    box-sizing: border-box;
    padding: 5px;
    margin: 5px;
    border: 1px solid #ccc;
}


.output2{
    width: 470px;  
    height: 80px;  
    box-sizing: border-box;
    padding: 5px;
    margin: 5px;
    border: 1px solid #ccc;
    transition: border-color 0.3s ease;
}


.output2.success {
    color: green;
}

.output2.error {
    color: red;
}

.output3{
    width: 450px;    
    padding: 5px;
    margin: 5px;
    border: 1px solid #ccc;
}

.subirarchivo{
    width: 350px;  
    height:150px;  
}

"""

# theme = gr.themes.Soft().set(
#     button_large_padding='*button_small_padding',
#     button_large_text_size='*body_text_size'
# )

theme = gr.themes.Soft().set(
    body_background_fill='*neutral_50',
    background_fill_primary='*button_primary_border_color',
    block_title_text_color='*primary_600',
    panel_border_color='*neutral_100',
    panel_border_color_dark='*neutral_400',
    input_border_color_dark='*color_accent',
    block_label_background_fill= '*neutral_100',
    checkbox_label_background_fill_selected= '*primary_200',
    checkbox_label_text_color_selected='black'
)


def buscar_palabras(palabras):
    df = buscar_por_palabras(palabras)
    archivo = descargar_df(df, "busqueda_palabras.xlsx")
    return df, archivo

def buscar_frase(frase):
    df = buscar_por_frase(frase)
    archivo = descargar_df(df, "busqueda_frase.xlsx")
    return df, archivo

def buscar_palabrasdc(palabras):
    df = buscar_por_palabrasdc(palabras)
    archivo = descargar_df(df, "busqueda_palabrasdc.xlsx")
    return df, archivo

def buscar_frasedc(frase):
    df = buscar_por_frasedc(frase)
    archivo = descargar_df(df, "busqueda_frasedc.xlsx")
    return df, archivo

def buscar_anomalias():
    df = posibles_anomalias_pancreas()
    archivo = descargar_df(df, "anomalias_pancreas.xlsx")
    return df, archivo
  
def calcular_errores():
    df = deteccion_errores()
    archivo = descargar_df(df, "deteccion_errores.xlsx")
    return df[['Fecha_Ing', 'Tipo_OS', 'Nro_OS', 'Nro_Medico', 'Cant. palabras mal escritas', 'Palabras mal escritas']], archivo

def calcular_errores_pormedico():
    df = calcular_errores_por_medico()
    archivo = descargar_df(df, "deteccion_errores_por_medico.xlsx")
    return df, archivo


def descargar_df_medico(medicos):
    df_filtrado, data_medico = filtrar_por_medico(medicos)
    df_filtrado_path = descargar_df(df_filtrado, "resumen_medico.xlsx")
    df_medico_path = descargar_df(data_medico, "informes_medico.xlsx")
    return df_filtrado_path, df_medico_path
    
def mostrar_histogramas():
    return graficar_errores_por_medico(), plot_errores_por_medico(), crear_histograma_errores()

def mostrar_histogramas2():
    return grafico_genero_por_tipo_de_acto(), plot_genero_distribucion()
                
def calcular_distribucion_genero():
    df, hist = mostrar_distribucion_genero_histograma()
    archivo = descargar_df(df, "distribucion_genero.xlsx")
    return df, hist, archivo

def calcular_genero_acto():
    df = genero_por_tipo_de_acto()
    archivo = descargar_df(df, "genero_por_tipo_de_acto.xlsx")
    return df, archivo

def calcular_edad_acto():
    df = estadisticas_edad_tipo_de_acto()
    archivo = descargar_df(df, "edad_por_tipo_de_acto.xlsx")
    return df, archivo

def submit_on_enter(numero_os):
    return numero_os

def handle_cambio_archivo(file):
    _, mensaje, estado = verificar_y_cargar_archivo(file)
    if estado == "success":
        background_color = "#7ad3be"
    else:
        background_color = "lightcoral"
    mensaje_coloreado = f"""
   <div style="background-color: {background_color}; width: 470px; height: 80px; box-sizing: border-box; padding: 5px; margin: 5px; color: black; border-radius: 6px;">
    {mensaje}
   </div>
    """
    
    return mensaje_coloreado


def mostrar_distribucion_genero_histograma():
    distribucion_df = distribucion_genero()
    histograma = histograma_edad_genero()
    return distribucion_df, histograma

def descargar_df(df, filename="dataframe.xlsx"):
    filepath = f"./{filename}"
    df.to_excel(filepath, index=False)
    return filepath

def actualizar_campos(tipo):
    if tipo == "Búsqueda por palabras":
        return gr.update(visible=True, label="Inserte palabras (separadas por espacios)"), gr.update(visible=False), gr.update()
    else:
        return gr.update(visible=False), gr.update(visible=True, label="Inserte frase (literal)"), gr.update()

def realizar_busqueda(entrada):
    tipo = tipo_busqueda.value
    if tipo == "Búsqueda por palabras":
        return buscar_palabras(entrada)
    else:
        return buscar_frase(entrada)
    
# bloques de la interfaz
with gr.Blocks(css= estilos,theme= theme) as interfaz:

    with gr.Tab("Cargar Datos"):
        with gr.Row():
            with gr.Column(scale=7):
                archivo_carga = gr.File(label="Archivo .xls", elem_classes="subirarchivo")
                resultado_carga = gr.HTML(label="Resultado de la Carga")
            with gr.Column(scale=10):
                pass

    archivo_carga.change(handle_cambio_archivo, inputs=archivo_carga, outputs=resultado_carga)

        
    with gr.Tab("Calidad"):
        with gr.Tabs():
            with gr.Tab("Calidad General"):
                with gr.Row():
                    with gr.Column(scale=7):
                        gr.Markdown("## Resumen de errores")
                        boton_errores_general = gr.Button("Calcular", elem_classes= "calidad")
                        texto_errores_general = gr.Textbox(show_label= False, elem_classes="output3")
                        boton_errores_general.click(lambda: errores_general(), None, texto_errores_general)
                    with gr.Column(scale=10):
                        gr.Markdown("## Cantidad de campos completos")
                        boton_cant_completos = gr.Button("Calcular", elem_classes= "calidad")
                        imagen_cant_completos = gr.Image(show_label=False)
                        boton_cant_completos.click(lambda: cantidad_completos(), None, imagen_cant_completos)
                    
            with gr.Tab("Detección de Errores"):
                gr.Markdown("## Detección de errores por informe")
                botonerrores = gr.Button("Calcular", elem_classes="calidad")
                df_errores = gr.DataFrame()
                archivo_errores = gr.File(label="Descargar como Excel", elem_classes="textbox")
                botonerrores.click(calcular_errores, None, [df_errores, archivo_errores])

        with gr.Tabs():
            with gr.Tab("Histogramas de errores por Médico"):
                gr.Markdown("## Errores por Médico")
                boton_histogramas = gr.Button("Calcular", elem_classes="calidad")

                with gr.Row():
                    histomed1 = gr.Image(label="Porcentaje informes con errores")
                    histomed2 = gr.Image(label="Cantidad informes con errores")
                    histomed = gr.Image(label="Tipos de errores ")
                        
                boton_histogramas.click(mostrar_histogramas, None, [histomed1, histomed2, histomed])
                
            with gr.Tab("Errores por Médico"):
                gr.Markdown("## Estadísticas errores por Médico")
                botonerroresxmedico = gr.Button("Calcular", elem_classes="calidad")
                df_erroresxmedico = gr.DataFrame()
                archivo_erroresxmedico = gr.File(label="Descargar como Excel", elem_classes="textbox")
                botonerroresxmedico.click(calcular_errores_pormedico, None, [df_erroresxmedico, archivo_erroresxmedico])
            with gr.Tab("Filtrar por médico"):
                gr.Markdown("## Filtrar por médico")
                with gr.Row():
                    with gr.Column(scale=7):
                        input_medicos = gr.Textbox(label="Ingrese número de médico")
                        boton_filtrar = gr.Button("Buscar", elem_classes="calidad")
                    with gr.Column(scale=10):
                        pass
                df_filtrado = gr.DataFrame(label="Estadísticas de errores")
                df_medico = gr.DataFrame(label="Informes del Médico")
                archivo_filtrado = gr.File(label="Descargar resumen de errores Médico")
                archivo_medico = gr.File(label="Descargar informes del médico")
               
                def actualizar_componentes(medicos):
                    datos_filtrados, data_medico = filtrar_por_medico(medicos)
                    df_filtrado_path, df_medico_path = descargar_df_medico(medicos)
                    return datos_filtrados, data_medico, df_filtrado_path, df_medico_path

            boton_filtrar.click(fn=actualizar_componentes, inputs=[input_medicos], outputs=[df_filtrado, df_medico, archivo_filtrado, archivo_medico])
            input_medicos.submit(fn=actualizar_componentes, inputs=[input_medicos], outputs=[df_filtrado, df_medico, archivo_filtrado, archivo_medico])
            
    with gr.Tab("Estadísticas Básicas"):
        with gr.Row():
            with gr.Column(scale=7):
                gr.Markdown("## Periodo de tiempo")
                gr.Button("Calcular", elem_classes="calidad").click(lambda: periodo_tiempo(), None, gr.Textbox(label="Años/meses/días", elem_classes="output"))
            with gr.Column(scale=10):
                pass
        gr.Markdown("## Distribución de género")
        boton_genero = gr.Button("Calcular", elem_classes="calidad")
        with gr.Row():
            df_genero = gr.DataFrame()
            hist_genero = gr.Image(label="Histograma de Edad por Género")
        archivo_genero = gr.File(label="Descargar como Excel", elem_classes="textbox")

        boton_genero.click(calcular_distribucion_genero, None, [df_genero, hist_genero, archivo_genero])
        
        with gr.Tabs():
            with gr.Tab("Resumen género por tipo de acto"):
                gr.Markdown("## Distribución de género por los 10 tipos de acto más frecuentes")
                boton_histogramas2 = gr.Button("Calcular", elem_classes="calidad")

                with gr.Row():
                    histogen2 = gr.Image(show_label=False)
                    histogen3= gr.Image(show_label=False)

                boton_histogramas2.click(mostrar_histogramas2, None, [histogen2, histogen3])
                
            with gr.Tab("Distribución de género por tipo de acto"):
                boton_acto = gr.Button("Calcular", elem_classes="calidad")
                df_genero_acto = gr.DataFrame(label="Distribución de género por tipo de acto")
                archivo_genero_acto = gr.File(label="Descargar como Excel", elem_classes="textbox")
                boton_acto.click(calcular_genero_acto, None, [df_genero_acto, archivo_genero_acto])
        
        with gr.Tabs():
            with gr.Tab("Resumen edad por tipo de acto"):
                gr.Markdown("## Distribución de edad por los 5 tipos de acto más frecuentes")
                gr.Button("Calcular", elem_classes="calidad").click(lambda: grafico_top5_estudios(), None, gr.Image(label="Distribución de edad por los 5 tipos de acto más frecuentes", width=600, height=500))
            with gr.Tab("Distribución de edad por tipo de acto"):
                boton_acto2 = gr.Button("Calcular", elem_classes="calidad")
                df_edad_acto = gr.DataFrame(label="Distribución de edad por tipo de acto")
                archivo_edad_acto = gr.File(label="Descargar como Excel", elem_classes="textbox")
                boton_acto2.click(calcular_edad_acto, None, [df_edad_acto, archivo_edad_acto])
        
    with gr.Tab("Búsquedas"):
        with gr.Tabs():
            with gr.Tab("Búsqueda en Informe"):
                with gr.Row():
                    with gr.Column(scale=7):
                        gr.Markdown("## Selecciona el tipo de búsqueda")
                        tipo_busqueda = gr.Radio(
                            ["Búsqueda por palabras", "Búsqueda por frase"],
                            show_label= False,
                            value="Búsqueda por palabras",  
                        )
                        
                        palabras_input = gr.Textbox(
                            label="Inserte palabras (separadas por espacios)",
                            elem_classes="input",
                            interactive=True,
                        )
                        
                        frase_input = gr.Textbox(
                            label="Inserte frase (literal)",
                            elem_classes="input",
                            interactive=True,
                        )
                        
                        boton_buscar = gr.Button("Buscar", elem_classes="calidad")
        
                    with gr.Column(scale=10):
                        df_busqueda = gr.DataFrame(label="Resultados de búsqueda")
                        archivo_busqueda = gr.File(label="Descargar como Excel", elem_classes="textbox")
        
                
                palabras_input.visible = True
                frase_input.visible = False
            
                tipo_busqueda.change(fn=actualizar_campos, inputs=tipo_busqueda, outputs=[palabras_input, frase_input, archivo_busqueda])
        
                palabras_input.submit(fn=lambda x: buscar_palabras(x), inputs=palabras_input, outputs=[df_busqueda, archivo_busqueda])
                frase_input.submit(fn=lambda x: buscar_frase(x), inputs=frase_input, outputs=[df_busqueda, archivo_busqueda])
                
            with gr.Tab("Búsqueda en Dato Clínico"):
                with gr.Row():
                    with gr.Column(scale=7):
                        gr.Markdown("## Selecciona el tipo de búsqueda")
                        tipo_busqueda = gr.Radio(
                            ["Búsqueda por palabras", "Búsqueda por frase"],
                            show_label= False,
                            value="Búsqueda por palabras",  
                        )
                        
                        palabras_input = gr.Textbox(
                            label="Inserte palabras (separadas por espacios)",
                            elem_classes="input",
                            interactive=True,
                        )
                        
                        frase_input = gr.Textbox(
                            label="Inserte frase (literal)",
                            elem_classes="input",
                            interactive=True,
                        )
                        
                        boton_buscar = gr.Button("Buscar", elem_classes="calidad")
        
                    with gr.Column(scale=10):
                        df_busqueda = gr.DataFrame(label="Resultados de búsqueda")
                        archivo_busqueda = gr.File(label="Descargar como Excel", elem_classes="textbox")
        
                
                palabras_input.visible = True
                frase_input.visible = False
            
                tipo_busqueda.change(fn=actualizar_campos, inputs=tipo_busqueda, outputs=[palabras_input, frase_input, archivo_busqueda])
        
                palabras_input.submit(fn=lambda x: buscar_palabrasdc(x), inputs=palabras_input, outputs=[df_busqueda, archivo_busqueda])
                frase_input.submit(fn=lambda x: buscar_frasedc(x), inputs=frase_input, outputs=[df_busqueda, archivo_busqueda])    
                
        with gr.Row():
            with gr.Column(scale=7):
                gr.Markdown("## Estudios con posibles anomalías en el Páncreas")
                gr.Markdown("**Criterio de búsqueda:** Combinación de palabras categorizadas como patologías en los informes, junto con términos relacionados con el páncreas y sus enfermedades")
                boton_anomalias = gr.Button("Buscar", elem_classes="calidad")
            with gr.Column(scale=10):  
                df_anomalias = gr.DataFrame(label="Resultados posible Páncreas patológico")
                archivo_anomalias = gr.File(label="Descargar como Excel", elem_classes="textbox")
            boton_anomalias.click(buscar_anomalias, None, [df_anomalias, archivo_anomalias])

    with gr.Tab("Informe"):
        with gr.Row():
            with gr.Column():
                numero_os_input = gr.Textbox(label="Inserte número de OS")
            with gr.Column():
                buscar_button = gr.Button("Buscar", elem_classes="calidad")
            with gr.Column():
                informe_output = gr.HTML()

        with gr.Row():
            leyendas_output = gr.HTML()
            
       
        numero_os_input.submit(submit_on_enter, inputs=numero_os_input, outputs=numero_os_input).then(
           mostrar_informe_resaltado, inputs=numero_os_input, outputs=[leyendas_output, informe_output]
       )

        buscar_button.click(fn=mostrar_informe_resaltado, inputs=numero_os_input, outputs=[leyendas_output, informe_output])      
            
if __name__ == "__main__":
    interfaz.launch()