import pandas as pd
import requests
from io import BytesIO
import dash
from dash import dcc
from dash import html
from dash import dash_table, Output,Input, callback_context
import plotly.graph_objs as go
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import numpy as np

#-------------------------------------------------------------Carga de datos--------------------------------------------------------------------------------------------------------
# Enlace convertido de descarga directa
onedrive_url = "https://immaipu-my.sharepoint.com/personal/karen_arancibia_maipu_cl/_layouts/15/download.aspx?share=Ec0mjRB4LrtCowyUYI11QVEBQ-XnXwoR48kodbA5qBUprg"

try:
    # Descargar el archivo
    response = requests.get(onedrive_url)
    response.raise_for_status()  # Verifica errores en la descarga

    # Leer el archivo Excel directamente desde la memoria
    df = pd.read_excel(BytesIO(response.content))
    df['Dif registro'] = pd.to_datetime(df['Dif registro'],dayfirst=True,format='mixed')
    print("Datos cargados con éxito:")
    #print(df.head())  # Muestra las primeras filas

except requests.exceptions.RequestException as e:
    print(f"Error al descargar el archivo: {e}")
except Exception as e:
    print(f"Error al procesar el archivo: {e}")


#-------------------------------------------------------------Graficos--------------------------------------------------------------------------------------------------------

import plotly.graph_objs as go

def crear_boton(df, columna, es_verde, columna_caudal):
    """
    Crea un "botón" en forma de círculo basado en el último valor de una columna.
    
    Args:
        df (pd.DataFrame): DataFrame con los datos.
        columna (str): Nombre de la columna del DataFrame.
        es_verde (str): condicion para el boton.
        
    Returns:
        go.Figure: Figura de Plotly con el "botón".
    """
    try:
        # Verificar que la columna exista en el DataFrame
        if columna not in df.columns:
            raise ValueError(f"La columna '{columna}' no existe en el DataFrame.")
        
        # Obtener el último valor de la columna
        ultimo_valor = df[columna].iloc[-1]
        
        
        # Definir el color del círculo
        color = 'green' if es_verde==ultimo_valor else 'red'

        #Valor Caudal
        ultimo_valor_caudal = df[columna_caudal].iloc[-1] if not columna_caudal==False else None

        #anotacion del boton:
        anotation_text= f"Caudal: {ultimo_valor_caudal} l/s " if not pd.isna(ultimo_valor_caudal) else "Sin informacion disponible"
        
        # Crear el gráfico con un solo punto
        fig = go.Figure(
            data=go.Scatter(
                x=[ultimo_valor],  # Eje X: último valor
                y=[0],             # Eje Y: posición fija (puedes cambiarla)
                mode='markers',
                marker=dict(
                    color=color,
                    size=100  # Tamaño grande para que parezca un botón
                ),
                name=f"Botón {'Verde' if es_verde else 'Rojo'}"
            )
        )
        
        # Configurar el diseño del gráfico
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',  # Fondo transparente
            plot_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(title=columna),
            yaxis=dict(title='', visible=False),  # Oculta el eje Y
            showlegend=False,
            annotations=[
                dict(
                    x=ultimo_valor,
                    y=-0.05,  # Posición debajo del botón
                    text=anotation_text,
                    showarrow=False,
                    font=dict(size=12, color="black")
                )
            ],
            height=400,  # Altura del gráfico
            width=300,   # Ancho del gráfico
        )
        
        return fig
    
    except Exception as e:
        print(f"Error al crear el botón: {e}")
        return None

#Crear Estanque

def generar_grafico_estanque(nivel_maximo, nivel_actual,estanque,unidad):
    """
    Genera un gráfico estilo "estanque" que muestra el nivel máximo y el nivel actual de agua.

    Args:
        nivel_maximo (float): Nivel máximo del estanque en metros.
        nivel_actual (float): Nivel actual del estanque en metros.
        estanque (string): Nombre del estanque.
        unidad (string): Unidad de medida del nivel.

    Returns:
        plotly.graph_objects.Figure: Gráfico del estanque.
    """
    # Crear la figura
    fig = go.Figure()

    # Agregar la barra del contenedor del estanque (nivel máximo)
    fig.add_trace(go.Bar(
        x=["Estanque"],
        y=[nivel_maximo],  # Altura máxima
        marker=dict(
            color='lightgrey',  # Fondo del estanque
            line=dict(color='lightgrey', width=10)  # Bordes del estanque
        ),
        hoverinfo='none',  # Sin información al pasar el mouse
        name="Capacidad Máxima",
        width=0.5  # Grosor de la barra
    ))

    # Agregar la barra del nivel actual (agua dentro del estanque)
    fig.add_trace(go.Bar(
        x=["Estanque"],
        y=[nivel_actual],  # Nivel actual de agua
        marker=dict(color='lightblue'),  # Color del agua
        hovertemplate=f"Nivel Actual: {nivel_actual:.2f} m",  # Mostrar el nivel actual en metros
        name="Nivel Actual",
        width=0.5
    ))

    # Ajustar el diseño para que parezca un estanque
    fig.update_layout(
        title=f"Nivel del Estanque {estanque}",
        barmode='overlay',  # Superponer barras
        yaxis=dict(
            title=f"Nivel [{unidad}]",
            range=[0, nivel_maximo],  # Rango desde 0 hasta el máximo
            showgrid=False,  # Sin líneas de la cuadrícula
            ticksuffix=f" {unidad}"  # Sufijo para mostrar unidades en el eje
        ),
        xaxis=dict(
            showticklabels=False  # Sin etiquetas en el eje X
        ),
        plot_bgcolor="rgba(0,0,0,0)",  # Fondo transparente
        height=500,  # Altura del gráfico
        showlegend=False,  # Ocultar leyenda
    )

    return fig

#Indicador generico

import plotly.graph_objects as go

def crear_indicador(valor, unidad,titulo, rangos_colores=None, width=300, height=300):
    """
    Crea un indicador grande con colores según el valor de la variable.
    
    Args:
        valor (float): Valor de la variable a mostrar.
        unidad (str): Unidad de la variable (por ejemplo, "l/s", "°C").
        rangos_colores (list of tuples, optional): Lista de rangos y colores. 
                                                   Ejemplo: [(0, 'red'), (20, 'yellow'), (60, 'green')].
                                                   Si no se proporciona, se usan rangos predeterminados.
    
    Returns:
        go.Figure: Figura de Plotly con el indicador.
    """
    # Rango de colores por defecto
    if rangos_colores is None:
        rangos_colores = [(0, 'red'), (20, 'yellow'), (60, 'green'), (float('inf'), 'blue')]
    
    # Determinar el color según el rango del valor
    color = 'black'  # color predeterminado
    for limite, color_rango in rangos_colores:
        if valor <= limite:
            color = color_rango
            break
    
    # Calcular el tamaño del título en función del ancho de la figura

    
    # Crear el indicador
    fig = go.Figure(go.Indicator(
        mode="number",
        value=valor,
        number={'suffix': f" {unidad}", 'font': {'size': 80, 'color': color}},
        title={"text": titulo, 'font': {'size': 20}},
        domain={'x': [0, 1], 'y': [0, 1]}
    ))

    # Configurar diseño del gráfico
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",  # Fondo transparente
        plot_bgcolor="rgba(0,0,0,0)",
        height=height,
        width=width
    )
    
    return fig





def crear_grafico_dia(dia, dataset, columna,variable,rango,width=1500,height=500):
    """
    Genera un gráfico que muestra las mediciones del día específico con un eje X fijo (00:00 a 24:00).

    Args:
        dia (datetime): Día de interés.
        dataset (DataFrame): Dataset con las mediciones.
        columna (str): Nombre de la columna con los datos a graficar.
        variable (str): Nombre del titulo y eje y.
        rango (tuple): rango valores para el eje y.

    Returns:
        go.Figure: Figura de Plotly con el gráfico.
    """
    # Asegurarse de que las columnas auxiliares existen
    if not {'año', 'mes', 'día', 'hora'}.issubset(dataset.columns):
        dataset['año'] = dataset['Dif registro'].dt.year
        dataset['mes'] = dataset['Dif registro'].dt.month
        dataset['dia'] = dataset['Dif registro'].dt.day
        dataset['hora'] = pd.to_datetime(dataset['Hora Registro'], format='%H:%M:%S', errors='coerce').dt.hour
        dataset['hora_registro'] = pd.to_datetime(dataset['Dif registro'].dt.strftime('%H:%M:%S'), format='%H:%M:%S', errors='coerce')
        dataset['hora_registro']=dataset['hora_registro'].apply(
            lambda row: row.hour + row.minute / 60 if pd.notnull(row) else None
        )
        dataset = dataset.dropna(subset=['hora_registro', 'hora'])
        
        dataset[columna] = dataset.apply(
            lambda row: np.nan if abs(row['hora_registro'] - row['hora']) >= 3 else row[columna],
            axis=1
        )
        
        dataset.loc[:, 'fecha'] = pd.to_datetime(
        dict(year=dataset['año'], month=dataset['mes'], day=dataset['dia']),
        errors='coerce'
    )
    dataset.loc[:, 'fecha'] = dataset['fecha'].dt.normalize()

    
    

    # Convertir el día seleccionado a datetime
    
    dia = pd.to_datetime(dia)
    
    

    

    # Filtrar los datos que coinciden con el día seleccionado
    datos_dia = dataset[
        (dataset['año'] == dia.year) & 
        (dataset['mes'] == dia.month) & 
        (dataset['dia'] == dia.day)
    ]
    #print(datos_dia)

    # Filtrar los datos de los últimos 7 días utilizando las columnas auxiliares
    inicio_7_dias = pd.to_datetime(dia - pd.Timedelta(days=7)).normalize()
    datos_7_dias = dataset[
        (dataset['fecha'] >= inicio_7_dias) & (dataset['fecha'] < dia)
    ]
    

    # Calcular el promedio por hora para los últimos 7 días
    promedio_7_dias = datos_7_dias.groupby('hora')[columna].mean()

    # Filtrar los datos del mes
    datos_mes = dataset[(dataset['mes'] == dia.month) & (dataset['año'] == dia.year)]
    promedio_mes = datos_mes.groupby('hora')[columna].mean()
    #print(dia.day,dia.month,dia.year)
    #print(dataset['mes'])
 
 

    # Filtrar los datos del año
    datos_año = dataset[(dataset['año'] == dia.year)]
    promedio_año = datos_año.groupby('hora')[columna].mean()

    # Crear el gráfico
    fig = go.Figure()

    # Mediciones del día seleccionado
    fig.add_trace(go.Scatter(
        x=datos_dia['hora'], 
        y=datos_dia[columna], 
        mode='lines+markers', 
        name='Medicion del Día',
        connectgaps=False
    ))

    # Promedio por hora de los últimos 7 días
    fig.add_trace(go.Scatter(
        x=promedio_7_dias.index, 
        y=promedio_7_dias.values, 
        mode='lines+markers', 
        name='Promedio 7 Días Anteriores',
        connectgaps=False,
        line=dict(dash='dash'),
        visible=False  # Inicialmente oculto
    ))

    # Promedio por hora del mes
    fig.add_trace(go.Scatter(
        x=promedio_mes.index, 
        y=promedio_mes.values, 
        mode='lines+markers', 
        name='Promedio Mes',
        connectgaps=False,
        line=dict(dash='dot'),
        visible=False  # Inicialmente oculto
    ))

    # Promedio por hora del año
    fig.add_trace(go.Scatter(
        x=promedio_año.index, 
        y=promedio_año.values, 
        mode='lines+markers', 
        name='Promedio Año',
        connectgaps=False,
        line=dict(dash='dashdot'),
        visible=False  # Inicialmente oculto
    ))

    # Crear los botones para alternar las curvas
    buttons = [
        {
            'label': 'Diario',
            'method': 'update',
            'args': [{'visible': [True, True, True, True]}, {'title': f"Mediciones del dia de {columna} - Día: {dia.strftime('%d-%m-%Y')}"}]
        },
        {
            'label': 'Semanal',
            'method': 'update',
            'args': [{'visible': [True, True, False, False]}, {'title': f"Promedio 7 Días Anteriores de {columna} - Día: {dia.strftime('%d-%m-%Y')}"}]
        },
        {
            'label': 'Mensual',
            'method': 'update',
            'args': [{'visible': [True, False, True, False]}, {'title': f"Promedio del Mes de {columna} - Día: {dia.strftime('%d-%m-%Y')}"}]
        },
        {
            'label': 'Anual',
            'method': 'update',
            'args': [{'visible': [True, False, False, True]}, {'title': f"Promedio del Año de {columna} - Día: {dia.strftime('%d-%m-%Y')}"}]
        }
    ]

    # Añadir el layout con los botones
    fig.update_layout(
        title=f"Mediciones de {columna} - Día: {dia.strftime('%d-%m-%Y')}",
        xaxis_title="Hora del Día",
        yaxis_title=variable,
        xaxis=dict(
            tickmode='array',
            tickvals=list(range(24)),
            ticktext=[f'{i:02}:00' for i in range(24)],
            range=[0, 23]
        ),
        yaxis=dict(
            title=variable,
            range=rango
        ),
        showlegend=True,
        legend=dict(
            traceorder='normal',  # Asegura que el orden de la leyenda no se vea afectado
            valign="top",
            x=0.65,  # Ubicación fija de la leyenda
            y=0.95
        ),
        autosize=False,  # Evita que el gráfico cambie de tamaño al alternar curvas
        width=width,  # Ancho fijo
        height=height,
        updatemenus=[{
            'buttons': buttons,
            'direction': 'down',
            'showactive': True,
            'active': 0,
            'x': 0.17,
            'xanchor': 'left',
            'y': 1.15,
            'yanchor': 'top'
        }]
    )

    return fig

def crear_grafico_horometro(mes,ano,dataset,columna,titulo):
        print(mes,ano)
        if not {'año', 'mes', 'día', 'hora'}.issubset(dataset.columns):
            dataset['año'] = dataset['Dif registro'].dt.year
            dataset['mes'] = dataset['Dif registro'].dt.month
            dataset['dia'] = dataset['Dif registro'].dt.day
            dataset['hora'] = pd.to_datetime(dataset['Hora Registro'], format='%H:%M:%S', errors='coerce').dt.hour
            dataset['hora_registro'] = pd.to_datetime(dataset['Dif registro'].dt.strftime('%H:%M:%S'), format='%H:%M:%S', errors='coerce')
            dataset['hora_registro']=dataset['hora_registro'].apply(
                lambda row: row.hour + row.minute / 60 if pd.notnull(row) else None
            )
            dataset = dataset.dropna(subset=['hora_registro', 'hora'])
            
            dataset[columna] = dataset.apply(
                lambda row: np.nan if abs(row['hora_registro'] - row['hora']) >= 3 else row[columna],
                axis=1
            )
            
            dataset.loc[:, 'fecha'] = pd.to_datetime(
            dict(year=dataset['año'], month=dataset['mes'], day=dataset['dia']),
            errors='coerce'
        )
        dataset.loc[:, 'fecha'] = dataset['fecha'].dt.normalize()

        # Filtrar los datos para el mes y año específicos
        datos_mes = dataset[(dataset['año'] == ano) & (dataset['mes'] == mes)]

        # Considerar solo la primera anotación por día (por cada fecha)
        datos_mes = datos_mes.groupby('fecha').first().reset_index()

        # Rellenar los días sin datos (forward fill)
        datos_mes = datos_mes.set_index('fecha').sort_index()
        # Crear una columna para marcar los días sin registros
        datos_mes['sin_dato'] = datos_mes[columna].isna()
        #llenar vacios
        datos_mes[columna] = datos_mes[columna].fillna(method='ffill')

        # Calcular la diferencia del horómetro
        datos_mes['diferencia'] = datos_mes[columna].diff().shift(-1)

 
        # Crear un DataFrame con todos los días del mes (1 al 31) como índice, siempre
        dias_mes = pd.Series(range(1, 32), name='dia')  # Serie de días del 1 al 31
        datos_mes_completos = pd.DataFrame(index=dias_mes)

        # Cambiar el índice de `datos_mes` al día del mes
        datos_mes = datos_mes.set_index('dia')

        # Unir los datos para asegurar que todos los días estén representados
        datos_mes_completos = datos_mes_completos.join(datos_mes[['diferencia', 'sin_dato']], how='left')

        # Crear el gráfico
        fig = go.Figure()

        # Agregar barras al gráfico
        fig.add_trace(go.Bar(
            x=datos_mes_completos.index,  # Días del mes
            y=datos_mes_completos['diferencia'],  # Diferencias del horómetro
            marker_color=['red' if x else 'skyblue' for x in datos_mes_completos['sin_dato']],  # Colores
            name='Diferencia Horómetro'
        ))

        # Personalizar el gráfico
        fig.update_layout(
            title=titulo,
            xaxis_title="Día del Mes",
            yaxis_title="Diferencia del Horómetro",
            xaxis=dict(tickmode='linear', tick0=1, dtick=1),  # Mostrar todos los días
            yaxis=dict(tickformat=".2f",range=[0,27]),  # Formato de las diferencias
            bargap=0.2,  # Separación entre barras
            template="plotly_white"  # Estilo
        )

        return fig

   




# Crear la aplicación Dash
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.MATERIA]) #usamos un tema por defecto


#-------------------------------------------------------------LAYOUT--------------------------------------------------------------------------------------------------------
# Layout de la aplicación
app.layout = dbc.Container(
    [
        # TITULO DASHBOARD (PLANTA SAN JOSE DE CHUCHUNCO)
        html.H1("PEAP SAN JOSE DE CHUCHUNCO", style={'text-align': 'center'}),

        
        # POZOS
        html.H2("Funcionamiento de pozos"),
        html.Div(
            children=[
                dcc.Graph(id='botonPozo1A'),
                dcc.Graph(id='botonPozo2A'),
                dcc.Graph(id='botonPozo3A'),
                dcc.Graph(id='botonPozo4A'),
                dcc.Graph(id='botonPozo5'),
            ],
            style={
                'display': 'grid',
                'grid-template-columns': 'repeat(5, 1fr)',  # 5 columnas
                'gap': '1px',
                'justify-items': 'center',
                'align-items': 'center',
                'max-width': '100%'
            }
        ),

        # Componente de intervalo que genera eventos cada 120 segundos
        dcc.Interval(
            id='intervalo-actualizacion',
            interval=120 * 1000,  # 120 segundos
            n_intervals=0
        ),

        # NIVELES DE ESTANQUE
        html.H2("Niveles de estanque"),
        html.Div(
            children=[
                dcc.Graph(id='Estanque3000', config={'displayModeBar': False}),
                dcc.Graph(id='Estanque6000', config={'displayModeBar': False}),
            ],
            style={
                'display': 'grid',
                'grid-template-columns': 'repeat(auto-fit, minmax(300px, 1fr))',  # Ajuste dinámico
                'gap': '20px',
                'justify-items': 'center',
                'align-items': 'center',
                'width': '100%',
                'padding': '10px',
                'box-sizing': 'border-box'
            }
        ),

        # BOTONES BOOSTER
        html.H2("Funcionamiento de Booster"),
        html.Div(
            children=[
                dcc.Graph(id='botonBooster1'),
                dcc.Graph(id='botonBooster2'),
                dcc.Graph(id='botonBooster3'),
                dcc.Graph(id='botonBooster4'),
                dcc.Graph(id='botonBooster5'),
            ],
            style={
                'display': 'grid',
                'grid-template-columns': 'repeat(5, 1fr)',
                'gap': '1px',
                'justify-items': 'center',
                'align-items': 'center',
                'max-width': '100%'
            }
        ),

        # Indicadores caudal
        html.H2("Caudales de salida"),
        html.Div(
            children=[
                dcc.Graph(id='indicador_caudal_estanque_1'),
                dcc.Graph(id='indicador_caudal_estanque_2')
            ],
            style={
                'display': 'grid',
                'grid-template-columns': 'repeat(2, 1fr)',
                'gap': '10px',
                'justify-items': 'center',
                'align-items': 'center',
                'max-width': '100%'
            }
        ),

        # PICKER DE FECHA Y GRÁFICOS HISTÓRICOS
        html.H1("Mediciones historicas"),
        html.Div(
            children=[
                # Colocamos el DatePicker en una fila separada, arriba de los gráficos
                dcc.DatePickerSingle(
                    id='date-picker',
                    date=pd.Timestamp.today(),
                    display_format='YYYY-MM-DD',
                    style={'padding': '10px', 'width': '100%', 'textAlign': 'center'}
                ),
            ],
            style={
                'display': 'grid',
                'justify-items': 'center',
                'align-items': 'center',
                'width': '100%',
                'padding': '20px',
            }
        ),
        html.H2("Mediciones historicas de estanques"),
        # Los gráficos históricos se muestran debajo del DatePicker
        html.Div(
            children=[
                dcc.Graph(id='caudal_6000_hist')
            ],
            style={
                'display': 'grid',
                'grid-template-columns': 'repeat(1, 1fr)',
                'gap': '10px',
                'justify-items': 'center',
                'align-items': 'center',
                'max-width': '100%'
            },
        ),
        html.Div(
            children=[
                dcc.Graph(id='presion_3000_hist')
            ],
            style={
                'display': 'grid',
                'grid-template-columns': 'repeat(1, 1fr)',
                'gap': '10px',
                'justify-items': 'center',
                'align-items': 'center',
                'max-width': '100%'
            },
        ),
        html.Div(
            children=[
                dcc.Graph(id='nivel_3000_hist')
            ],
            style={
                'display': 'grid',
                'grid-template-columns': 'repeat(1, 1fr)',
                'gap': '10px',
                'justify-items': 'center',
                'align-items': 'center',
                'max-width': '100%'
            },
        ),
        html.Div(
            children=[
                dcc.Graph(id='presion_6000_hist')
            ],
            style={
                'display': 'grid',
                'grid-template-columns': 'repeat(1, 1fr)',
                'gap': '10px',
                'justify-items': 'center',
                'align-items': 'center',
                'max-width': '100%'
            },
        ),
        html.Div(
            children=[
                dcc.Graph(id='nivel_6000_hist')
            ],
            style={
                'display': 'grid',
                'grid-template-columns': 'repeat(1, 1fr)',
                'gap': '10px',
                'justify-items': 'center',
                'align-items': 'center',
                'max-width': '100%'
            },
        ),        
        #Mediciones historicas pozos
        html.H2("Mediciones historicas de pozos"),
        # Los gráficos históricos se muestran debajo del DatePicker
        html.Div(
            children=[
                dcc.Graph(id='caudal_pozo_1a_hist'),
                dcc.Graph(id='caudal_pozo_2a_hist'),
                dcc.Graph(id='caudal_pozo_3a_hist'),
                dcc.Graph(id='caudal_pozo_4a_hist'),
                dcc.Graph(id='caudal_pozo_5_hist')
            ],
            style={
                'display': 'grid',
                'grid-template-columns': 'repeat(2, 3fr)',
                'gap': '10px',
                'justify-items': 'center',
                'align-items': 'center',
                'max-width': '100%'
            },
        ),

        # HOROMETROS
        html.H1("Horómetros"),
        html.Div(
            children=[
                html.Div(
                    children=[
                        html.Label("Seleccionar Año:", style={"font-weight": "bold", "margin-right": "10px"}),
                        dcc.Dropdown(
                            id="selector-ano",
                            options=[{"label": str(ano), "value": ano} for ano in range(2020, 2031)],
                            value=2024,
                            clearable=False,
                            style={"width": "150px"},
                        ),
                    ],
                    style={"margin-bottom": "15px", "display": "flex", "align-items": "center"},
                ),
                html.Div(
                    children=[
                        html.Label("Seleccionar Mes:", style={"font-weight": "bold", "margin-right": "10px"}),
                        dcc.Dropdown(
                            id="selector-mes",
                            options=[
                                {"label": f"{i:02d} - {pd.to_datetime(i, format='%m').strftime('%B')}", "value": i}
                                for i in range(1, 13)
                            ],
                            value=12,
                            clearable=False,
                            style={"width": "200px"},
                        ),
                    ],
                    style={"margin-bottom": "15px", "display": "flex", "align-items": "center"},
                ),
            ],
            style={"display": "flex", "flex-direction": "column", "align-items": "center", "justify-content": "center"},
        ),
        html.H2("Horómetros de Pozos"),
        html.Div(
            children=[
                dcc.Graph(id="horometro_pozo_1a"),
                dcc.Graph(id="horometro_pozo_2a"),
                dcc.Graph(id="horometro_pozo_3a"),
                dcc.Graph(id="horometro_pozo_4a"),
                dcc.Graph(id="horometro_pozo_5"),
            ],
            style={
                'display': 'grid',
                'grid-template-columns': 'repeat(2, 1fr)',
                'gap': '20px',
                'width': '100%',
                'max-width': '1200px',
                'margin': '20px auto',
            },
        ),
        html.H2("Horómetros de Boosters"),
        html.Div(
            children=[
                dcc.Graph(id="horometro_booster_1"),
                dcc.Graph(id="horometro_booster_2"),
                dcc.Graph(id="horometro_booster_3"),
                dcc.Graph(id="horometro_booster_4"),
                dcc.Graph(id="horometro_booster_5"),
            ],
            style={
                'display': 'grid',
                'grid-template-columns': 'repeat(2, 1fr)',
                'gap': '20px',
                'width': '100%',
                'max-width': '1200px',
                'margin': '20px auto',
            },
        ),
        # TABLA CON LAS MEDICIONES
        html.Div(
            dash_table.DataTable(
                id='tabla-datos',
                data=df.to_dict('records'),
                columns=[{"name": i, "id": i} for i in df.columns],
                style_table={'height': '400px', 'overflowY': 'auto', 'overflowX': 'auto'}
            )
        ),






    ],
    fluid=True,  # Activar el contenedor fluido para que ocupe todo el ancho
    style={'padding': '0', 'max-width': '100%'},
)

# Callback para actualizar solo la tabla cada 60 segundos
@app.callback(
    [Output('tabla-datos', 'data'),Output('botonPozo1A', 'figure'),Output('botonPozo2A', 'figure'),Output('botonPozo3A', 'figure'),Output('botonPozo4A', 'figure'),Output('botonPozo5', 'figure'),
    Output('Estanque3000', 'figure'),Output('Estanque6000', 'figure'),
    Output('botonBooster1', 'figure'),Output('botonBooster2', 'figure'),Output('botonBooster3', 'figure'),Output('botonBooster4', 'figure'),Output('botonBooster5', 'figure'),
    Output('indicador_caudal_estanque_1', 'figure'),Output('indicador_caudal_estanque_2', 'figure'),
    Output('caudal_6000_hist','figure', allow_duplicate=True),Output('presion_3000_hist','figure', allow_duplicate=True),Output('nivel_3000_hist','figure', allow_duplicate=True),
    Output('presion_6000_hist','figure', allow_duplicate=True),Output('nivel_6000_hist','figure', allow_duplicate=True),
    Output('horometro_pozo_1a','figure', allow_duplicate=True),Output('horometro_pozo_2a','figure', allow_duplicate=True),Output('horometro_pozo_3a','figure', allow_duplicate=True),Output('horometro_pozo_4a','figure', allow_duplicate=True),
    Output('horometro_pozo_5','figure', allow_duplicate=True),
    Output('horometro_booster_1','figure', allow_duplicate=True),Output('horometro_booster_2','figure', allow_duplicate=True),Output('horometro_booster_3','figure', allow_duplicate=True),Output('horometro_booster_4','figure', allow_duplicate=True),
    Output('horometro_booster_5','figure', allow_duplicate=True),
    Output('caudal_pozo_1a_hist','figure', allow_duplicate=True),Output('caudal_pozo_2a_hist','figure', allow_duplicate=True),Output('caudal_pozo_3a_hist','figure', allow_duplicate=True),Output('caudal_pozo_4a_hist','figure', allow_duplicate=True),
    Output('caudal_pozo_5_hist','figure', allow_duplicate=True)],  # Salida de la tabla
    [Input('intervalo-actualizacion', 'n_intervals')],
    prevent_initial_call='initial_duplicate'  # Trigger: cada 60 segundos
)
def actualizar_dashboard(n_intervals):
    try:
        # Descargar y actualizar los datos de la tabla cada 60 segundos
        response = requests.get(onedrive_url)
        response.raise_for_status()  # Verifica errores en la descarga
        
        # Leer los nuevos datos
        df_actualizado = pd.read_excel(BytesIO(response.content))
        df=df_actualizado
        df['Dif registro'] = pd.to_datetime(df['Dif registro'],dayfirst=True,format='mixed')
        #Botones pozos
        figura_boton_1a = crear_boton(df_actualizado, 'Pozo 1A - Funcionamiento', 'Sí','Pozo 1A - Caudal l/s')
        figura_boton_2a = crear_boton(df_actualizado, 'Pozo 2A - Funcionamiento', 'Sí','Pozo 2A - Caudal l/s')
        figura_boton_3a = crear_boton(df_actualizado, 'Pozo 3A - Funcionamiento', 'Sí','Pozo 3A - Caudal l/s')
        figura_boton_4a = crear_boton(df_actualizado, 'Pozo 4A - Funcionamiento', 'Sí','Pozo 4A - Caudal l/s')
        figura_boton_5 = crear_boton(df_actualizado, 'Pozo 5 - Funcionamiento', 'Sí','Pozo 5 - Caudal l/s')
        #Estanques
        figura_estanque_3000= generar_grafico_estanque(3000, df['Elevado 3000 m3 - Nivel de Estanque - m3'].iloc[-1],'elevado 3000','m3')#cambiar volumen por altura
        figura_estanque_6000= generar_grafico_estanque(6, df['Semi Enterrado 6000 m3 - Nivel de Estanque - m'].iloc[-1],'semi enterrado 6000','m')#cambiar volumen por altura
        #Botones Booster
        figura_boton_booster1 = crear_boton(df_actualizado, 'Booster 1 - Funcionamiento', 'Sí',False)
        figura_boton_booster2 = crear_boton(df_actualizado, 'Booster 2 - Funcionamiento', 'Sí',False)
        figura_boton_booster3 = crear_boton(df_actualizado, 'Booster 3 - Funcionamiento', 'Sí',False)
        figura_boton_booster4 = crear_boton(df_actualizado, 'Booster 4 - Funcionamiento', 'Sí',False)
        figura_boton_booster5 = crear_boton(df_actualizado, 'Booster 5 - Funcionamiento', 'Sí',False)
        #Indicadores
        Indicador_estanque_1 = crear_indicador(df_actualizado['Semi Enterrado 6000 m3 - Caudal de salida l/s'].iloc[-1],'l/s',"Estanque 6000m3")
        Indicador_estanque_2 = crear_indicador(df_actualizado['Elevado 3000 m3 - Caudal de salida l/s'].iloc[-1],'l/s',"Estanque 3000m3")
        #Historicos
        Caudal_6000_hist=crear_grafico_dia(df_actualizado['Dif registro'].iloc[-1],df_actualizado,'Semi Enterrado 6000 m3 - Caudal de salida l/s','Caudal de salida l/s',[0,300])
        presion_3000_hist=crear_grafico_dia(df_actualizado['Dif registro'].iloc[-1],df_actualizado,'Elevado 3000 m3 - Presion de Salida - MCA','Presión de Salida Estanque Elevado- MCA',[0,30])
        nivel_3000_hist=crear_grafico_dia(df_actualizado['Dif registro'].iloc[-1],df_actualizado,'Elevado 3000 m3 - Nivel de Estanque - m3','Nivel Estanque Elevado - M3',[0,3000])
        presion_6000_hist=crear_grafico_dia(df_actualizado['Dif registro'].iloc[-1],df_actualizado,'Semi Enterrado 6000 m3 - Presión de Salida - MCA','Presión de Salida Estanque Semienterrado- MCA',[0,35])
        nivel_6000_hist=crear_grafico_dia(df_actualizado['Dif registro'].iloc[-1],df_actualizado,'Semi Enterrado 6000 m3 - Nivel de Estanque - m','Nivel de Estanque Semienterrado - m',[0,6])
        #Historicos pozos
        caudal_1a_hist=crear_grafico_dia(df_actualizado['Dif registro'].iloc[-1],df_actualizado,'Pozo 1A - Caudal l/s','Caudal Pozo 1A',[0,100],width=900,height=500)
        caudal_2a_hist=crear_grafico_dia(df_actualizado['Dif registro'].iloc[-1],df_actualizado,'Pozo 2A - Caudal l/s','Caudal Pozo 2A',[0,100],width=900,height=500)
        caudal_3a_hist=crear_grafico_dia(df_actualizado['Dif registro'].iloc[-1],df_actualizado,'Pozo 3A - Caudal l/s','Caudal Pozo 3A',[0,100],width=900,height=500)
        caudal_4a_hist=crear_grafico_dia(df_actualizado['Dif registro'].iloc[-1],df_actualizado,'Pozo 4A - Caudal l/s','Caudal Pozo 4A',[0,100],width=900,height=500)
        caudal_5_hist=crear_grafico_dia(df_actualizado['Dif registro'].iloc[-1],df_actualizado,'Pozo 5 - Caudal l/s','Caudal Pozo 5',[0,100],width=900,height=500)

        #Horometro
        horometro_pozo_1a=crear_grafico_horometro(df_actualizado['Dif registro'].iloc[-1].month,df_actualizado['Dif registro'].iloc[-1].year,df_actualizado,'Pozo 1A - Horómetro','Pozo 1A - Diferencia Horómetro')
        horometro_pozo_2a=crear_grafico_horometro(df_actualizado['Dif registro'].iloc[-1].month,df_actualizado['Dif registro'].iloc[-1].year,df_actualizado,'Pozo 2A - Horómetro','Pozo 2A - Diferencia Horómetro')
        horometro_pozo_3a=crear_grafico_horometro(df_actualizado['Dif registro'].iloc[-1].month,df_actualizado['Dif registro'].iloc[-1].year,df_actualizado,'Pozo 3A - Horómetro','Pozo 3A - Diferencia Horómetro')
        horometro_pozo_4a=crear_grafico_horometro(df_actualizado['Dif registro'].iloc[-1].month,df_actualizado['Dif registro'].iloc[-1].year,df_actualizado,'Pozo 4A - Horómetro','Pozo 4A - Diferencia Horómetro')
        horometro_pozo_5=crear_grafico_horometro(df_actualizado['Dif registro'].iloc[-1].month,df_actualizado['Dif registro'].iloc[-1].year,df_actualizado,'Pozo 5 - Horómetro','Pozo 5 - Diferencia Horómetro')
        #Horometro booster

        horometro_booster_1=crear_grafico_horometro(df_actualizado['Dif registro'].iloc[-1].month,df_actualizado['Dif registro'].iloc[-1].year,df_actualizado,'Booster 1 - Horómetro','Booster 1 - Diferencia Horómetro')
        horometro_booster_2=crear_grafico_horometro(df_actualizado['Dif registro'].iloc[-1].month,df_actualizado['Dif registro'].iloc[-1].year,df_actualizado,'Booster 2 - Horómetro','Booster 2 - Diferencia Horómetro')
        horometro_booster_3=crear_grafico_horometro(df_actualizado['Dif registro'].iloc[-1].month,df_actualizado['Dif registro'].iloc[-1].year,df_actualizado,'Booster 3 - Horómetro','Booster 3 - Diferencia Horómetro')
        horometro_booster_4=crear_grafico_horometro(df_actualizado['Dif registro'].iloc[-1].month,df_actualizado['Dif registro'].iloc[-1].year,df_actualizado,'Booster 4 - Horómetro','Booster 4 - Diferencia Horómetro')
        horometro_booster_5=crear_grafico_horometro(df_actualizado['Dif registro'].iloc[-1].month,df_actualizado['Dif registro'].iloc[-1].year,df_actualizado,'Booster 5 - Horómetro','Booster 5 - Diferencia Horómetro')
  

        
        return (df_actualizado.to_dict('records'),figura_boton_1a,figura_boton_2a,figura_boton_3a,
        figura_boton_4a,figura_boton_5,figura_estanque_3000,figura_estanque_6000,figura_boton_booster1,
        figura_boton_booster2,figura_boton_booster3,figura_boton_booster4,figura_boton_booster5,
        Indicador_estanque_1,Indicador_estanque_2,
        Caudal_6000_hist,presion_3000_hist,nivel_3000_hist,presion_6000_hist,nivel_6000_hist,
        horometro_pozo_1a,horometro_pozo_2a,horometro_pozo_3a,horometro_pozo_4a,horometro_pozo_5,
        horometro_booster_1,horometro_booster_2,horometro_booster_3,horometro_booster_4,horometro_booster_5,
        caudal_1a_hist,caudal_2a_hist,caudal_3a_hist,caudal_4a_hist,caudal_5_hist
         
        )# Devolver los nuevos datos para la tabla
    except requests.exceptions.RequestException as e:
        print(f"Error al descargar el archivo: {e}")
        return dash.no_update  # No actualizar en caso de error

@app.callback(
    Output('caudal_6000_hist', 'figure'),Output('presion_3000_hist', 'figure'),Output('nivel_3000_hist','figure'),Output('presion_6000_hist','figure'),Output('nivel_6000_hist','figure'),
    Output('caudal_pozo_1a_hist','figure'),Output('caudal_pozo_2a_hist','figure'),Output('caudal_pozo_3a_hist','figure'),Output('caudal_pozo_4a_hist','figure'),
    Output('caudal_pozo_5_hist','figure'),
    [Input('date-picker', 'date')]
)
def actualizar_grafico(fecha):
    # Obtener el contexto del callback

    # Convertir las fechas si están presentes
    fecha = pd.to_datetime(fecha).strftime('%Y-%m-%d') if fecha else None
    
    # Si ambos han cambiado, actualizar ambos gráficos
    figura_caudal_6000 = crear_grafico_dia(fecha, df, 'Semi Enterrado 6000 m3 - Caudal de salida l/s', 'Caudal de salida l/s', [0, 300])
    figura_presion_3000 = crear_grafico_dia(fecha, df, 'Elevado 3000 m3 - Presion de Salida - MCA', 'Presión de Salida - MCA', [0, 30])
    figura_nivel_3000 = crear_grafico_dia(fecha, df, 'Elevado 3000 m3 - Nivel de Estanque - m3','NIvel Estanque - M3',[0,3000])
    figura_presion_6000=crear_grafico_dia(fecha,df,'Semi Enterrado 6000 m3 - Presión de Salida - MCA','Presión de Salida Estanque Semienterrado- MCA',[0,35])
    figura_nivel_6000=crear_grafico_dia(fecha,df,'Semi Enterrado 6000 m3 - Nivel de Estanque - m','Nivel de Estanque Semienterrado - m',[0,6])
    figura_1a_hist=crear_grafico_dia(fecha,df,'Pozo 1A - Caudal l/s','Caudal Pozo 1A',[0,100],width=900,height=500)
    figura_2a_hist=crear_grafico_dia(fecha,df,'Pozo 2A - Caudal l/s','Caudal Pozo 2A',[0,100],width=900,height=500)
    figura_3a_hist=crear_grafico_dia(fecha,df,'Pozo 3A - Caudal l/s','Caudal Pozo 3A',[0,100],width=900,height=500)
    figura_4a_hist=crear_grafico_dia(fecha,df,'Pozo 4A - Caudal l/s','Caudal Pozo 4A',[0,100],width=900,height=500)
    figura_5_hist=crear_grafico_dia(fecha,df,'Pozo 5 - Caudal l/s','Caudal Pozo 5',[0,100],width=900,height=500)
    return figura_caudal_6000, figura_presion_3000,figura_nivel_3000,figura_presion_6000,figura_nivel_6000,figura_1a_hist,figura_2a_hist,figura_3a_hist,figura_4a_hist,figura_5_hist

@app.callback(
    Output("horometro_pozo_1a", "figure"),Output("horometro_pozo_2a", "figure"),Output("horometro_pozo_3a", "figure"),Output("horometro_pozo_4a", "figure"),Output("horometro_pozo_5", "figure"),  # Gráfico actualizado
    Output("horometro_booster_1", "figure"), Output("horometro_booster_2", "figure"), Output("horometro_booster_3", "figure"), Output("horometro_booster_4", "figure"), Output("horometro_booster_5", "figure"), 
    [Input("selector-mes", "value"), Input("selector-ano", "value")],  # Mes y Año seleccionados
)
def actualizar_grafico_horometro(mes, ano):
    # Crear el gráfico con el mes y año seleccionados
    horometro_pozo_1a = crear_grafico_horometro(mes, ano, df,'Pozo 1A - Horómetro','Pozo 1A - Diferencia Horómetro')
    horometro_pozo_2a = crear_grafico_horometro(mes, ano, df,'Pozo 2A - Horómetro','Pozo 2A - Diferencia Horómetro')
    horometro_pozo_3a = crear_grafico_horometro(mes, ano, df,'Pozo 3A - Horómetro','Pozo 3A - Diferencia Horómetro')
    horometro_pozo_4a = crear_grafico_horometro(mes, ano, df,'Pozo 4A - Horómetro','Pozo 4A - Diferencia Horómetro')
    horometro_pozo_5 = crear_grafico_horometro(mes, ano, df,'Pozo 5 - Horómetro','Pozo 5 - Diferencia Horómetro')
    horometro_booster_1=crear_grafico_horometro(mes, ano, df,'Booster 1 - Horómetro','Booster 1 - Diferencia Horómetro')
    horometro_booster_2=crear_grafico_horometro(mes, ano, df,'Booster 2 - Horómetro','Booster 2 - Diferencia Horómetro')
    horometro_booster_3=crear_grafico_horometro(mes, ano, df,'Booster 3 - Horómetro','Booster 3 - Diferencia Horómetro')
    horometro_booster_4=crear_grafico_horometro(mes, ano, df,'Booster 4 - Horómetro','Booster 4 - Diferencia Horómetro')
    horometro_booster_5=crear_grafico_horometro(mes, ano, df,'Booster 5 - Horómetro','Booster 5 - Diferencia Horómetro')
    return horometro_pozo_1a,horometro_pozo_2a,horometro_pozo_3a,horometro_pozo_4a,horometro_pozo_5,horometro_booster_1,horometro_booster_2,horometro_booster_3,horometro_booster_4,horometro_booster_5


# Ejecutar la aplicación
if __name__ == '__main__':
    app.run_server(debug=True)
