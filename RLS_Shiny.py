import pandas as pd
import plotly.graph_objs as go
import plotly.io as pio
from shiny import App, render, ui

# Función para cargar y validar los datos
def cargar_y_validar_datos(filepath):
    try:
        data = pd.read_excel(filepath)
        if data.shape[1] != 3:
            return None, "El archivo debe contener exactamente tres columnas."
        if not all(isinstance(col, str) for col in data.columns):
            return None, "Cada columna debe tener un encabezado."
        if not pd.api.types.is_numeric_dtype(data.iloc[:, 1]) or not pd.api.types.is_numeric_dtype(data.iloc[:, 2]):
            return None, "Las columnas de la variable independiente y dependiente deben ser numéricas."
        if data.isnull().any().any():
            return None, "Las columnas no deben contener valores nulos."
        print("Archivo cargado y validado exitosamente.")
        return data, None
    except Exception as e:
        print(f"Error al cargar el archivo: {e}")
        return None, "No se pudo cargar el archivo. Verifica que sea un archivo .xlsx válido."

# Función de cálculo de regresión lineal y estadísticos
def calcular_estadisticos(data):
    X = data.iloc[:, 1]
    Y = data.iloc[:, 2]
    n = len(X)
    
    b = X.cov(Y) / X.var()
    a = Y.mean() - b * X.mean()
    r = X.corr(Y)
    r2 = r ** 2
    error_estandar = ((Y - (a + b * X)) ** 2).sum() / (n - 2)
    error_estandar = error_estandar ** 0.5

    return {
        "pendiente": b,
        "interseccion": a,
        "coef_correlacion": r,
        "coef_determinacion": r2,
        "error_estandar": error_estandar,
    }

# Función para crear gráfico de dispersión interactivo y recta
def crear_grafico_interactivo(data, a, b, x_hypothetical=None):
    X = data.iloc[:, 1]
    Y = data.iloc[:, 2]

    # Extender el rango de X para la recta
    X_range = X.max() - X.min()
    X_min, X_max = X.min() - X_range * 0.5, X.max() + X_range * 0.5
    X_line = pd.Series([X_min, X_max])
    Y_line = a + b * X_line

    scatter = go.Scatter(x=X, y=Y, mode="markers", name="Datos")
    line = go.Scatter(x=X_line, y=Y_line, mode="lines", name=f"Y = {a:.2f} + {b:.2f}X")

    if x_hypothetical is not None:
        y_hypothetical = a + b * x_hypothetical
        point = go.Scatter(x=[x_hypothetical], y=[y_hypothetical], mode="markers", marker=dict(color="gold"), name="Punto hipotético")
        fig = go.Figure(data=[scatter, line, point])
    else:
        fig = go.Figure(data=[scatter, line])

    fig.update_layout(
        xaxis_title=data.columns[1],
        yaxis_title=data.columns[2],
        title="Gráfico de dispersión y recta de regresión",
    )

    return fig

# Interfaz Shiny
app_ui = ui.page_fluid(
    ui.row(
        ui.column(
            4,
            ui.input_file("file", "Selecciona un archivo .xlsx", accept=[".xlsx"]),
            ui.output_text("status"),
            ui.input_slider("signo", "Signo", min=-1, max=1, value=1, step=2),
            ui.input_slider("centenas", "Centenas", min=0, max=9, value=0),
            ui.input_slider("decenas", "Decenas", min=0, max=9, value=0),
            ui.input_slider("unidades", "Unidades", min=0, max=9, value=0),
        ),
        ui.column(
            8,
            ui.output_ui("plot"),
            ui.row(
                ui.column(6, ui.output_ui("estadisticos")),
                ui.column(6, ui.output_ui("interpretacion"))
            )
        )
    )
)

def server(input, output, session):
    @output
    @render.text
    def status():
        if input.file() is None:
            return "No se ha cargado ningún archivo."
        else:
            data, error_msg = cargar_y_validar_datos(input.file()[0]['datapath'])
            if error_msg:
                return error_msg
            else:
                return f"Archivo cargado: {input.file()[0]['name']}."

    @output
    @render.ui
    def estadisticos():
        if input.file() is None:
            return "No se ha cargado ningún archivo."
        data, error_msg = cargar_y_validar_datos(input.file()[0]['datapath'])
        if data is not None:
            stats = calcular_estadisticos(data)
            return ui.HTML(
                f"<ul>"
                f"<li>Pendiente (b): {stats['pendiente']:.2f}</li>"
                f"<li>Intersección (a): {stats['interseccion']:.2f}</li>"
                f"<li>Coeficiente de correlación (r): {stats['coef_correlacion']:.2f}</li>"
                f"<li>Coeficiente de determinación (R²): {stats['coef_determinacion']:.2f}</li>"
                f"<li>Error estándar de la estimación: {stats['error_estandar']:.2f}</li>"
                f"</ul>"
            )

    @output
    @render.ui
    def interpretacion():
        if input.file() is None:
            return "No se ha cargado ningún archivo."
        data, error_msg = cargar_y_validar_datos(input.file()[0]['datapath'])
        if data is not None:
            stats = calcular_estadisticos(data)
            x_hypothetical = input.signo() * (input.centenas() * 100 + input.decenas() * 10 + input.unidades())
            y_hypothetical = stats["interseccion"] + stats["pendiente"] * x_hypothetical
            return ui.HTML(f"Si X = {x_hypothetical}, entonces Y = {y_hypothetical:.2f}.")

    @output
    @render.ui
    def plot():
        if input.file() is None:
            return
        data, error_msg = cargar_y_validar_datos(input.file()[0]['datapath'])
        if data is not None:
            x_hypothetical = input.signo() * (input.centenas() * 100 + input.decenas() * 10 + input.unidades())
            stats = calcular_estadisticos(data)
            fig = crear_grafico_interactivo(data, stats["interseccion"], stats["pendiente"], x_hypothetical)
            fig_html = pio.to_html(fig, full_html=False)
            return ui.HTML(fig_html)

app = App(app_ui, server)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)