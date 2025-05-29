from shiny import App, ui, reactive, render

# ------------------- FUNCION DE GAUSS-JORDAN -------------------
def gj(m):
    n = len(m)
    pasos = []
    for j in range(n):
        if m[j][j] == 0:
            try:
                k = next(i for i in range(j+1, n) if m[i][j] != 0)
                m[j], m[k] = m[k], m[j]
            except:
                raise ValueError("No hay solución única (división por cero)")
        m[j] = [x / m[j][j] for x in m[j]]
        for i in range(n):
            if i != j:
                m[i] = [a - m[i][j] * b for a, b in zip(m[i], m[j])]
        pasos.append([row[:] for row in m])
    solucion = [round(r[-1], 6) for r in m]
    return solucion, pasos

# ------------------- INTERFAZ UI -------------------
app_ui = ui.page_fluid(
    ui.tags.style("""
        body {
            background-color: #0d1117;
            color: white;
            font-family: 'Segoe UI', sans-serif;
        }
        .shiny-input-container {
            margin-bottom: 10px;
            color: white;
        }
        .panel-heading {
            font-weight: bold;
            font-size: 20px;
            color: #70e0ff;
            margin-top: 10px;
            border-bottom: 1px solid #70e0ff;
        }
        .btn-primary {
            background-color: #00cfff;
            border-color: #00cfff;
            color: #000;
        }
        .btn-primary:hover {
            background-color: #00e1ff;
        }
        .shiny-text-output {
            background-color: #1a1f2e;
            border: 1px solid #00cfff;
            padding: 15px;
            border-radius: 8px;
            white-space: pre-wrap;
            font-size: 16px;
            color: white;
        }
        input[type="text"], input[type="number"] {
            background-color: #1a1f2e;
            color: white;
            border: 1px solid #00cfff;
            border-radius: 4px;
            padding: 6px;
        }
        label {
            color: white !important;
        }
    """),

    ui.layout_sidebar(
        ui.sidebar(
            ui.input_numeric("n", "Número de incógnitas", 3, min=2, max=6),
            ui.input_checkbox("restricciones", "¿Restringir valores?", False),
            ui.output_ui("restricciones_input"),
            ui.input_action_button("resolver", "Resolver", class_="btn btn-primary"),
        ),
        ui.h2("Método de Gauss-Jordan", class_="panel-heading"),
        ui.output_ui("matriz_input"),
        ui.br(),
        ui.h3("Resultado", class_="panel-heading"),
        ui.output_text_verbatim("resultado", placeholder=True),
    )
)

# ------------------- SERVIDOR -------------------
def server(input, output, session):
    @output
    @render.ui
    def matriz_input():
        return ui.div(
            *[ui.input_text(f"fila_{i}", f"Ecuación {i+1} (separa con espacios)", "") for i in range(input.n())]
        )

    @output
    @render.ui
    def restricciones_input():
        if not input.restricciones():
            return None
        return ui.div(
            *[ui.input_text(f"rango_{i}", f"Rango x{i+1} (min,max)", "") for i in range(input.n())]
        )

    resolver_clicks = reactive.value(0)

    @reactive.effect
    def _():
        if input.resolver() > resolver_clicks():
            resolver_clicks.set(input.resolver())
            output["resultado"].invalidate()

    @output
    @render.text
    def resultado():
        try:
            n = input.n()
            m = []
            for i in range(n):
                fila_str = input.__getitem__(f"fila_{i}")()
                try:
                    valores = list(map(float, fila_str.strip().split()))
                except ValueError:
                    return f"❌ Error: Entrada inválida en Ecuación {i+1}. Asegúrate de usar números separados por espacios."
                if len(valores) != n + 1:
                    return f"⚠️ Ecuación {i+1} debe tener {n + 1} valores (incluyendo el término independiente)."
                m.append(valores)

            solucion, pasos = gj(m)
            res = [f"x{i+1} = {v}" for i, v in enumerate(solucion)]

            if input.restricciones():
                for i in range(n):
                    rango_raw = input.__getitem__(f"rango_{i}")()
                    if rango_raw:
                        try:
                            minimo, maximo = map(float, rango_raw.strip().split(","))
                            if not (minimo <= solucion[i] <= maximo):
                                res.append(f"⚠️ x{i+1} = {solucion[i]} fuera de rango [{minimo}, {maximo}]")
                        except:
                            res.append(f"⚠️ Rango inválido para x{i+1}")

            res.append("\nPasos intermedios:")
            for idx, p in enumerate(pasos):
                res.append(f"Paso {idx+1}:")
                for fila in p:
                    res.append("   " + "  ".join(f"{v:.2f}" for v in fila))
                res.append("")

            return "\n".join(res)

        except Exception as e:
            return f"❌ Error: {e}"

# ------------------- CREAR APP -------------------
app = App(app_ui, server)
