import dash_bootstrap_components as dbc
import numpy as np
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output, State, Patch

external_stylesheets = [dbc.themes.DARKLY]

app = Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = html.Div(
    dcc.Graph(id='plot')
)

N_POINTS = 60

DIAMOND_COLOUR = 'rgba(0,0,255,1)'

DIAMOND = np.array(
        [
            [1, 0],
            [0, 1],
            [-1, 0],
            [0, -1],
            [1, 0]
        ]
    )


def generate_vertices():
    radius = 0.3

    angles = np.linspace(0, 2 * np.pi, N_POINTS)
    xs = np.sin(angles) * radius
    ys = np.cos(angles) * radius

    vertices = DIAMOND * 3

    offsets = np.hstack([xs.reshape(-1, 1), ys.reshape(-1, 1)])

    out = []
    for i in range(N_POINTS):
        out.append(vertices + offsets[i, :])

    out = np.vstack(out)

    return {'x': out[:, 0].tolist(), 'y': out[:, 1].tolist()}


def get_vertices(idx: int):
    vertices = generate_vertices()
    return vertices[idx % N_POINTS]


figure = go.Figure(
    layout=dict(
        xaxis=dict(
            scaleanchor="y",
            scaleratio=1,
            visible=False
        ),
        yaxis=dict(
            scaleanchor="y",
            scaleratio=1,
            visible=False
        ),
        showlegend=False,
        margin=dict(l=0, r=0, t=0, b=50),
        plot_bgcolor='rgba(0, 0, 0, 0)',
        paper_bgcolor='rgba(0, 0, 0, 0)'
    )
)

figure.add_trace(go.Scatter(x=[], y=[], mode='lines', line=dict(width=4)))


squares = [
    DIAMOND + np.array([3, 0]),
    DIAMOND + np.array([0, 3]),
    DIAMOND + np.array([-3, 0]),
    DIAMOND + np.array([0, -3]),
]


for square in squares:
    figure.add_trace(
        go.Scatter(
            x=square[:, 0].tolist(),
            y=square[:, 1].tolist(),
            fill='toself',
            fillcolor="#222",
            mode='lines',
            line=dict(color="#222")
        )
    )

figure.layout.template = 'plotly_dark'

app.layout = dbc.Container(
    [
        dbc.Row(
            html.H2('Motion Binding'),
            class_name="mb-3"
        ),
        dbc.Row(
            [
                dbc.Col(
                    [
                        dcc.Graph(
                            id='graph',
                            figure=figure
                        ),
                        dcc.Interval(
                            id='interval',
                            interval=25,
                            n_intervals=0,
                            max_intervals=-1
                        ),
                        dcc.Store(
                            id='store',
                            data=generate_vertices()
                        ),
                        dcc.Store(
                            id='offset',
                            data=0
                        ),
                    ],
                    width=9
                ),
                dbc.Col(
                    [
                        dbc.Stack(
                            [
                                html.Div('Occluders'),
                                dbc.Button(
                                    'Visible',
                                    id='visible'
                                ),
                                dbc.Button(
                                    'Transparent',
                                    id='transparent',
                                ),
                                dbc.Button(
                                    'Grey',
                                    id='grey'
                                ),
                                dbc.Button(
                                    'Invisible',
                                    id='invisible'
                                ),
                            ],
                            gap=2
                        )
                    ],
                    width=3
                )
            ]
        ),
        dbc.Row(
            [
                html.Div("Inspired by Michael Bach's: "),
                html.A(
                    "152 Visual Phenomena & Optical Illusions",
                    href='https://michaelbach.de/ot/mot-motionBinding/index.html',
                    target='_blank'
                )
            ],
            class_name="mb-3"
        ),
    ]
)


app.clientside_callback(
    """
    function (n_intervals, data, offset) {
        offset = offset % data.x.length;
        const end = offset + 5;
        return [[{x: [data.x.slice(offset, end)], y: [data.y.slice(offset, end)]}, [0], 5], end]
    }
    """,
    Output('graph', 'extendData'),
    Output('offset', 'data'),
    Input('interval', 'n_intervals'),
    State('store', 'data'),
    State('offset', 'data')
)


@app.callback(
    Output('graph', 'figure'),
    Input('transparent', 'n_clicks'),
    prevent_initial_call=True
)
def make_transparent(n_clicks: int):
    patched_figure = Patch()

    transparent = 'rgba(0, 0, 0, 0.5)'

    for i in range(1, 5):
        patched_figure["data"][i]["fillcolor"] = transparent
        patched_figure["data"][i]["line"]["color"] = transparent

    return patched_figure


@app.callback(
    Output('graph', 'figure', allow_duplicate=True),
    Input('grey', 'n_clicks'),
    prevent_initial_call=True
)
def make_grey(n_clicks: int):
    patched_figure = Patch()

    transparent = 'rgba(100, 100, 100, 0.5)'

    for i in range(1, 5):
        patched_figure["data"][i]["fillcolor"] = transparent
        patched_figure["data"][i]["line"]["color"] = transparent

    return patched_figure


@app.callback(
    Output('graph', 'figure', allow_duplicate=True),
    Input('invisible', 'n_clicks'),
    prevent_initial_call=True
)
def make_invisible(n_clicks: int):
    patched_figure = Patch()

    transparent = "#222"

    for i in range(1, 5):
        patched_figure["data"][i]["fillcolor"] = transparent
        patched_figure["data"][i]["line"]["color"] = transparent

    return patched_figure


@app.callback(
    Output('graph', 'figure', allow_duplicate=True),
    Input('visible', 'n_clicks'),
    prevent_initial_call=True
)
def make_visible(n_clicks: int):
    patched_figure = Patch()

    for i in range(1, 5):
        patched_figure["data"][i]["fillcolor"] = DIAMOND_COLOUR
        patched_figure["data"][i]["line"]["color"] = DIAMOND_COLOUR

    return patched_figure


if __name__ == '__main__':
    app.run(debug=True)
