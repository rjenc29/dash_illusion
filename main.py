from typing import Dict, List, Tuple

import dash_bootstrap_components as dbc
import numpy as np
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output, State, Patch

external_stylesheets = [dbc.themes.DARKLY]

app = Dash(
    __name__,
    external_stylesheets=external_stylesheets,
    update_title=None
)
server = app.server

N_POINTS = 60

DIAMOND = np.array(
    [
        [1, 0],
        [0, 1],
        [-1, 0],
        [0, -1],
        [1, 0]
    ]
)


def generate_vertices() -> Dict[str, List[float]]:
    radius = 0.5

    angles = np.linspace(0, 2 * np.pi, N_POINTS)
    xs = np.sin(angles) * radius
    ys = np.cos(angles) * radius

    vertices = DIAMOND * 3

    offsets = np.hstack(
        [
            xs.reshape(-1, 1),
            ys.reshape(-1, 1)
        ]
    )

    out = []
    for i in range(N_POINTS):
        out.append(vertices + offsets[i, :])

    out = np.vstack(out)

    return {
        'x': out[:, 0].tolist(),
        'y': out[:, 1].tolist()
    }


figure = go.Figure(
    layout=dict(
        xaxis=dict(
            scaleanchor="y",
            scaleratio=1,
            ticks="",
            showticklabels=False,
            dtick=1
        ),
        yaxis=dict(
            scaleanchor="y",
            scaleratio=1,
            ticks="",
            showticklabels=False,
            dtick=1
        ),
        showlegend=False,
        margin=dict(l=0, r=0, t=0, b=50),
        plot_bgcolor='rgba(0, 0, 0, 0)',
        paper_bgcolor='rgba(0, 0, 0, 0)'
    )
)

figure.add_trace(
    go.Scatter(
        x=[],
        y=[],
        mode='lines',
        line=dict(width=6)
    )
)


OCCLUDERS = [
    DIAMOND + np.array([3, 0]),
    DIAMOND + np.array([0, 3]),
    DIAMOND + np.array([-3, 0]),
    DIAMOND + np.array([0, -3]),
]


for occluder in OCCLUDERS:
    figure.add_trace(
        go.Scatter(
            x=occluder[:, 0].tolist(),
            y=occluder[:, 1].tolist(),
            fill='toself',
            fillcolor='rgba(34, 34, 34, 1)',
            mode='lines',
            line=dict(color='rgba(34, 34, 34, 1)')
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
                            figure=figure,
                            config={
                                'staticPlot': True
                            }
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
                                    'Invisible',
                                    id='invisible'
                                ),
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
                                dbc.Row(
                                    [
                                        dbc.Col(
                                            [
                                                html.Div(
                                                    'Alpha'
                                                )
                                            ],
                                            width=2
                                        ),
                                        dbc.Col(
                                            [
                                                dcc.Slider(
                                                    min=0,
                                                    max=1,
                                                    value=1,
                                                    id='alpha'
                                                )
                                            ],
                                            width=10
                                        ),
                                    ]
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
        dbc.Row(
            [
                html.Div("Source code: "),
                html.A(
                    "https://github.com/rjenc29/dash_illusion",
                    href='https://github.com/rjenc29/dash_illusion',
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


def make_patch(colour: str) -> Patch:
    patched_figure = Patch()

    for i in range(1, 5):
        patched_figure["data"][i]["fillcolor"] = colour
        patched_figure["data"][i]["line"]["color"] = colour

    return patched_figure


@app.callback(
    Output('graph', 'figure'),
    Output('alpha', 'value'),
    Input('transparent', 'n_clicks'),
    prevent_initial_call=True
)
def make_transparent(n_clicks: int) -> Tuple[Patch, float]:
    colour = 'rgba(34, 34, 34, 0.7)'
    return make_patch(colour=colour), 0.7


@app.callback(
    Output('graph', 'figure', allow_duplicate=True),
    Output('alpha', 'value', allow_duplicate=True),
    Input('grey', 'n_clicks'),
    prevent_initial_call=True
)
def make_grey(n_clicks: int) -> Tuple[Patch, float]:
    colour = 'rgba(100, 100, 100, 0.5)'
    return make_patch(colour=colour), 0.5


@app.callback(
    Output('graph', 'figure', allow_duplicate=True),
    Output('alpha', 'value', allow_duplicate=True),
    Input('invisible', 'n_clicks'),
    prevent_initial_call=True
)
def make_invisible(n_clicks: int) -> Tuple[Patch, float]:
    colour = 'rgba(34, 34, 34, 1)'
    return make_patch(colour=colour), 1.0


@app.callback(
    Output('graph', 'figure', allow_duplicate=True),
    Output('alpha', 'value', allow_duplicate=True),
    Input('visible', 'n_clicks'),
    prevent_initial_call=True
)
def make_visible(n_clicks: int) -> Tuple[Patch, float]:
    colour = 'rgba(0,0,255,1)'
    return make_patch(colour=colour), 1.0


@app.callback(
    Output('graph', 'figure', allow_duplicate=True),
    Input('alpha', 'value'),
    State('graph', 'figure'),
    prevent_initial_call=True
)
def change_alpha(alpha: float, figure: go.Figure) -> Patch:
    colour = figure['data'][-1]['fillcolor']

    rgb = colour.rsplit(',', maxsplit=1)[0]
    new_colour = f'{rgb}, {alpha})'

    return make_patch(colour=new_colour)


if __name__ == '__main__':
    app.run(debug=True)
