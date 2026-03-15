import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dash import Dash, dcc, html, Input, Output

# ── Carica dati ───────────────────────────────────────────────────────────────

scores = pd.read_csv('model/scores_output.csv')
sim    = pd.read_csv('simulation/simulation_output.csv')

SCENARIO_COLORS = {'lieve': '#2196F3', 'medio': '#FF9800', 'grave': '#F44336'}
ALL_IDS = sorted(sim['profile_id'].unique())

# ── App ───────────────────────────────────────────────────────────────────────

app = Dash(__name__)

app.layout = html.Div([
    html.H2('FlowScore — Demo', style={'textAlign': 'center', 'fontFamily': 'sans-serif'}),

    html.Div([
        # ── Dropdown profilo ──────────────────────────────────────────────────
        html.Div([
            html.Label('Seleziona profilo:', style={'fontWeight': 'bold'}),
            dcc.Dropdown(
                id='profile-selector',
                options=[{'label': f'Profilo {i}', 'value': i} for i in ALL_IDS],
                value=ALL_IDS[0],
                clearable=False,
                style={'width': '300px'},
            ),
        ]),

        # ── Filtro per FlowScore ──────────────────────────────────────────────
        html.Div([
            html.Label('Filtra per FlowScore:', style={'fontWeight': 'bold'}),
            html.Div(id='score-range-label',
                     style={'fontSize': '13px', 'color': '#555', 'marginBottom': '6px'}),
            dcc.RangeSlider(
                id='score-filter',
                min=0, max=100, step=1,
                value=[0, 100],
                marks={0: '0', 25: '25', 50: '50', 75: '75', 100: '100'},
                tooltip={'placement': 'bottom', 'always_visible': False},
            ),
        ], style={'width': '400px'}),
    ], style={'display': 'flex', 'gap': '60px', 'alignItems': 'flex-end',
              'padding': '20px', 'fontFamily': 'sans-serif'}),

    dcc.Graph(id='main-chart', style={'height': '950px'}),
], style={'maxWidth': '1200px', 'margin': 'auto'})


@app.callback(
    Output('profile-selector', 'options'),
    Output('profile-selector', 'value'),
    Output('score-range-label', 'children'),
    Input('score-filter', 'value'),
    Input('profile-selector', 'value'),
)
def filter_by_score(score_range, current_id):
    lo, hi = score_range
    filtered = scores[(scores['flowscore'] >= lo) & (scores['flowscore'] <= hi)]
    options  = [{'label': f'Profilo {int(r.id)}  ({r.flowscore:.1f})', 'value': int(r.id)}
                for _, r in filtered.iterrows()]
    ids_in   = filtered['id'].tolist()
    new_val  = current_id if current_id in ids_in else (ids_in[0] if ids_in else None)
    label    = f'{len(ids_in)} profili con FlowScore {lo}–{hi}'
    return options, new_val, label


@app.callback(Output('main-chart', 'figure'), Input('profile-selector', 'value'))
def update(profile_id):
    score_row  = scores[scores['id'] == profile_id]
    flowscore  = score_row['flowscore'].values[0] if len(score_row) else None
    sim_p      = sim[sim['profile_id'] == profile_id]

    fig = make_subplots(
        rows=3, cols=2,
        specs=[
            [{'type': 'indicator'},            {'type': 'xy'}],
            [{'type': 'xy'},                   {'type': 'xy'}],
            [{'type': 'xy', 'colspan': 2},     None          ],
        ],
        subplot_titles=(
            f'FlowScore: {flowscore:.1f}/100' if flowscore else 'FlowScore',
            'Traiettoria debito per scenario',
            'Distribuzione FlowScore (tutti i profili)',
            'FlowScore vs Fragility Index',
            '% Debito finale / Debito richiesto per scenario',
            '',
        ),
        vertical_spacing=0.12,
        horizontal_spacing=0.12
    )

    # ── Gauge flowscore ───────────────────────────────────────────────────────
    fig.add_trace(go.Indicator(
        mode='gauge+number',
        value=flowscore,
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': '#4CAF50' if flowscore and flowscore >= 70 else '#FF9800' if flowscore and flowscore >= 40 else '#F44336'},
            'steps': [
                {'range': [0, 40],  'color': '#ffebee'},
                {'range': [40, 70], 'color': '#fff3e0'},
                {'range': [70, 100],'color': '#e8f5e9'},
            ],
            'threshold': {'line': {'color': 'black', 'width': 2}, 'value': flowscore}
        },
        title={'text': 'Solvibilità'},
    ), row=1, col=1)

    # ── Traiettorie debito ────────────────────────────────────────────────────
    for scenario, color in SCENARIO_COLORS.items():
        s = sim_p[sim_p['scenario'] == scenario]
        fig.add_trace(go.Scatter(
            x=s['t'], y=s['debt'],
            mode='lines+markers',
            name=scenario.capitalize(),
            line={'color': color, 'width': 2},
            marker={'size': 5},
            legendgroup=scenario,
        ), row=1, col=2)

    # ── Istogramma flowscore ──────────────────────────────────────────────────
    fig.add_trace(go.Histogram(
        x=scores['flowscore'], nbinsx=30,
        marker_color='steelblue', opacity=0.75,
        name='Tutti', showlegend=False
    ), row=2, col=1)
    if flowscore:
        fig.add_trace(go.Scatter(
            x=[flowscore, flowscore], y=[0, 50],
            mode='lines', line={'color': 'red', 'dash': 'dash', 'width': 2},
            name='Profilo selezionato', showlegend=False
        ), row=2, col=1)

    # ── Scatter flowscore vs fragility_index ─────────────────────────────────
    for scenario, color in SCENARIO_COLORS.items():
        s_all = sim[sim['scenario'] == scenario].drop_duplicates('profile_id')
        merged = s_all.merge(scores, left_on='profile_id', right_on='id')
        fig.add_trace(go.Scatter(
            x=merged['flowscore'], y=merged['fragility_index'],
            mode='markers',
            marker={'color': color, 'size': 4, 'opacity': 0.4},
            name=scenario.capitalize(),
            legendgroup=scenario,
            showlegend=False,
        ), row=2, col=2)

    # Evidenzia profilo selezionato nello scatter
    if flowscore:
        p_scatter = sim_p.drop_duplicates('scenario')
        for _, row in p_scatter.iterrows():
            fig.add_trace(go.Scatter(
                x=[flowscore], y=[row['fragility_index']],
                mode='markers',
                marker={'color': SCENARIO_COLORS[row['scenario']], 'size': 12,
                        'symbol': 'star', 'line': {'width': 1, 'color': 'black'}},
                name=f"#{profile_id} {row['scenario']}",
                showlegend=False,
            ), row=2, col=2)

    # ── Bar chart perc_debito_finale per scenario ─────────────────────────────
    perc_data = sim_p[sim_p['t'] == sim_p['t'].max()].copy()
    for scenario, color in SCENARIO_COLORS.items():
        row_s = perc_data[perc_data['scenario'] == scenario]
        if len(row_s):
            perc = row_s['perc_debito_finale'].values[0] * 100
            fig.add_trace(go.Bar(
                x=[scenario.capitalize()],
                y=[perc],
                name=scenario.capitalize(),
                marker_color=color,
                text=[f'{perc:.1f}%'],
                textposition='outside',
                legendgroup=scenario,
                showlegend=False,
            ), row=3, col=1)

    fig.update_layout(
        height=950,
        legend={'orientation': 'h', 'y': -0.04},
        margin={'t': 60},
    )
    fig.update_xaxes(title_text='Mese', row=1, col=2)
    fig.update_yaxes(title_text='Debito (€)', row=1, col=2)
    fig.update_xaxes(title_text='FlowScore', row=2, col=1)
    fig.update_yaxes(title_text='Conteggio', row=2, col=1)
    fig.update_xaxes(title_text='FlowScore', row=2, col=2)
    fig.update_yaxes(title_text='Fragility Index', row=2, col=2)
    fig.update_xaxes(title_text='Scenario', row=3, col=1)
    fig.update_yaxes(title_text='% Debito finale / Richiesto', row=3, col=1)
    fig.add_trace(go.Scatter(
        x=['Lieve', 'Medio', 'Grave'], y=[100, 100, 100],
        mode='lines', line={'dash': 'dash', 'color': 'gray', 'width': 1},
        showlegend=False, hoverinfo='skip',
    ), row=3, col=1)

    return fig


if __name__ == '__main__':
    app.run(debug=True)
