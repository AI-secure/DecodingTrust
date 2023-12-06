import json
import dash
import numpy as np
from dash import dcc
import plotly.colors
from dash import html
from itertools import chain
import plotly.graph_objects as go
from dash.dependencies import Input, Output
from plotly.subplots import make_subplots


DEFAULT_PLOTLY_COLORS = plotly.colors.DEFAULT_PLOTLY_COLORS


def to_rgba(rgb, alpha=1):
    return 'rgba' + rgb[3:][:-1] + f', {alpha})'


PERSPECTIVES = [
    "Toxicity", "Stereotype Bias", "Adversarial Robustness", "Out-of-Distribution Robustness",
    "Robustness to Adversarial<br>Demonstrations", "Privacy", "Machine Ethics", "Fairness"
]
PERSPECTIVES_LESS = [
    "Toxicity", "Adversarial Robustness", "Out-of-Distribution Robustness",
    "Robustness to Adversarial Demonstrations", "Privacy", "Machine Ethics", "Fairness"
]


MAIN_SCORES = {
    "gpt-3.5-turbo-0301": [
        47,  # Toxicity
        87,  # Bias
        (67.37 + 49.23 + 50.42 + 59.73) / 4,  # Adv
        73.58311416938508,  # OoD
        0.8128416017653167 * 100,  # Adv Demo
        100 - 29.87106667,  # Privacy
        86.38,  # Machine Ethics
        100 * (1 - 0.2243)  # Fairness
    ],
    "gpt-4-0314": [
        41,  # Toxicity
        77,  # Bias
        (78.18 + 55.64 + 58.99 + 63.34) / 4,  # Adv
        87.54700929561338,  # OoD
        0.7794299606265144 * 100,  # Adv Demo
        100 - 33.8863,  # Privacy
        76.60,  # Machine Ethics
        100 * (1 - 0.3633)  # Fairness
    ],
    "alpaca-native": [
        22,  # Toxicity
        43,  # Bias
        (61.53 + 46.01 + 31.75) / 3,  # Adv
        51.785353417708116,  # OoD
        0.3415288335064037 * 100,  # Adv Demo
        100 - 53.60593333,  # Privacy
        30.43,  # Machine Ethics
        100 * (1 - 0.0737)  # Fairness
    ],
    "vicuna-7b-v1.3": [
        28,  # Toxicity
        81,  # Bias
        (52.55 + 52.21 + 51.71) / 3,  # Adv
        59.099378173030225,  # OoD
        0.5798818449290412 * 100,  # Adv Demo
        100 - 27.0362,  # Privacy
        48.22, # Machine Ethics
        100 * (1 - 0.1447)  # Fairness
    ],
    "Llama-2-7b-chat-hf": [
        80,  # Toxicity
        97.6,  # Bias
        (70.06 + 43.11 + 39.87) / 3,  # Adv
        75.65278958829596,  # OoD
        0.5553782796815506 * 100,  # Adv Demo
        100 - 2.605133333,  # Privacy
        40.58,  # Machine Ethics
        100  # Fairness
    ],
    "mpt-7b-chat": [
        40,  # Toxicity
        84.6,  # Bias
        (71.73 + 48.37 + 18.50) / 3,  # Adv
        64.26350715713153,  # OoD
        0.5825403080650745 * 100,  # Adv Demo
        100 - 21.07083333,  # Privacy
        26.11,  # Machine Ethics
        100 - 0  # Fairness
    ],
    "falcon-7b-instruct": [
        39,  # Toxicity
        87,  # Bias
        (73.92 + 41.58 + 16.44) / 3,  # Adv
        51.4498348176422,  # OoD
        0.33947969885773627 * 100,  # Adv Demo
        100 - 29.73776667,  # Privacy
        50.28,  # Machine Ethics
        100 - 0  # Fairness
    ],
    "RedPajama-INCITE-7B-Instruct": [
        18,
        73,  # Bias
        (66.02 + 48.22 + 20.20) / 3,  # Adv
        54.21313771953284,  # OoD
        0.5850598823122187 * 100,
        100 - 23.36082,  # Privacy
        27.49,  # Ethics
        100  # Fairness
    ]
}

ADV_TASKS = ["sst2", "qqp", "mnli"]
adv_results = {
    "hf/mosaicml/mpt-7b-chat": {"sst2": {"acc": 71.73}, "qqp": {"acc": 48.37}, "mnli": {"acc": 18.50}},
    "hf/togethercomputer/RedPajama-INCITE-7B-Instruct": {"sst2": {"acc": 66.02}, "qqp": {"acc": 48.22}, "mnli": {"acc": 20.2}},
    "hf/tiiuae/falcon-7b-instruct": {"sst2": {"acc": 73.92}, "qqp": {"acc": 41.58}, "mnli": {"acc": 16.44}},
    "hf/lmsys/vicuna-7b-v1.3": {"sst2": {"acc": 52.55}, "qqp": {"acc": 52.21}, "mnli": {"acc": 51.71}},
    "hf/chavinlo/alpaca-native": {"sst2": {"acc": 61.53}, "qqp": {"acc": 46.01}, "mnli": {"acc": 31.75}},
    "hf/meta-llama/Llama-2-7b-chat-hf": {"sst2": {"acc": 31.75}, "qqp": {"acc": 43.11}, "mnli": {"acc": 39.87}},
    "openai/gpt-3.5-turbo-0301": {"sst2": {"acc": 70.78}, "qqp": {"acc": 48.72}, "mnli": {"acc": 50.18}},
    "openai/gpt-4-0314": {"sst2": {"acc": 80.43}, "qqp": {"acc": 46.25}, "mnli": {"acc": 60.87}}
}

OOD_TASK = {"knowledge": ["qa_2020", "qa_2023"],
            "style": ["base", "shake_w", "augment", "shake_p0", "shake_p0.6", "bible_p0", "bible_p0.6", "romantic_p0",
                      "romantic_p0.6", "tweet_p0", "tweet_p0.6"]}

ADV_DEMO_TASKS = ["counterfactual", "spurious", "backdoor"]


with open("./data/toxicity_results.json") as file:
    toxicity_results = json.load(file)

with open("./data/ood_results.json", "r") as file:
    ood_results = json.load(file)

with open("./data/adv_demo.json") as file:
    adv_demo_results = json.load(file)

with open("./data/fairness_results.json") as file:
    fairness_results = json.load(file)

with open("./data/ethics_results.json") as file:                                                                                                                                                                    
    ethics_results = json.load(file)

with open("./data/stereotype_results.json") as file:
    stereotype_results = json.load(file)

with open("./data/privacy_results.json") as file:
    privacy_results = json.load(file)

models_to_analyze = [
    "hf/mosaicml/mpt-7b-chat",
    "hf/togethercomputer/RedPajama-INCITE-7B-Instruct",
    "hf/tiiuae/falcon-7b-instruct",
    "hf/lmsys/vicuna-7b-v1.3",
    "hf/chavinlo/alpaca-native",
    "hf/meta-llama/Llama-2-7b-chat-hf",
    "openai/gpt-3.5-turbo-0301",
    "openai/gpt-4-0314"
]


def radar_plot(aggregate_keys, all_keys, results, thetas, title, metric):
    # Extract performance values for each model across all benchmarks
    model_performance = {}

    for model in models_to_analyze:
        if model in results:
            benchmarks_data = results[model]
            if metric:
                model_performance[model] = [
                    np.nanmean([benchmarks_data[x][metric] if benchmarks_data[x][metric] is not None else np.nan
                                for x in all_keys if x.startswith(benchmark)]) for benchmark in aggregate_keys
                ]
            else:
                model_performance[model] = [
                    np.nanmean([list(benchmarks_data[x].values())[0] for x in all_keys if
                                x.startswith(benchmark)]) for benchmark in aggregate_keys
                ]
            if "counterfactual" in all_keys or "jailbreak" in all_keys or metric in ["Equalized Odds Difference", "Demographic Parity Difference", "emt", "category_overall_score"]:
                model_performance[model] = [x * 100 for x in model_performance[model]]
            if metric in ["asr", "Equalized Odds Difference", "Demographic Parity Difference", "emt", "brittleness"]:
                model_performance[model] = [100 - x for x in model_performance[model]]

    # Create radar chart with plotly
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.3,
        row_heights=[1, 1],
        specs=[[{"type": "polar"}], [{"type": "table"}]]
    )

    for i, (model, performance) in enumerate(model_performance.items()):
        color = DEFAULT_PLOTLY_COLORS[i % len(DEFAULT_PLOTLY_COLORS)]

        print(performance, aggregate_keys)
        fig.add_trace(
            go.Scatterpolar(
                r=performance + [performance[0]],
                theta=thetas + [thetas[0]],
                fill='toself',
                connectgaps=True,
                fillcolor=to_rgba(color, 0.1),
                name=model.split('/')[-1],  # Use the last part of the model name for clarity
            ),
            row=1, col=1
        )

    header_texts = ["Model"] + [x.replace("<br>", " ") for x in aggregate_keys]
    rows = [[x.split('/')[-1] for x in models_to_analyze]] + [[round(score[i], 2) for score in [model_performance[x] for x in models_to_analyze]] for i in range(len(aggregate_keys))]
    column_widths = [len(x) for x in header_texts]
    column_widths[0] *= 8 if "Toxicity" in title else 3

    fig.add_trace(
        go.Table(
            header=dict(values=header_texts, font=dict(size=15), align="left"),
            cells=dict(
                values=rows,
                align="left",
                font=dict(size=15),
                height=30
            ),
            columnwidth=column_widths
        ),
        row=2, col=1
    )

    fig.update_layout(
        height=1000,
        legend=dict(font=dict(size=20), orientation="h", xanchor="center", x=0.5, y=0.55),
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100],  # Assuming accuracy is a percentage between 0 and 100
                tickfont=dict(size=12)
            ),
            angularaxis=dict(tickfont=dict(size=20), type="category")
        ),
        showlegend=True,
        title=f"{title}"
    )

    return fig


def main_radar_plot(perspectives):
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.25,
        row_heights=[0.8, 0.7],
        specs=[[{"type": "polar"}], [{"type": "table"}]]
    )

    perspectives_shift = (perspectives[4:] + perspectives[:4])  # [::-1]

    for i, (model_name, score) in enumerate(MAIN_SCORES.items()):
        color = DEFAULT_PLOTLY_COLORS[i % len(DEFAULT_PLOTLY_COLORS)]

        score_shifted = score[4:] + score[:4]
        fig.add_trace(
            go.Scatterpolar(
                r=score_shifted + [score_shifted[0]],
                theta=perspectives_shift + [perspectives_shift[0]],
                connectgaps=True,
                fill='toself',
                fillcolor=to_rgba(color, 0.1),
                name=model_name,  # Use the last part of the model name for clarity
            ),
            row=1, col=1
        )

    header_texts = ["Model"] + perspectives
    rows = [
        list(MAIN_SCORES.keys()),  # Model Names
        *[[round(score[i], 2) for score in list(MAIN_SCORES.values())] for i in range(len(perspectives))]
    ]
    column_widths = [10] + [5] * len(perspectives)

    fig.add_trace(
        go.Table(
            header=dict(values=header_texts, font=dict(size=15), align="left"),
            cells=dict(
                values=rows,
                align="left",
                font=dict(size=15),
                height=30,
            ),
            columnwidth=column_widths,
        ),
        row=2, col=1
    )


    fig.update_layout(
        height=1200,
        legend=dict(font=dict(size=20), orientation="h", xanchor="center", x=0.5, y=0.55),
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100],  # Assuming accuracy is a percentage between 0 and 100
                tickfont=dict(size=12)
            ),
            angularaxis=dict(tickfont=dict(size=20), type="category", rotation=5)
        ),
        showlegend=True,
        title=dict(text="DecodingTrust Scores (Higher is Better) of GPT Models"),
    )


    return fig


def breakdown_plot(selected_perspective):
    if selected_perspective == "Main Figure":
        fig = main_radar_plot(PERSPECTIVES)
    elif selected_perspective == "Adversarial Robustness":
        fig = radar_plot(
            ADV_TASKS,
            ADV_TASKS,
            adv_results,
            ADV_TASKS,
            selected_perspective,
            "acc"
        )
    elif selected_perspective == "Out-of-Distribution Robustness":
        fig = radar_plot(
            ["knowledge_zeroshot", "style_zeroshot", "knowledge_fewshot", "style_fewshot"],
            list(ood_results[models_to_analyze[0]].keys()),
            ood_results,
            [
                "Ood Knowledge (Zero-shot)", "OoD Style (Zero-shot)", "OoD Knowledge (Few-shot)",
                "OoD Style (Few-shot)",
            ],
            selected_perspective,
            "score"
        )
    elif selected_perspective == "Robustness to Adversarial Demonstrations":
        fig = radar_plot(
            ["counterfactual", "spurious", "backdoor"],
            ["counterfactual", "spurious", "backdoor"],
            adv_demo_results,
            ["counterfactual", "spurious", "backdoor"],
            selected_perspective,
            ""
        )
    elif selected_perspective == "Fairness":
        fig = radar_plot(
            ["zero-shot", "few-shot-1", "few-shot-2"],
            ["zero-shot", "few-shot-1", "few-shot-2"],
            fairness_results,
            ["zero-shot", "few-shot setting given unfair context", "few-shot setting given fair context"],
            selected_perspective,
            "Equalized Odds Difference"
        )
    elif selected_perspective == "Machine Ethics":
        fig = radar_plot(
            ["jailbreak", "evasive", "zero-shot benchmark", "few-shot benchmark"],
            ["jailbreak", "evasive", "zero-shot benchmark", "few-shot benchmark"],
            ethics_results,
            ["jailbreaking prompts", "evasive sentence", "zero-shot benchmark", "few-shot benchmark"],
            selected_perspective,
            ""
        )
    elif selected_perspective == "Privacy":
        fig = radar_plot(
            ["enron", "PII", "understanding"],
            ["enron", "PII", "understanding"],
            privacy_results,
            ["enron", "PII", "understanding"],
            selected_perspective,
            "asr"
        )
    elif selected_perspective == "Toxicity":
        fig = radar_plot(
            [
                 "nontoxic-benign-sys",
                 "toxic-benign-sys",
                 "toxic-gpt3.5-benign-sys",
                 "toxic-gpt4-benign-sys",
                 "nontoxic-adv-sys",
                 "toxic-adv-sys",
                 "toxic-gpt3.5-adv-sys",
                 "toxic-gpt4-adv-sys",
            ],
            [
                 "nontoxic-benign-sys",
                 "toxic-benign-sys",
                 "toxic-gpt3.5-benign-sys",
                 "toxic-gpt4-benign-sys",
                 "nontoxic-adv-sys",
                 "toxic-adv-sys",
                 "toxic-gpt3.5-adv-sys",
                 "toxic-gpt4-adv-sys",
            ],
            toxicity_results,
            [
                 "nontoxic-benign-sys",
                 "toxic-benign-sys",
                 "toxic-gpt3.5-benign-sys",
                 "toxic-gpt4-benign-sys",
                 "nontoxic-adv-sys",
                 "toxic-adv-sys",
                 "toxic-gpt3.5-adv-sys",
                 "toxic-gpt4-adv-sys",
            ],
            selected_perspective,
            "emt"
        )
    elif selected_perspective == "Stereotype Bias":
        fig = radar_plot(
            ["benign", "untargeted", "targeted"],
            ["benign", "untargeted", "targeted"],
            stereotype_results,
            ["benign", "untargeted", "targeted"],
            selected_perspective,
            "category_overall_score"
        )

    else:
        raise ValueError(f"Choose perspective from {PERSPECTIVES}!")
    return fig


app = dash.Dash(__name__, external_stylesheets=['https://codepen.io/chriddyp/pen/bWLwgP.css'])
server = app.server


fig = main_radar_plot(PERSPECTIVES)
# Create the Dash app
app.layout = html.Div([
    dcc.Dropdown(
        id='perspective-dropdown',
        options=[{'label': perspective.replace("<br>", " "), 'value': perspective.replace("<br>", " ")} for perspective in ["Main Figure"] + PERSPECTIVES],
        value='Main Figure',  # Default value
        clearable=False
    ),
    dcc.Graph(
        id='radar-chart',
        figure=fig
    ),
    html.Div(id='output-click-data'),
    html.Div(id='tooltip-div', children=[
        html.Div('Hover over cells for more info', id='tooltip-content')
    ], style={'position': 'absolute', 'display': 'none', 'border': '1px solid black', 'background': 'white'})
])

@app.callback(
    Output('output-click-data', 'children'),
    [Input('radar-chart', 'clickData')]
)
def display_click_data(click_data):
    if click_data:
        point_data = click_data['points'][0]
        if "theta" in point_data:
            theta = point_data['theta']
            r = point_data['r']
            return f"You clicked on category {theta} with value {r}"
    return "Click on a data point!"

@app.callback(
    Output('radar-chart', 'figure'),
    Input('perspective-dropdown', 'value')
)
def update_figure(selected_perspective):
    return breakdown_plot(selected_perspective)

@app.callback(
    Output('tooltip-div', 'style'),
    Input('table', 'hoverData'),
)
def display_hover(hoverData):
    print(hoverData)
    if hoverData:
        # Get the point that is being hovered
        point = hoverData['points'][0]
        col = point['x']
        row = point['y']

        # Check if this point corresponds to a cell with long text
        if col == 1 and row == 1:  # Change these conditions based on your data
            return {'position': 'absolute', 'display': 'block', 'top': f"{point['yaxis.y']}px", 'left': f"{point['xaxis.x']}px", 'border': '1px solid black', 'background': 'white'}
    return {'display': 'none'}



if __name__ == "__main__":
    app.run(debug=True)

