import plotly.colors
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import matplotlib.pyplot as plt
import argparse

DEFAULT_PLOTLY_COLORS = plotly.colors.DEFAULT_PLOTLY_COLORS


def to_rgba(rgb, alpha=1):
    return 'rgba' + rgb[3:][:-1] + f', {alpha})'

def radar_plot(results, thetas, selected_models):
    # Extract performance values for each model across all benchmarks
    model_performance = {}
    for model in selected_models:
        if model in results:
            benchmarks_data = results[model]
            model_performance[model] = [benchmarks_data[subfield] for subfield in benchmarks_data.keys()]

    # Create radar chart with plotly
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.2,
        row_heights=[1, 0.4],
        specs=[[{"type": "polar"}], [{"type": "table"}]]
    )

    for i, (model, performance) in enumerate(model_performance.items()):
        color = DEFAULT_PLOTLY_COLORS[i % len(DEFAULT_PLOTLY_COLORS)]

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

    header_texts = ["Model"] + [x.replace("<br>", " ") for x in thetas]
    rows = [[x.split('/')[-1] for x in selected_models]] + [[round(score[i], 2) for score in [model_performance[x] for x in selected_models]] for i in range(len(thetas))]
    column_widths = [len(x) for x in header_texts]
    column_widths[0] *= len(thetas)

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
        legend=dict(font=dict(size=20), orientation="h", xanchor="center", x=0.5, y=0.35),
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100],  # Assuming accuracy is a percentage between 0 and 100
                tickfont=dict(size=12)
            ),
            angularaxis=dict(tickfont=dict(size=20), type="category")
        ),
        showlegend=True,
        # title=f"{title}"
    )

    return fig


def main_radar_plot(main_scores, selected_models):
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.2,
        row_heights=[1.0, 0.5],
        specs=[[{"type": "polar"}], [{"type": "table"}]]
    )
    model_scores = {}
    for model in selected_models:
        model_name = os.path.basename(model)
        model_scores[model_name] = main_scores[model_name]
    perspectives = list(model_scores[os.path.basename(selected_models[0])].keys())
    perspectives_shift = perspectives
    for i, model_name in enumerate(model_scores.keys()):
        color = DEFAULT_PLOTLY_COLORS[i % len(DEFAULT_PLOTLY_COLORS)]
        score_shifted = list(model_scores[model_name].values())
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
        list(model_scores.keys()),  # Model Names
        *[[round(score[perspective], 2) for score in list(model_scores.values())] for perspective in perspectives]
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
        legend=dict(font=dict(size=20), orientation="h", xanchor="center", x=0.5, y=0.35),
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100],  # Assuming accuracy is a percentage between 0 and 100
                tickfont=dict(size=12)
            ),
            angularaxis=dict(tickfont=dict(size=20), type="category", rotation=5)
        ),
        showlegend=True,
        title=dict(text="DecodingTrust Scores (Higher is Better)"),
    )
    return fig


def breakdown_plot(scenario_results, subfields, selected_models):
    fig = radar_plot(scenario_results, subfields, selected_models)
    return fig

def update_subscores(target_model, main_scores, config_dicts):
    perspectives = []
    target_model = target_model.split('/')[-1]
    curr_main_scores = {}
    curr_main_scores[target_model] = {}
    for perspective in main_scores[target_model].keys():
        curr_main_scores[target_model][config_dicts[perspective]["name"]] = main_scores[target_model][perspective]
        perspectives.append(config_dicts[perspective]["name"])
    return curr_main_scores

def generate_plot(model, main_scores, sub_scores, config_dict, out_path="plots"):
    curr_main_scores = update_subscores(model,  main_scores, config_dict)
    for idx, perspective in enumerate(config_dict.keys()):
        if config_dict[perspective]["sub_plot"] == False:
            continue
        if "openai/gpt-4-0314" not in sub_scores[perspective].keys():
            model_list = [model]
        else:
            model_list = [model, "openai/gpt-4-0314"]
        subplot = breakdown_plot(sub_scores[perspective], list(sub_scores[perspective][model].keys()), model_list)
        perspective_name = config_dict[perspective]["name"].replace(" ", "_")
        subplot.write_image(f"{out_path}/{perspective_name}_breakdown.png", width=1400, height=700)
    plot = main_radar_plot(curr_main_scores, [model])
    plot.write_image(f"{out_path}/main.png", width=1400, height=700)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="hf/meta-llama/Llama-2-7b-chat-hf")
    args = parser.parse_args()
    os.makedirs("./plots", exist_ok=True)
    generate_plot(args.model)