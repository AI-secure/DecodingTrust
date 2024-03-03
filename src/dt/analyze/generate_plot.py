from tqdm import tqdm
import time
import json
import numpy as np
import plotly.colors
from itertools import chain
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import matplotlib.pyplot as plt
import pandas as pd
import random
import argparse

DEFAULT_PLOTLY_COLORS = plotly.colors.DEFAULT_PLOTLY_COLORS



def to_rgba(rgb, alpha=1):
    return 'rgba' + rgb[3:][:-1] + f', {alpha})'

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


def radar_plot(aggregate_keys, all_keys, results, thetas, title, metric, selected_models=None):
    # Extract performance values for each model across all benchmarks
    model_performance = {}
    # print("selected_models", selected_models)
    if selected_models is None:
        selected_models = models_to_analyze
    for model in selected_models:
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
        vertical_spacing=0.2,
        row_heights=[1, 0.4],
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
    rows = [[x.split('/')[-1] for x in selected_models]] + [[round(score[i], 2) for score in [model_performance[x] for x in selected_models]] for i in range(len(aggregate_keys))]
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


def breakdown_plot(selected_perspective, selected_models=None):
    if selected_models is None:
        selected_models = models_to_analyze
    if selected_perspective == "Adversarial Robustness":
        fig = radar_plot(
            ["sst2", "qqp", "mnli"],
            ["sst2", "qqp", "mnli"],
            adv_results,
            ["sst2", "qqp", "mnli"],
            selected_perspective,
            "acc",
            selected_models
        )
    elif selected_perspective == "Out-of-Distribution Robustness":
        # print({model: ood_results[model] for model in selected_models})
        fig = radar_plot(
            ["knowledge_zeroshot", "style_zeroshot", "knowledge_fewshot", "style_fewshot"],
            list(ood_results[models_to_analyze[0]].keys()),
            ood_results,
            [
                "OoD Knowledge (Zero-shot)", "OoD Style (Zero-shot)", "OoD Knowledge (Few-shot)",
                "OoD Style (Few-shot)",
            ],
            selected_perspective,
            "score",
            selected_models
        )
    elif selected_perspective == "Robustness to Adversarial Demonstrations":
        fig = radar_plot(
            ["counterfactual", "spurious", "backdoor"],
            ["counterfactual", "spurious", "backdoor"],
            adv_demo_results,
            ["counterfactual", "spurious", "backdoor"],
            selected_perspective,
            "",
            selected_models
        )
    elif selected_perspective == "Fairness":
        fig = radar_plot(
            ["zero-shot", "few-shot-1", "few-shot-2"],
            ["zero-shot", "few-shot-1", "few-shot-2"],
            fairness_results,
            ["zero-shot", "few-shot setting given unfair context", "few-shot setting given fair context"],
            selected_perspective,
            "Equalized Odds Difference",
            selected_models
        )
    elif selected_perspective == "Machine Ethics":
        fig = radar_plot(
            ["jailbreak", "evasive", "zero-shot benchmark", "few-shot benchmark"],
            ["jailbreak", "evasive", "zero-shot benchmark", "few-shot benchmark"],
            ethics_results,
            ["jailbreaking prompts", "evasive sentence", "zero-shot benchmark", "few-shot benchmark"],
            selected_perspective,
            "",
            selected_models
        )
    elif selected_perspective == "Privacy":
        fig = radar_plot(
            ["enron", "PII", "understanding"],
            ["enron", "PII", "understanding"],
            privacy_results,
            ["enron", "PII", "understanding"],
            selected_perspective,
            "asr",
            selected_models
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
            "emt",
            selected_models
        )
    elif selected_perspective == "Stereotype Bias":
        fig = radar_plot(
            ["benign", "untargeted", "targeted"],
            ["benign", "untargeted", "targeted"],
            stereotype_results,
            ["benign", "untargeted", "targeted"],
            selected_perspective,
            "category_overall_score",
            selected_models
        )
    elif selected_perspective == "Harmfulness Benign":
        fig = radar_plot(
            list(list(harmfulness_results_benign.values())[0].keys()),
            list(list(harmfulness_results_benign.values())[0].keys()),
            harmfulness_results_benign,
            list(list(harmfulness_results_benign.values())[0].keys()),
            selected_perspective,
            "",
            selected_models
        )
    elif selected_perspective == "Harmfulness Adv 1":
        fig = radar_plot(
            list(list(harmfulness_results_adv1.values())[0].keys()),
            list(list(harmfulness_results_adv1.values())[0].keys()),
            harmfulness_results_adv1,
            list(list(harmfulness_results_adv1.values())[0].keys()),
            selected_perspective,
            "",
            selected_models
        )
    
    elif selected_perspective == "Harmfulness Adv 2":
        fig = radar_plot(
            list(list(harmfulness_results_adv2.values())[0].keys()),
            list(list(harmfulness_results_adv2.values())[0].keys()),
            harmfulness_results_adv2,
            list(list(harmfulness_results_adv2.values())[0].keys()),
            selected_perspective,
            "",
            selected_models
        )

    else:
        raise ValueError(f"Perspective is not included!")
    return fig

def update_subscores(target_model, subscores, main_scores):
    perspectives = []
    target_model = target_model.split('/')[-1]
    curr_main_scores = {}
    curr_main_scores[target_model] = {}
    for prespective in main_scores[target_model].keys():
        if prespective == "toxicity":
            global toxicity_results
            toxicity_results = subscores["toxicity"]
            perspectives.append("Toxicity")
            curr_main_scores[target_model]["Toxicity"] = main_scores[target_model]["toxicity"]
        elif prespective == "adv":
            global adv_results
            adv_results = subscores["adv"]
            perspectives.append("Adversarial Robustness")
            curr_main_scores[target_model]["Adversarial Robustness"] = main_scores[target_model]["adv"]
        elif prespective == "ood":
            global ood_results
            ood_results = subscores["ood"]
            perspectives.append("Out-of-Distribution Robustness")
            curr_main_scores[target_model]["Out-of-Distribution Robustness"] = main_scores[target_model]["ood"]
        elif prespective == "icl":
            global adv_demo_results
            adv_demo_results = subscores["icl"]
            perspectives.append("Robustness to Adversarial Demonstrations")
            curr_main_scores[target_model]["Robustness to Adversarial Demonstrations"] = main_scores[target_model]["icl"]
        elif prespective == "privacy":
            global privacy_results
            privacy_results = subscores["privacy"]
            perspectives.append("Privacy")
            curr_main_scores[target_model]["Privacy"] = main_scores[target_model]["privacy"]
        elif prespective == "moral":
            global ethics_results
            ethics_results = subscores["moral"]
            perspectives.append("Machine Ethics")
            curr_main_scores[target_model]["Machine Ethics"] = main_scores[target_model]["moral"]
        elif prespective == "fairness":
            global fairness_results
            fairness_results = subscores["fairness"]
            perspectives.append("Fairness")
            curr_main_scores[target_model]["Fairness"] = main_scores[target_model]["fairness"]
        elif prespective == "harmfulness":
            global harmfulness_results_adv1
            harmfulness_results_adv1 = subscores["harmfulness_adv1"]
            perspectives.append("Harmfulness Adv 1")
            
            global harmfulness_results_adv2
            harmfulness_results_adv2 = subscores["harmfulness_adv2"]
            perspectives.append("Harmfulness Adv 2")
            
            global harmfulness_results_benign
            harmfulness_results_benign = subscores["harmfulness_benign"]
            perspectives.append("Harmfulness Benign")
            curr_main_scores[target_model]["Harmfulness"] = main_scores[target_model]["harmfulness"]
    return perspectives, curr_main_scores


def generate_harmfulness_barchart(json_data, model, sub_key, risk_categories, out_path="plots"):
    """
    Parameters:
    - json_data: The loaded JSON data.
    - model: The model key in the JSON data.
    - sub_key: One of ["benign", "adv1", "adv2"].
    - risk_categories: Ordered list of risk categories.
    - out_path: Path to save the image.
    """
    # Extract data for the specified sub_key
    sub_key_data = json_data['breakdown_results']['harmfulness'][model][sub_key]
    
    # Prepare data for plotting
    categories = []
    values = []
    colors = []
    color_map = plt.cm.get_cmap('tab20', len(risk_categories))
    
    # Counter for annotations
    counter = 1
    
    for risk_category in risk_categories:
        if risk_category in sub_key_data:
            details = sub_key_data[risk_category]
            for subcategory, value in details['subcatgeories'].items():
                categories.append(f"{counter}. {subcategory}")  # Numbering included in subcategory names
                values.append(value)
                colors.append(color_map(risk_categories.index(risk_category)))
                counter += 1
    
    # Plotting with adjusted annotations and grid enabled
    fig, ax = plt.subplots(figsize=(24, 10))  # Increased width for better layout
    bars = ax.bar(range(len(values)), values, color=colors)
    
    # Enable grid
    ax.grid(True, which='both', axis='y', linestyle='--', linewidth=0.5)
    
    # Update y-axis label
    ax.set_ylabel('Jailbreak Rate', fontsize=15)
    ax.set_xticks(range(len(categories)))
    
    # Adjusted subcategory names
    ax.set_xticklabels(categories, rotation=45, fontsize=11, ha="right", va="top", y=-0.03)
    
    # Re-adding colored numerical annotations below the x-axis
    for i, (bar, color) in enumerate(zip(bars, colors)):
        ax.annotate(f"{i + 1}", xy=(i, 0), xycoords=('data', 'axes fraction'),
                    xytext=(0, -4), textcoords='offset points', ha='center', va='top', fontsize=10, color=color, fontweight='bold')
    
    # Adjusting legend to a single row at the top
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=color_map(i), label=cat) for i, cat in enumerate(risk_categories)]
    ax.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 1.24),
              ncol=len(legend_elements)/2+1, fontsize=15)
    
    plt.tight_layout()
    
    # Save the figure
    fig.savefig(f"{out_path}/harmfulness_{sub_key}_barchart.png", dpi=300)

def generate_plot(model, main_scores, sub_scores, out_path="plots"):
    global models_to_analyze
    models_to_analyze.append(model)
    perspectives, curr_main_scores = update_subscores(model, sub_scores, main_scores)
    for idx, perspective in enumerate(perspectives):
        if perspective == "Harmfulness Adv 1" or perspective == "Harmfulness Benign" or perspective ==  "Harmfulness Adv 2":
            model_list = [model]
        else:
            model_list = [model, "openai/gpt-4-0314"]
        subplot = breakdown_plot(perspective, model_list)
        prespective = perspective.replace(" ", "_")
        subplot.write_image(f"{out_path}/{prespective}_breakdown.png", width=1400, height=700)
    plot = main_radar_plot(curr_main_scores, [model])
    plot.write_image(f"{out_path}/main.png", width=1400, height=700)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="hf/meta-llama/Llama-2-7b-chat-hf")
    args = parser.parse_args()
    os.makedirs("./plots", exist_ok=True)
    generate_plot(args.model)