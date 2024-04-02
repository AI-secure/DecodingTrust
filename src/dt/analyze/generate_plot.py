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