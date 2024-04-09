import os
import matplotlib.pyplot as plt
import json
from dt.analyze.generate_plot import breakdown_plot
import argparse


curr_path = os.path.abspath(__file__)
curr_dir = os.path.dirname(curr_path)



def get_scores(target_model, results_target, subfields):
    subscores = {}
    sub_types = ["benign", "adv1", "adv2"]
    harmfulness_results = {}
    for prompt_type in sub_types:
        harmfulness_results[prompt_type] = {}
        harmfulness_results[prompt_type][target_model] = {}
        for subfield in subfields:
            harmfulness_results[prompt_type][target_model][subfield] = results_target["breakdown_results"]["harmfulness"][target_model][prompt_type][subfield]['score']
        subscores["harmfulness_" + prompt_type] = harmfulness_results[prompt_type]
    return subscores

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

def generate_radar_plot(model, sub_scores, out_path):
    model_list = [model]
    sub_types = {"benign": "Harmfulness Benign", "adv1" : "Harmfulness Adv 1", "adv2" : "Harmfulness Adv 2"}
    for scenario in sub_types.keys():
        perspective = "harmfulness_" + scenario
        perspective_name = sub_types[scenario]
        subplot = breakdown_plot(sub_scores[perspective], list(sub_scores[perspective][model].keys()), model_list)
        perspective_name = perspective_name.replace(" ", "_")
        subplot.write_image(f"{out_path}/{perspective_name}_breakdown.png", width=1400, height=700)

def generate_plot(target_model, result_dir="./results", target_path="./src/dt/analyze/extra_plots/harmfulness/plots"):
    subfields = ['illegal_activity', 'harm_children', 'hate_harass_violence', 'malware',
            'physical_harm', 'economic_harm', 'fraudulent_deceptive', 'adult_content', 'political', 'privacy_violation',
            'financial_advice']
    if not os.path.exists(target_path):
        os.makedirs(target_path)
    with open(os.path.join(result_dir, "summary.json")) as file:
        results_target = json.load(file)
    subscores = get_scores(target_model, results_target, subfields)
    generate_radar_plot(target_model, subscores, target_path)
    generate_harmfulness_barchart(results_target, target_model, 'adv1', subfields, target_path)
    generate_harmfulness_barchart(results_target, target_model, 'adv2', subfields, target_path)
    generate_harmfulness_barchart(results_target, target_model, 'benign', subfields, target_path)
            

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default="vertexai/gemini-pro")
    args = parser.parse_args()

    generate_plot(args.model)