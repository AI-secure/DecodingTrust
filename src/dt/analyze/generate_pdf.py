from jinja2 import Environment
import jinja2
import os
import numpy as np
import json
from dt.analyze.generate_plot import generate_plot
import shutil
import argparse
import importlib
import yaml


PERSPECTIVE_CONVERSION = {"adv_demonstration" : "icl", "advglue" : "adv", "machine_ethics" : "moral", "adv-glue-plus-plus" : "adv"}
FAILURE_MODULE_PATH = "dt.analyze.failure_extraction"
curr_path = os.path.abspath(__file__)
curr_dir = os.path.dirname(curr_path)


def replace_latex_special_chars(text):
    latex_special_char = {
        "#": "\\#",
        "$": "\\$",
        "%": "\\%",
        "&": "\\&",
        "^": "\\^",
        "_": "\\_",
        "|": "\\|",
        "~": "\\~",
        "{": "\\{",
        "}": "\\}"
    }
    return ''.join(latex_special_char.get(c, c) for c in text)

def check_risk_quantile(results, target):
    high_risk = results[0]
    low_risk = results[1]
    if target <= high_risk:
        return "high"
    elif target >= low_risk:
        return "low"
    else:
        return "moderate"

def add_configs(perspectives):
    config_dict = {}
    for perspective in perspectives:
        with open(f"{curr_dir}/configs/{perspective}.yaml", "r") as file:
            config_dict[perspective] = yaml.safe_load(file)
    return config_dict

def add_target_model(target_model, perspectives, results_target, config_dicts):
    main_scores = {}
    model_name = target_model.split("/")[-1]
    main_scores[model_name] = {}
    subscores = {}
    for perspective in perspectives:
        if perspective == "adv":
            main_scores[model_name][perspective] = results_target["aggregated_results"]["adv-glue-plus-plus"]["score"][target_model]
            raw_results = results_target["breakdown_results"]["adv-glue-plus-plus"][target_model]
        elif perspective == "moral":
            main_scores[model_name][perspective] = results_target["aggregated_results"]["machine_ethics"]["agg_score"][target_model]
            raw_results = results_target["breakdown_results"]["machine_ethics"][target_model]
        elif perspective == "icl":
            main_scores[model_name][perspective] = results_target["aggregated_results"]["adv_demonstration"]["score"][target_model]
            raw_results = results_target["breakdown_results"]["adv_demonstration"][target_model]
        elif perspective == "privacy":
            main_scores[model_name][perspective] = results_target["aggregated_results"]["privacy"]["privacy_score"][target_model]
            raw_results = results_target["breakdown_results"]["privacy"][target_model]
        elif perspective == "harmfulness":
            sub_types = ["benign", "adv1", "adv2"]
            main_scores[model_name][perspective] = min([results_target["aggregated_results"]["harmfulness"]["score"][target_model][sub_type] for sub_type in sub_types])
            raw_results = results_target["breakdown_results"]["harmfulness"][target_model]
        else:
            main_scores[model_name][perspective] = results_target["aggregated_results"][perspective]["score"][target_model]
            raw_results = results_target["breakdown_results"][perspective][target_model]
        subscores[perspective] = {}
        subscores[perspective][target_model] = {}
        raw_scores = raw_results
        for scenario, scenario_names in config_dicts[perspective]["subfields"].items():
            subscenario_scores = []
            for name in scenario_names:
                if perspective == "harmfulness":
                    raw_scores = raw_results[scenario]
                if perspective == "ood":
                    raw_scores = raw_results["subscores"]
                for subscenario_name in raw_scores.keys():
                    if perspective == "toxicity":
                        subscenario = subscenario_name.split("/")[-1]
                    else:
                        subscenario = subscenario_name
                    if name.lower() == subscenario.lower(): 
                        if config_dicts[perspective]["metric"]["name"] is not None:
                            subscenario_score = raw_scores[subscenario_name][config_dicts[perspective]["metric"]["name"]]
                        else:
                            subscenario_score = raw_scores[subscenario_name]
                            if isinstance(subscenario_score, dict):
                                subscenario_score = list(subscenario_score.values())[0]
                        if config_dicts[perspective]["metric"]["scale"] == True:
                            subscenario_score = subscenario_score * 100
                        if config_dicts[perspective]["metric"]["minus"] == True:
                            subscenario_score = 100 - subscenario_score
                        subscenario_scores.append(subscenario_score)
            subscores[perspective][target_model][scenario] = np.mean(subscenario_scores) 
    return main_scores, subscores

def generate_pdf(target_model, perspectives="toxicity,advglue,ood,adv_demonstration,privacy,machine_ethics,fairness,harmfulness", result_dir="./results", quantile=0.3):
    target_path = os.path.join(curr_dir, f"reports_latex/{target_model}/report")
    perspectives = perspectives.split(",")
    if os.path.exists(target_path):
        shutil.rmtree(target_path)
    shutil.copytree(os.path.join(curr_dir, "report_template"), target_path)

    with open(os.path.join(result_dir, "summary.json")) as file:
        results_target = json.load(file)
    # Only generate avaiable perspectives
    available_perspectives = [perspective for perspective in results_target["breakdown_results"].keys() if target_model in results_target["breakdown_results"][perspective].keys()]
    available_perspectives = [PERSPECTIVE_CONVERSION[perspective] if perspective in PERSPECTIVE_CONVERSION.keys() else perspective for perspective in available_perspectives]
    perspectives = [PERSPECTIVE_CONVERSION[perspective] if perspective in PERSPECTIVE_CONVERSION.keys() else perspective for perspective in perspectives]
    perspectives = [perspective for perspective in perspectives if perspective in available_perspectives]
    config_dicts = add_configs(perspectives)
    main_scores, subscores = add_target_model(target_model, perspectives, results_target, config_dicts)
    generate_plot(target_model, main_scores, subscores, config_dicts, os.path.join(target_path, "Images"))

    eval_dicts = {"perspectives": []}
    # Add main results to DT-report
    for scenario in perspectives:
        scenario = scenario.lower()
        eval_dict = {}
        main_threshold = config_dicts[scenario]["threshold"]
        main_results = main_scores[target_model.split("/")[-1]][scenario]
        eval_dict["level"] = check_risk_quantile(main_threshold, main_results)
        eval_dict["perspective_name"] = scenario
        eval_dict["display_name"] = config_dicts[scenario]["name"]
        eval_dict["subfields"] = []
        for category in config_dicts[scenario]["categories"].keys():
            subfield_dict = {}
            scores = []
            threshold = config_dicts[scenario]["categories"][category]["threshold"]
            for subfield in config_dicts[scenario]["categories"][category]["subfields"]:
                scores.append(subscores[scenario][target_model][subfield])
            scores = np.mean(scores)
            subfield_dict["level"] = check_risk_quantile(threshold, scores)
            subfield_dict["name"] = category.lower()
            eval_dict["subfields"].append(subfield_dict)
        if config_dicts[scenario]["additional_plots"] is not None:
            for image_path in config_dicts[scenario]["additional_plots"]:
                if os.path.exist(os.path.join(curr_dir, image_path)):
                    shutil.copy(os.path.join(curr_dir, image_path), os.path.join(target_path, "Images"))
                else:
                    ValueError(f"Path {image_path} is not found. Please generate the plots first.")
        eval_dicts["perspectives"].append(eval_dict)

    env = Environment(
        loader=jinja2.FileSystemLoader(target_path),
        block_start_string='<BLOCK>',
        block_end_string='</BLOCK>',
        variable_start_string='<VAR>',
        variable_end_string='</VAR>',
    )

    # Filling context
    template = env.get_template("context.tex")
    output = template.render(model=os.path.basename(target_model))
    with open(os.path.join(target_path, "context.tex"), "w") as f:
        f.write(output)

    # Filling main results
    template = env.get_template("Content/main_results.tex")
    output = template.render(**eval_dicts)
    with open(os.path.join(target_path, "Content/main_results.tex"), "w") as f:
        f.write(output)

    # Filling scenario results
    template = env.get_template("Content/scenario_results.tex")
    output = template.render(**eval_dicts)
    with open(os.path.join(target_path, "Content/scenario_results.tex"), "w") as f:
        f.write(output)

    # Filling scenario overview
    template = env.get_template("Content/risk_analysis.tex")
    output = template.render(**eval_dicts)
    with open(os.path.join(target_path, "Content/risk_analysis.tex"), "w") as f:
        f.write(output)

    recommendations = []
    for perspective in perspectives:
        if config_dicts[perspective]["recommendations"]:
            recommendations.append(perspective)
    if len(recommendations) > 0:
        eval_dicts["recommendations"] = recommendations
    # Filling recommendations
    template = env.get_template("Content/recommendations.tex")
    output = template.render(**eval_dicts)
    with open(os.path.join(target_path, "Content/recommendations.tex"), "w") as f:
        f.write(output)

    # Retriving all detailed overview failure examples
    for perspective in perspectives:
        for name in config_dicts[perspective]["categories"].keys():
            if config_dicts[perspective]["categories"][name]["failure_example"]:
                try:
                    module = importlib.import_module(f"{FAILURE_MODULE_PATH}.{perspective}_failure")
                    fn_name = f"extract_{perspective}_examples"
                    extract_fn = getattr(module, fn_name)
                except:
                    ValueError(f"Failure extraction function for perspective {perspective} is not founded")
                example_dict = {}
                filename = f"{name}_example.tex"
                subfield = np.random.choice(config_dicts[perspective]["categories"][name]["subfields"])
                examples = extract_fn(target_model, subfield, result_dir)
                try:
                    example_dict["example"] = {}
                    example_dict["example"]["input"] = replace_latex_special_chars(examples[0]["Query"])
                    example_dict["example"]["output"] = replace_latex_special_chars(examples[0]["Outputs"])
                except:
                    example_dict = {}

                template = env.get_template(f"Content/{perspective}/{filename}")
                output = template.render(**example_dict)

                with open(os.path.join(target_path, f"Content/{perspective}/{filename}"), "w") as f:
                    f.write(output)

    report_saving_dir = os.path.join(curr_dir, "reports")
    if not os.path.exists(report_saving_dir):
        os.makedirs(report_saving_dir)
    report_saving_path = os.path.join(report_saving_dir, f'{os.path.basename(target_model)}.pdf')

    os.system(f"cd {target_path} && tar -czvf {os.path.basename(target_model)}.tar.gz ./ ")
    os.system(f"cd {target_path} && curl -F 'files[]=@{os.path.basename(target_model)}.tar.gz' -o {report_saving_path} 'https://texlive2020.latexonline.cc/data?command=pdflatex&target=main.tex'")

            

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default="vertexai/gemini-pro", required=True)
    parser.add_argument('--quantile', type=float, default=0.3)
    parser.add_argument('--result_dir', type=str, default="./results")
    parser.add_argument('--perspectives', type=str, default="toxicity,advglue,ood,adv_demonstration,privacy,machine_ethics,fairness,harmfulness")
    args = parser.parse_args()

    generate_pdf(args.model, args.perspectives, args.result_dir, args.quantile)