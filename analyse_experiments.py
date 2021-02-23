import argparse

import yaml

from experiment_utils import create_name, get_summary_results

parser = argparse.ArgumentParser(description='Run a grid defined in a given ')



configs = [
    "config/experiment_with_predictor_3_4.yml",
    "config/experiment_with_predictor_3_4_removed_classes.yml",
    "config/experiment_no_predictor_3_4.yml",
    "config/experiment_no_predictor_3_4_removed_classes.yml"
]


for config_name in configs:


    with open(config_name) as f:
        config = yaml.load(f)

    print(config_name)


    name = create_name(config)

    results_summary = get_summary_results(name)

    print(results_summary)
