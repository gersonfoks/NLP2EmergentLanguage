import argparse

import yaml

from experiment_utils import create_name, get_summary_results

parser = argparse.ArgumentParser(description='Run a grid defined in a given ')

parser.add_argument('--config', default="config/experiment_with_predictor_3_4.yml", required=False)

args = parser.parse_args()

with open(args.config) as f:
    config = yaml.load(f)


name = create_name(config)

results_summary = get_summary_results(name)

print(results_summary)
