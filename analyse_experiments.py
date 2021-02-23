import argparse
import matplotlib
from matplotlib import pyplot as plt
import yaml
import numpy as np

from experiment_utils import create_name, get_summary_results

parser = argparse.ArgumentParser(description='Run a grid defined in a given ')

configs = [
    "config/experiment_with_predictor_3_4.yml",
    #"config/experiment_with_predictor_3_4_removed_classes.yml",
    "config/experiment_no_predictor_3_4.yml"
    #"config/experiment_no_predictor_3_4_removed_classes.yml"
]

groupnames = []
means = []
stds = []

for config_name in configs:

    with open(config_name) as f:
        config = yaml.load(f)

    print(config_name)

    name = create_name(config)

    results_summary = get_summary_results(name)
    for key, value in results_summary.items():
        print(key + ":   " + str(value))
        if key != "val accuracy predictor_epoch":
            means.append(value[0])
            stds.append(value[1])
    groupnames = results_summary.keys()
   
    print("")

bars1=np.arange(len(results_summary.keys()))
bars2=[i+0.4 for i in bars1]
plt.style.use('ggplot')
plt.bar(bars1,means[:5],0.4,yerr=stds[:5],capsize=4,label="predictability pressure ON")
plt.bar(bars2,means[5:],0.4,yerr=stds[5:],capsize=4,label="predictability pressure OFF")
plt.xticks(bars1+0.2,groupnames)
# plt.bar(results_summary.keys(),[meanstd[0] for meanstd in results_summary.values()],yerr=[meanstd[1] for meanstd in results_summary.values()],w=0.4)
# plt.bar(results_summary.keys(),[meanstd[0] for meanstd in results_summary.values()],yerr=[meanstd[1] for meanstd in results_summary.values()],w=0.4)
plt.legend()
plt.show()
