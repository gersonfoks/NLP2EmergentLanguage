import argparse
import matplotlib
from matplotlib import pyplot as plt
import yaml
import numpy as np

from experiment_utils import create_name, get_summary_results_filtered

parser = argparse.ArgumentParser(description='Run a grid defined in a given ')

configs = [
    "config/experiment_with_predictor_3_4_high_weight.yml",
    "config/experiment_with_predictor_3_4.yml",
    #"config/experiment_with_predictor_3_4_removed_classes.yml",
    "config/experiment_no_predictor_3_4.yml"
    #"config/experiment_with_predictor_3_4"
    #"config/experiment_no_predictor_3_4_removed_classes.yml"
]
configs = [
    "config/experiment_with_predictor_3_4_high_weight_removed_classes.yml",
    #"config/experiment_with_predictor_3_4.yml",
    "config/experiment_with_predictor_3_4_removed_classes.yml",
    #"config/experiment_no_predictor_3_4.yml"
    #"config/experiment_with_predictor_3_4"
    "config/experiment_no_predictor_3_4_removed_classes.yml"
]
# configs = [
#     "config/experiment_no_predictor_3_4.yml",
#     "config/experiment_with_predictor_3_4.yml",
#     "config/experiment_with_predictor_3_4_high_weight.yml"
# ]

groupnames = []
means = []
stds = []

for config_name in configs:

    with open(config_name) as f:
        config = yaml.load(f)

    print(config_name)

    name = create_name(config)
    #keep_out = set(["val accuracy predictor_epoch","msg_len","distinct symbols","bigram entropy","symbol entropy"])
    keep_out = set(["val accuracy predictor_epoch","msg_len","distinct symbols","bigram entropy","symbol entropy"])
    results_summary = get_summary_results_filtered(name)
    for key, value in results_summary.items():
        print(key + ":   " + str(value))
        print("HOI")
        if key not in keep_out:
            print("DOEI")
            means.append(value[0])
            stds.append(value[1])
            if key not in set(groupnames):
                print("ADD")
                groupnames.append(key)
    # if config_name=="config/experiment_no_predictor_3_4.yml":
    #     groupnames = results_summary.keys()
   
    print("")


#groupnames=["Bigram Entropy", "Distinct Symbols", "MSG Length", "Symbol Entropy", "Validation ACC"]
groupnames = ["accuracy on held out set"]

# ordering = [4,3,0,1,2]
# groupnames = [groupnames[i] for i in ordering]

a = len(groupnames)
bars1=np.arange(len(groupnames))
bars2=[i+0.2 for i in bars1]
bars3=[i+0.4 for i in bars1]
plt.style.use('ggplot')
print(len(groupnames))
print(len(means[:a]))
print(len(means[a:2*a]))
print(len(means[2*a:3*a]))
plt.bar(bars1,means[:a],0.2,yerr=stds[:a],capsize=4,label="predictability pressure ON-HIGH",color="red")
plt.bar(bars2,means[a:2*a],0.2,yerr=stds[a:2*a],capsize=4,label="predictability pressure ON-LOW",color="blue")
plt.bar(bars3,means[2*a:3*a],0.2,yerr=stds[2*a:3*a],capsize=4,label="predictability pressure OFF",color="green")
plt.xticks(bars1+0.2,groupnames,fontsize=24)
plt.yticks(fontsize=24)
plt.legend(fontsize=24)
plt.title("Comparison of baseline model with predictability pressure models for different pressure levels", fontsize=24)
plt.show()
