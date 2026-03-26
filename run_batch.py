import argparse
import os
import pickle
import sys
import time
from datetime import timedelta

import matplotlib.pyplot as plt
import pandas as pd
from mesa.batchrunner import BatchRunner

from fire_evacuation.agent import Human
from fire_evacuation.model import FireEvacuation

DIR = os.path.dirname(os.path.realpath(__file__))
OUTPUT_DIR = DIR + "/output"

GRAPH_DPI = 100
GRAPH_WIDTH = 1920
GRAPH_HEIGHT = 1080


def merge_dataframes():
    directory = OUTPUT_DIR + "/batch_results/"
    previous_dataframe_files = [
        f
        for f in os.listdir(directory)
        if (os.path.isfile(os.path.join(directory, f)) and "dataframe_" in f)
    ]

    if previous_dataframe_files:
        dataframes = []
        print("Merging these dataframes:", previous_dataframe_files)
        for f in previous_dataframe_files:
            df = pickle.load(open(directory + f, "rb"))
            dataframes.append(df)
        return pd.concat(dataframes, ignore_index=True), len(dataframes)

    return None, None


parser = argparse.ArgumentParser()
parser.add_argument("runs", help="Number of repeat runs to do for each parameter setup.", type=int)
parser.add_argument("human_count", help="Number of humans in the simulation.", type=int)

try:
    args = parser.parse_args()
except Exception:
    parser.print_help()
    sys.exit(1)

runs = args.runs if args.runs else 1
human_count = args.human_count if args.human_count else 1

fixed_params = dict(
    floor_plan_file="floorplan_testing.txt",
    human_count=human_count,
    collaboration_percentage=50,
    fire_probability=0.8,
    visualise_vision=False,
    random_spawn=True,
    save_plots=False,
    replan_interval=3,
)

variable_params = {
    "guidance_profile": ["static", "dynamic_no_prediction", "dynamic_prediction"],
}


model_reporter = {
    "AverageEvacuationTime": lambda m: m.get_average_evacuation_time(),
    "SurvivalRate": lambda m: m.get_survival_rate(),
    "Deaths": lambda m: FireEvacuation.count_human_status(m, Human.Status.DEAD),
    "SmokeExposure": lambda m: m.get_average_smoke_exposure(),
    "PathChangeCount": lambda m: m.get_average_path_changes(),
    "ExitCongestion": lambda m: max(m.exit_congestion_map.values() or [0]),
}

print("Running batch test with %i runs for each parameter and %i human agents." % (runs, human_count))

batch_start = time.time()
for i in range(1, runs + 1):
    iteration_start = time.time()

    param_run = BatchRunner(
        FireEvacuation,
        variable_parameters=variable_params,
        fixed_parameters=fixed_params,
        model_reporters=model_reporter,
    )

    param_run.run_all()

    iteration_end = time.time()
    end_timestamp = time.strftime("%Y%m%d-%H%M%S")

    dataframe = param_run.get_model_vars_dataframe()

    def label_mode(row):
        return row["guidance_profile"]

    dataframe["guidance_variant"] = dataframe.apply(label_mode, axis=1)
    dataframe.to_pickle(path=OUTPUT_DIR + "/batch_results/dataframe_" + end_timestamp + ".pickle")

    elapsed = iteration_end - iteration_start
    print("Batch runner finished iteration %i. Took: %s" % (i, str(timedelta(seconds=elapsed))))

    dataframe, count = merge_dataframes()
    del dataframe["Run"]

    fig = plt.figure(figsize=(GRAPH_WIDTH / GRAPH_DPI, GRAPH_HEIGHT / GRAPH_DPI), dpi=GRAPH_DPI)
    plt.scatter(dataframe.guidance_variant, dataframe.SurvivalRate)
    fig.suptitle(
        "Guidance Mode vs Survival: " + str(human_count) + " Human Agents, " + str(count) + " Iterations",
        fontsize=20,
    )
    plt.xlabel("Guidance Variant", fontsize=14)
    plt.ylabel("Survival Rate (%)", fontsize=14)
    plt.ylim(0, 100)
    plt.savefig(OUTPUT_DIR + "/batch_graphs/batch_run_guidance_survival_" + end_timestamp + ".png", dpi=GRAPH_DPI)
    plt.close(fig)

    fig = plt.figure(figsize=(GRAPH_WIDTH / GRAPH_DPI, GRAPH_HEIGHT / GRAPH_DPI), dpi=GRAPH_DPI)
    ax = fig.gca()
    dataframe.boxplot(
        ax=ax,
        column="AverageEvacuationTime",
        by="guidance_variant",
        showmeans=True,
    )
    fig.suptitle(
        "Guidance Mode vs Evacuation Time: " + str(human_count) + " Human Agents, " + str(count) + " Iterations",
        fontsize=20,
    )
    plt.xlabel("Guidance Variant", fontsize=14)
    plt.ylabel("Average Evacuation Time", fontsize=14)
    plt.savefig(OUTPUT_DIR + "/batch_graphs/batch_run_guidance_time_" + end_timestamp + ".png", dpi=GRAPH_DPI)
    plt.close(fig)

batch_end = time.time()
elapsed = batch_end - batch_start
print("Batch runner finished all iterations. Took: %s" % str(timedelta(seconds=elapsed)))
