import json
import os
from matplotlib import pyplot as plt
import sys; sys.path.append("/home/shaharpit/MeamGdH")
from src.potentials import EmbeddedAtomElementPotential, EmbeddedAtomInteractionPotential
from src.simulators import Simulator


def fit_history_plot(opt_dir: str, image_path: str):
    # collecting data
    history = []
    for iter_dir in os.listdir(opt_dir):
        with open(os.path.join(opt_dir, iter_dir, "optimizer.json"), "r") as f:
            history.extend(json.load(f)["history"])
    # plotting
    plt.plot(range(len(history)), history)
    plt.savefig(image_path)

def prediction_fit_plot(pred_dir: str, property: str, image_path: str):
    """Plot of predicted vs. true property (on training set)"""
    # loading simulator
    with open("/home/shaharpit/MeamGdH/data/parameters.json", "r") as f:
        parameters = json.load(f)
    gd_potential = EmbeddedAtomElementPotential(name="Gd", symbol="Gd", **parameters["pure"]["Gd"])
    h_potential = EmbeddedAtomElementPotential(name="H", symbol="H", **parameters["pure"]["H"])
    gdh_potential = EmbeddedAtomInteractionPotential(name="GdH", **parameters["interaction"])
    simulator = Simulator([gd_potential, h_potential, gdh_potential])
    # reading true and calculated properties
    with open("../data/data.json", "r") as f:
        data_dict = json.load(f)
    if property == "energy":
        true = data_dict["energy"]
        pred = simulator.get_relative_energies(pred_dir)
    elif property == "force":
        true = data_dict["forces"]
        pred = simulator.get_forces(pred_dir)
    else:
        ValueError("Unknown property {}. allowed values are energy and force".format(property))
    # making plotting vectors
    x = []
    y = []
    for key in true.keys():
        if key in pred:
            print(key, round(true[key], 2), round(pred[key], 2))
            x.append(true[key])
            y.append(pred[key])
            # for pred_fs, true_fs in zip(pred[key], true[key]): 
            #     x.extend(true_fs)
            #     y.extend(pred_fs)
    # plotting
    plt.scatter(x, y)
    plt.savefig(image_path)



if __name__ == "__main__":
    # plt.figure()
    # fit_history_plot("../opt", "../results/fit_history.png")
    plt.figure()
    prediction_fit_plot("../opt/1/10/5", "energy", "../results/energy_fit_plot.png")
    # prediction_fit_plot("../opt/9/10/9", "force", "../results/force_fit_plot.png")