# main script for parameter optimization
import json
import os
from skopt import gp_minimize
import sys; sys.path.append("/home/shaharpit/MeamGdH")
from src.simulators import Simulator
from src.optimizers import PsoOptimizer
from src.Variable import Variable
from src.potentials import EmbeddedAtomElementPotential, EmbeddedAtomInteractionPotential
from src.losses import AveragePairLoss
sys.path.append("/home/shaharpit/Personal/TorinaX")
from torinax.utils import pymatgen as pmt_utils

def main():
    # reading potentials
    with open("/home/shaharpit/MeamGdH/data/parameters.json", "r") as f:
        parameters = json.load(f)
    gd_potential = EmbeddedAtomElementPotential(name="Gd", symbol="Gd", **parameters["pure"]["Gd"])
    h_potential = EmbeddedAtomElementPotential(name="H", symbol="H", **parameters["pure"]["H"])
    gdh_potential = EmbeddedAtomInteractionPotential(name="GdH", **parameters["interaction"])
    # defining simulator
    simulator = Simulator([gd_potential, h_potential, gdh_potential])
    opt_dir = "/home/shaharpit/MeamGdH/opt"
    # setting sructure directory
    structs_dir = "/home/shaharpit/MeamGdH/dft/single_point/comps/cifs"
    # setting loss function
    loss = AveragePairLoss(energy_weight=1, force_weight=1, pair_loss_function="mae") # more or less equal importance to force and energy prediction
    # reading data
    with open("../data/data.json", "r") as f:
        data_dict = json.load(f)
    true_energies = data_dict["energy"]
    true_forces = data_dict["forces"]
    # defining target function for optimizer
    def target_func(input_vector):
        w, c1, c2 = input_vector
        # finding parent dir
        parent_dir = os.path.join(opt_dir, str(len(os.listdir(opt_dir))))
        os.mkdir(parent_dir)
        variables = [
            Variable("delr", gdh_potential, 0, 1, 0.1),
            Variable("Cmax221", gdh_potential, 2, 4, 2.8),
            Variable("Cmax112", gdh_potential, 2, 4, 2.8),
            Variable("Cmax121", gdh_potential, 2, 4, 2.8),
            Variable("Cmax122", gdh_potential, 2, 4, 2.8)
        ]
        nparticles = 10
        niters = 10
        # reading structures
        structs = {}
        for cif in os.listdir(structs_dir):
            name = cif.split(".")[0]
            if not cif.split(".")[1] == "cif":
                name += "." + cif.split(".")[1]
            structs[name] = pmt_utils.read_structure_from_cif(os.path.join(structs_dir, cif))
        optimizer = PsoOptimizer(simulator, variables, loss, parent_dir, nparticles, niters, w, c1, c2)
        optimizer.fit(structs, true_energies, true_forces)
        # saving optimizer's configuration for future reference
        with open(os.path.join(parent_dir, "optimizer.json"), "w") as f:
            d = {"w": w, "c1": c1, "c2": c2, "history": optimizer.history, "pbests": optimizer.pbest_history}
            json.dump(d, f, indent=2)
        return optimizer.history[-1]
    # running optimization
    results = gp_minimize(
        func=target_func,
        dimensions=[
            (1e-3, 1e1), # range for w
            (1e-3, 1e1), # range for c1
            (1e-3, 1e1), # range for c2
        ],
        n_calls=10
    )
    print("ALL DONE !!")

if __name__ == "__main__":
    main()
