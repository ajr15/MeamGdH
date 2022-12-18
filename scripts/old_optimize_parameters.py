# main script for parameter optimization
import json
import os
import numpy as np
from typing import List
from utils import run_lammps, write_potential_files, pmt_utils, parse_force_file, parse_lammps_output

def run_computations(parameters: dict, comp_dir: str, struct_dir: str, struct_names: List[str]):
    """Method to run all the required computations for an iteration"""
    # changing directory to comp dir
    org_dir = os.path.abspath(os.getcwd())
    struct_dir = os.path.abspath(struct_dir)
    os.chdir(comp_dir)
    # setting up 
    write_potential_files(parameters)
    # running all files
    for name in struct_names:
        struct = pmt_utils.read_structure_from_cif(os.path.join(struct_dir, name + ".cif"))
        run_lammps(struct, "{}.in".format(name), "{}.out".format(name), name)
    # returning to original directory
    os.chdir(org_dir)

def parse_results(comp_dir: str) -> dict:
    """Method to read all results from a computation directory. returns dictionary with 'energy', 'xyz' and 'forces' keys"""
    ajr = {}
    for fname in os.listdir(comp_dir):
        if fname.endswith(".out"):
            name = os.path.splitext(fname)[0]
            res = {}
            res["energy"] = parse_lammps_output(os.path.join(comp_dir, name + ".out"))
            xyz, forces = parse_force_file(os.path.join(comp_dir, name + ".forces"))
            res["xyz"] = xyz
            res["forces"] = forces
            ajr[name] = res
    return ajr

def calc_error(true_dict, pred_dict, weights=None):
    """Calculate prediction error"""
    if not weights:
        weights = {"energy": 0, "forces": 1}
    energy_err = 0
    force_err = 0
    atom_counter = 0
    name_counter = 0
    for name in true_dict:
        if name in pred_dict:
            name_counter += 1
            # convert pred energy from eV to Ry
            energy_err += abs(true_dict[name]["energy"] - pred_dict[name]["energy"] * 0.0735)
            true_fs = true_dict[name]["forces"]
            pred_fs = pred_dict[name]["forces"]
            for true_f, pred_f in zip(true_fs, pred_fs):
                # convert pred forces from eV/angstrom to Ry/au
                force_err += sum([abs(t - p / 1.8897 * 0.0735) for t, p in zip(true_f, pred_f)])
                atom_counter += 1
    return weights["energy"] * energy_err / name_counter + weights["forces"] * force_err / atom_counter

def target_func(parent_comp_dir: str, struct_dir: str, struct_names: List[str], alat=0, b1=0, b2=0, b3=0, t1=0, t2=0, t3=0, delr=0, cmax121=0, cmax122=0, cmax112=0, cmax221=0, iter=0) -> float:
    """Target optimization function - returns the prediction error for the given parameters"""
    # setting all parameters
    with open("parameters.json", "r") as f:
        parameters = json.load(f)
    # updating parameters
    parameters["pure"]["Gd"]["alat"] = alat
    parameters["pure"]["Gd"]["b1"] = b1
    parameters["pure"]["Gd"]["b2"] = b2
    parameters["pure"]["Gd"]["b3"] = b3
    parameters["pure"]["Gd"]["t1"] = t1
    parameters["pure"]["Gd"]["t2"] = t2
    parameters["pure"]["Gd"]["t3"] = t3
    parameters["interaction"]["delr"] = delr
    parameters["interaction"]["Cmax(1,2,1)"] = cmax121
    parameters["interaction"]["Cmax(1,2,2)"] = cmax122
    parameters["interaction"]["Cmax(1,1,2)"] = cmax112
    parameters["interaction"]["Cmax(2,2,1)"] = cmax221
    # creating computation directory
    comp_dir = os.path.join(parent_comp_dir, str(iter))
    if not os.path.isdir(comp_dir):
        os.mkdir(comp_dir)
    # saving parameter file in comp directory
    with open(os.path.join(comp_dir, "params.json"), "w") as f:
        json.dump(parameters, f)
    # running computation
    run_computations(parameters, comp_dir, struct_dir, struct_names)
    # reading results
    pred = parse_results(comp_dir)
    with open("data/data.json") as f:
        true = json.load(f)
    return calc_error(true, pred)

def particle_swarm_optimize(target_func, parameter_priors: dict, parent_comp_dir, struct_dir, struct_names, w, c1, c2, niter, nparticles):
    # init
    # randomly sample n particles in the space
    print("INITIALIZING")
    particles = []
    particle_velocities = []
    for _ in range(nparticles):
        d = {k: v.sample() for k, v in parameter_priors.items()}
        particles.append(d)
        particle_velocities.append({k: np.random.rand() for k in parameter_priors.keys()})
    # run target func on all particles
    errors = []
    for particle in particles:
        err = target_func(parent_comp_dir, struct_dir, struct_names, iter=0, **particle)
        errors.append(err)
    # finding the best particle (crucial for next steps)
    gbest = particles[errors.index(min(errors))]
    gbest_err = min(errors)
    pbest = particles[errors.index(min(errors))]
    # main run
    print("STARTING MAIN RUN")
    for i in range(1, niter + 1):
        print("ITERATION {} OUT OF {}".format(i, niter))
        errors = []
        for particle, velocities in zip(particles, particle_velocities):
            # randomizing process
            r1 = np.random.rand()
            r2 = np.random.rand()
            # updating velocities & particles
            for k, v in particle.items():
                velocities[k] = w * velocities[k] + c1 * r1 * (pbest[k] - v) + c2 * r2 * (gbest[k] - v)
                particle[k] = particle[k] + velocities[k]
            # running target func
            err = target_func(parent_comp_dir, struct_dir, struct_names, iter=i, **particle)
            errors.append(err)
        # update gbest and pbest
        pbest_err = min(errors)
        pbest = particles[errors.index(min(errors))]
        if pbest_err < gbest_err:
            gbest = particles[errors.index(min(errors))]
            gbest_err = pbest_err
        print("global best =", gbest_err)
        print("iteration best =", pbest_err)

class UniformDist:

    def __init__(self, low, high):
        self.low = low
        self.high = high

    def sample(self) -> float:
        return np.random.uniform(self.low, self.high)

if __name__ == "__main__":
    # initializing
    parameter_priors = {
        "alat": UniformDist(2, 6), 
        "b1": UniformDist(1, 5), 
        "b2": UniformDist(1, 5), 
        "b3": UniformDist(1, 5), 
        "t1": UniformDist(-2, 2), 
        "t2": UniformDist(-2, 2), 
        "t3": UniformDist(-2, 2), 
        "delr": UniformDist(0.5, 5), 
        "cmax121": UniformDist(0.5, 5), 
        "cmax122": UniformDist(0.5, 5), 
        "cmax112": UniformDist(0.5, 5), 
        "cmax221": UniformDist(0.5, 5)
    }
    # optimizing parameters
    struct_names = [os.path.splitext(fname)[0] for fname in os.listdir("structures") if fname.endswith(".cif")]
    particle_swarm_optimize(target_func, 
                                parameter_priors, 
                                "lammps", 
                                "structures", 
                                struct_names, 
                                w=0.5, 
                                c1=0.3, 
                                c2=0.3, 
                                niter=30, 
                                nparticles=15)
    import matplotlib.pyplot as plt
    # estimation of fit
    # reading true data
    with open("data/data.json") as f:
        true = json.load(f)
    # reading predicted data
    pred = parse_results("lammps/30")
    true_energies = []
    pred_energies = []
    true_forces = []
    pred_forces = []
    for k in true.keys():
        true_energies.append(true[k]["energy"])
        for vec in true[k]["forces"]:
            true_forces.extend(vec)
            # true_forces.append(vec[2])
        pred_energies.append(pred[k]["energy"])
        for vec in pred[k]["forces"]:
            # convering predicted forces to correct units
            pred_forces.extend([x / 1.8897 * 0.0735 for x in vec])
            # pred_forces.append([x / 1.8897 * 0.0735 for x in vec][2])
        print(k, true[k]["energy"], pred[k]["energy"])
    plt.figure()
    plt.title("Energies")
    plt.scatter(true_energies, pred_energies)
    plt.ylabel("pred")
    plt.xlabel("true")
    plt.figure()
    plt.title("Forces")
    plt.scatter(true_forces, pred_forces)
    plt.ylabel("pred")
    plt.xlabel("true")
    plt.show()
        
