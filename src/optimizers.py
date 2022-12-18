from typing import Dict, List
import numpy as np
import os
import sys; sys.path.append("/home/shaharpit/Personal/TorinaX")
from torinax.base import Structure
from .simulators import Simulator
from .Variable import Variable
from .losses import AveragePairLoss
from .utils import print_msg

class PsoOptimizer:

    """Wrapper around the sklearn.BaseEstimator for optimization of the potential function. Allows for bayesian optimization on hyperparameters of optimizer"""

    def __init__(self, 
                    simulator: Simulator, 
                    variables: List[Variable], 
                    loss: AveragePairLoss,
                    comp_dir: str,
                    nparticles: int,
                    niters: int,
                    w: float,
                    c1: float,
                    c2: float):
        self.simulator = simulator
        self.variables = variables
        self.loss = loss
        self.comp_dir = comp_dir
        self.nparticles = nparticles
        self.niters = niters
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.history = []
        self.pbest_history = []

    def target(self, structs: Dict[str, Structure], variables: List[Variable], parent_dir: str, true_energies: Dict[str, float], true_forces: Dict[str, np.array]) -> float:
        """The target function of the simulator"""
        self.simulator.run(structs, variables, parent_dir)
        pred_energies = self.simulator.get_relative_energies(parent_dir)
        pred_forces = self.simulator.get_forces(parent_dir)
        return self.loss.evaluate(true_energies, true_forces, pred_energies, pred_forces)

    def fit(self, structs: Dict[str, Structure], true_energies: Dict[str, float], true_forces: Dict[str, np.array], verbose: int=1):
        """Optimize simulator variables on given structures and data"""
        # initializing N particles with random locations and velocities
        if verbose > 0:
            print_msg("Starting PSO Optimization")
            print("files of this computation are stored at:", self.comp_dir)
            print("initializing...")
        particles = []
        particle_velocities = []
        for _ in range(self.nparticles):
            vars = []
            for variable in self.variables:
                kwds = variable.__dict__
                kwds["value"] = np.random.uniform(variable.min_value, variable.max_value)
                vars.append(Variable(**kwds))
            particles.append(vars)
            particle_velocities.append([np.random.rand() * min([var.value - var.min_value, variable.max_value - var.value]) for var in self.variables])
        # run target func on all particles
        errors = []
        iter_dir = os.path.join(self.comp_dir, "0")
        if not os.path.isdir(iter_dir):
            os.mkdir(iter_dir)
        if verbose > 0:
            print_msg("Calculating First Scores")
        for i, particle in enumerate(particles):
            if verbose > 0:
                print("{} out of {} ({:.2f}%)".format(i + 1, len(particles), (i + 1) / len(particles) * 100))
            parent_dir = os.path.join(iter_dir, str(i))
            if not os.path.isdir(parent_dir):
                os.mkdir(parent_dir)
            err = self.target(structs, particle, parent_dir, true_energies, true_forces)
            errors.append(err)
        # finding the best particle (crucial for next steps)
        gbest = particles[errors.index(min(errors))]
        gbest_err = min(errors)
        pbest = particles[errors.index(min(errors))]
        # main run
        if verbose > 0:
            print_msg("MAIN OPTIMIZATION LOOP")
        for iter_count in range(1, self.niters + 1):
            if verbose > 0:        
                print_msg("ITER. {}".format(iter_count), "*")
            errors = []
            iter_dir = os.path.join(self.comp_dir, str(iter_count))
            if not os.path.isdir(iter_dir):
                os.mkdir(iter_dir)
            for particle_count, (particle, velocities) in enumerate(zip(particles, particle_velocities)):
                if verbose > 0:
                    print("{} out of {} ({:.2f}%)".format(particle_count + 1, len(particles), (particle_count + 1) / len(particles) * 100))
                # randomizing process
                r1 = np.random.rand()
                r2 = np.random.rand()
                # updating velocities & particles
                for i in range(len(self.variables)):
                    velocities[i] = self.w * velocities[i] + self.c1 * r1 * (pbest[i].value - particle[i].value) + self.c2 * r2 * (gbest[i].value - particle[i].value)
                    new_value = particle[i].value + velocities[i]
                    # update values only to legal ones !
                    if new_value <= particle[i].max_value and new_value >= particle[i].min_value:
                        particle[i].value = new_value
                # running target func
                parent_dir = os.path.join(iter_dir, str(particle_count))
                if not os.path.isdir(parent_dir):
                    os.mkdir(parent_dir)
                err = self.target(structs, particle, parent_dir, true_energies, true_forces)
                errors.append(err)
            # update gbest and pbest
            pbest_err = min(errors)
            self.history.append(pbest_err)
            self.pbest_history.append(errors.index(min(errors)))
            pbest = particles[errors.index(min(errors))]
            if pbest_err < gbest_err:
                gbest = particles[errors.index(min(errors))]
                gbest_err = pbest_err
            if verbose > 0:
                print("DONE ITERATION. PBEST =", pbest_err, "GBEST =", gbest_err)

