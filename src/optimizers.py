from typing import Dict, List
import numpy as np
import os
from ..config import torinax
from torinax.core import Structure
from .simulators import Simulator
from .Variable import Variable
from .losses import Loss

class PsoOptimizer:

    """Wrapper around the sklearn.BaseEstimator for optimization of the potential function. Allows for bayesian optimization on hyperparameters of optimizer"""

    def __init__(self, 
                    simulator: Simulator, 
                    variables: List[Variable], 
                    loss: Loss,
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

    def target(self, structs: Dict[str, Structure], variables: List[Variable], parent_dir: str, true_energies: Dict[str, float], true_forces: Dict[str, np.array]) -> float:
        """The target function of the simulator"""
        self.simulator.run(structs, variables, parent_dir)
        pred_energies = self.simulator.get_energies(parent_dir)
        pred_forces = self.simulator.get_forces(parent_dir)
        return self.loss.evaluate(true_energies, true_forces, pred_energies, pred_forces)

    def fit(self, structs: Dict[str, Structure], true_energies: Dict[str, float], true_forces: Dict[str, np.array], verbose: int=1):
        """Optimize simulator variables on given structures and data"""
        # initializing N particles with random locations and velocities
        particles = []
        particle_velocities = []
        for _ in range(self.nparticles):
            vars = []
            for variable in self.variables:
                kwds = variable.__dict__
                kwds["value"] = np.random.uniform(variable.min_value, variable.max_value)
                vars.append(Variable(**kwds))
            particles.append(vars)
            particle_velocities.append([np.random.rand() for k in range(len(self.variables))])
        # run target func on all particles
        errors = []
        for i, particle in enumerate(particles):
            err = self.target(structs, particle, os.path.join(self.comp_dir, str(i)), true_energies, true_forces)
            errors.append(err)
        # finding the best particle (crucial for next steps)
        gbest = particles[errors.index(min(errors))]
        gbest_err = min(errors)
        pbest = particles[errors.index(min(errors))]
        # main run
        for i in range(1, self.niter + 1):
            errors = []
            for particle, velocities in zip(particles, particle_velocities):
                # randomizing process
                r1 = np.random.rand()
                r2 = np.random.rand()
                # updating velocities & particles
                for i in range(len(self.variables)):
                    velocities[i] = self.w * velocities[i] + self.c1 * r1 * (pbest[i].value - particle[i].value) + self.c2 * r2 * (gbest[i].value - particle[i].value)
                    particle[i].value = particle[i].value + velocities[i]
                # running target func
                err = self.target(structs, particle, os.path.join(self.comp_dir, str(i)), true_energies, true_forces)
                errors.append(err)
            # update gbest and pbest
            pbest_err = min(errors)
            self.history.append(pbest_err)
            pbest = particles[errors.index(min(errors))]
            if pbest_err < gbest_err:
                gbest = particles[errors.index(min(errors))]
                gbest_err = pbest_err

