from typing import List, Union, Dict
import os
import numpy as np
from copy import copy
from .potentials import EmbeddedAtomElementPotential, EmbeddedAtomInteractionPotential
from .Variable import Variable
from ..config import torinax
from torinax.io import LammpsIn
from torinax.core import Structure


class Simulator:

    """Molecular dynamics simulator object, simulates material properties for given structure
    
    ARGS:
        - parent_dir (str): path to the simulation directory (place for input and output files)
        - parameters (dict): simulation parameter dictionary, defines ALL the parameters required for simulation"""

    def __init__(self, potentials: List[Union[EmbeddedAtomElementPotential, EmbeddedAtomInteractionPotential]]):
        self.potentials = potentials

    def single_run(self, struct: Structure, prefix: str, parent_dir: str):
        """Running the simulation for a given structure"""
        pair_coeff = "pair_coeff * * library.meam" 
        if "H" in [atom.symbol for atom in struct.atoms]:
            pair_coeff += " Gd H interaction.meam Gd H"
        else:
            pair_coeff += " Gd NULL Gd"
        kwdict = {"input_string": "\n".join([
                                                "pair_style meam",
                                                pair_coeff,
                                                "thermo_style custom step pe",
                                                "dump 1 all custom 1 {}.forces id type x y z fx fy fz".format(os.path.join(parent_dir, prefix)),
                                                "run 0"
                                            ])}
        infile = os.path.join(parent_dir, "{}.in".format(prefix))
        outfile = os.path.join(parent_dir, "{}.out".format(prefix))
        input_file = LammpsIn(infile)
        input_file.write_file(struct, kwdict)
        os.system("lmp -in {} > {}".format(infile, outfile))

    def get_updated_potentials(self, variables: List[Variable]):
        potentials = []
        for potential in self.potentials:
            new = copy(potential)
            for variable in variables:
                if variable.potential == new.name and hasattr(new, self.name):
                    new.__setattr__(self.name, self.value)
            potentials.append(new)
        return potentials

    def run(self, structs: Dict[str, Structure], variables: List[Variable], parent_dir: str):
        potentials = self.get_updated_potentials(variables)
        self._write_potential_files(potentials, parent_dir)
        for name, struct in structs.items():
            self.single_run(struct, name)

    @staticmethod
    def _write_potential_files(potentials, parent_dir):
        """Method to write LAMMPS potential files for the current simulation"""
        # writing first all the 
        with open(os.path.join(parent_dir, "library.meam"), "w") as f:
            # writing the constant header of the file
            f.write("\n".join([
                "# DATE: 2022-11-16 UNITS: metal DATE:2022-11-16 CONTRIBUTOR: Someone ",
                "# DATA FOR OPTIMIZED POTENTIALS FOR Gd-H SYSTEM",
                "# elt        lat     z       ielement     atwt",
                "# alpha      b0      b1      b2           b3    alat    esub    asub",
                "# t0         t1              t2           t3            rozero  ibar"
            ]))
            f.write("\n")
            # writing parameter block for the parameters in the "pure" section of the param dictionary
            for pot in potentials:
                if type(pot) is EmbeddedAtomElementPotential:
                    f.write(pot.write_lammps_string() + "\n")
        # writing the meam interaction parameters
        with open(os.path.join(parent_dir, "interaction.meam"), "w") as f:
            f.write("# MUST HAVE COMMENT LINE\n")
            for pot in potentials:
                if type(pot) is EmbeddedAtomInteractionPotential:
                    f.write(pot.write_lammps_string() + "\n")

    @staticmethod
    def parse_lammps_output(lmp_file: str) -> float:
        """Method to read energy value and atomic forces from LAMMPS output"""
        with open(lmp_file, "r") as f:
            pe_line = False
            for line in f.readlines():
                if not pe_line and "PotEng" in line:
                    pe_line = True
                    continue
                if pe_line:
                    return float(line.split()[-1])

    @staticmethod
    def parse_force_file(force_file: str):
        """Method to parse the output atomic force file. returns list of atom coordinates and list of atomic forces"""
        with open(force_file, "r") as f:
            force_block = False
            fxyzs = [] # atomic forces list
            for line in f.readlines():
                if "ITEM: ATOMS id type x y z fx fy fz" in line:
                    force_block = True
                    continue
                if force_block:
                    v = line.split()
                    fxyzs.append([float(x) for x in v[5:]])
        # converting to proper np.array
        forces = np.zeros((len(fxyzs), 3))
        for i in range(len(fxyzs)):
            for j in range(3):
                forces[i, j] = fxyzs[i][j]
        return forces

    def get_energies(self, parent_dir: str) -> Dict[str, float]:
        """Method to get the energy outputs from a simulation"""
        ajr = {}
        for fname in os.listdir(parent_dir):
            if fname.endswith(".out"):
                name = os.path.splitext(fname)[0]
                ajr[name] = self.parse_lammps_output(os.path.join(parent_dir, name + ".out"))
        return ajr

    def get_forces(self, parent_dir: str) -> Dict[str, np.array]:
        """Method to read atomic force values from simulation. returns forces as Nx3 array (for N atoms)"""
        ajr = {}
        for fname in os.listdir(parent_dir):
            if fname.endswith(".forces"):
                name = os.path.splitext(fname)[0]
                forces = self.parse_force_file(os.path.join(parent_dir, name + ".forces"))
                ajr[name] = forces
        return ajr


