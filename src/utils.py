"""Main utils file for fitting 2NN-MEAM potetial for the Gd-H system"""
import sys; sys.path.append("/home/shaharpit/Personal/TorinaX")
from torinax.utils import pymatgen as pmt_utils
from torinax.io import LammpsIn
from pymatgen.core import Structure
import numpy as np
from typing import List
import json
import os
from itertools import product


def print_msg(msg: str, sep: str="="):
    l = len(msg) + 10
    print(sep * l)
    print(" " * 5 + msg)
    print(sep * l)

def run_lammps(struct, infile: str, outfile: str, name: str) -> dict:
    """Method to run LAMMPS calculation for a specific material.
    ARGS:
        - struct: structure to calculate on
        - infile: path to input file
        - outfile: path to output file
    RETURNS: (dict) dictionary with property names (keys) and values"""
    # writing LAMMPS input files
    pair_coeff = "pair_coeff * * library.meam" 
    if "H" in [atom.symbol for atom in struct.atoms]:
        pair_coeff += " Gd H interaction.meam Gd H"
    else:
        pair_coeff += " Gd NULL Gd"
    kwdict = {"input_string": "\n".join([
                                            "pair_style meam",
                                            pair_coeff,
                                            "thermo_style custom step pe",
                                            "dump 1 all custom 1 {}.forces id type x y z fx fy fz".format(name),
                                            "run 0"
                                        ])}
    input_file = LammpsIn(infile)
    input_file.write_file(struct, kwdict)
    os.system("/home/shaharpit/lammps/lammps-23Jun2022/src/lmp_mpi -in {} > {}".format(infile, outfile))

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

def parse_force_file(force_file: str):
    """Method to parse the output atomic force file. returns list of atom coordinates and list of atomic forces"""
    with open(force_file, "r") as f:
        force_block = False
        xyzs = [] # coordinate list
        fxyzs = [] # atomic forces list
        for line in f.readlines():
            if "ITEM: ATOMS id type x y z fx fy fz" in line:
                force_block = True
                continue
            if force_block:
                v = line.split()
                xyzs.append([float(x) for x in v[2:5]])
                fxyzs.append([float(x) for x in v[5:]])
        return xyzs, fxyzs

def write_potential_files(params_dict: dict):
    """Method to write the required LAMMPS potential files for the calculation"""
    # writing library.meam file - file with pure parameters information
    with open("library.meam", "w") as f:
        # writing the constant header of the file
        f.write("\n".join([
            "# DATE: 2012-06-29 UNITS: metal DATE: 2007-06-11 CONTRIBUTOR: Greg Wagner, gjwagne@sandia.gov CITATION: Baskes, Phys Rev B, 46, 2727-2742 (1992) ",
            "# meam data from vax files fcc,bcc,dia    11/4/92",
            "# elt        lat     z       ielement     atwt",
            "# alpha      b0      b1      b2           b3    alat    esub    asub",
            "# t0         t1              t2           t3            rozero  ibar"
        ]))
        f.write("\n")
        # writing parameter block for the parameters in the "pure" section of the param dictionary
        d = params_dict["pure"]
        for k, params in d.items():
            l = [
                "'{}'        '{}'   {}      {}           {}".format(k, params["lat"], params["Z"], params["ielement"], params["atwt"]), 
                "{}        {}     {}     {}          {}   {}   {}    {}".format(params["alpha"], params["b0"], params["b1"], params["b2"], params["b3"], params["alat"], params["esub"], params["asub"]), 
                "{}         {}            {}        {}         {}      {}".format(params["t0"], params["t1"], params["t2"], params["t3"], params["rosero"], params["ibar"])
                ]
            f.write("\n".join(l))
            f.write("\n")
    # writing the meam interaction parameters
    with open("interaction.meam", "w") as f:
        f.write("# MUST HAVE COMMENT LINE\n")
        for k, v in params_dict["interaction"].items():
            if type(v) is str:
                s = "'{}'".format(v)
            else:
                s = str(v)
            f.write("{} = {}\n".format(k, s))


def find_gdh2_positions(struct: Structure):
    """Method to find the list of possible hydrogen positions for the GdH2 substructure"""
    # setting reference positions to determine the relative positions of hydrogens and Gd atoms
    base_pos = np.array([3.614, 4.166, 4.517])
    h1_pos = np.array([3.823, 1.957, 5.700])
    h2_pos = np.array([1.706, 3.129, 5.800])
    # setting relative positions
    v1 = base_pos - h1_pos
    v2 = base_pos - h2_pos
    # for each atom make the positions using the transformation vectors
    positions = []
    for site in struct.sites:
        if site.species_string == "Gd":
            pos1 = site.coords + v1
            pos2 = site.coords + v2
            positions.append(pos1)
            positions.append(pos2)
    return positions


def find_gdh_positions(struct: Structure):
    """Method to find the list of possible hydrogen positions for the GdH substructure"""
    # setting reference positions to determine the relative positions of hydrogens and Gd atoms
    base_pos = np.array([3.614, 4.166, 4.517])
    h_pos = np.array([3.716, 4.149, 6.565])
    # setting relative positions
    v = base_pos - h_pos
    # for each atom make the positions using the transformation vectors
    positions = []
    for site in struct.sites:
        if site.species_string == "Gd":
            pos = site.coords + v
            positions.append(pos)
    return positions


def make_hydride(struct: Structure, positions: List[np.array]):
    """Method to make a hydride structure given hydrogn position IDs"""
    nstruct = struct.copy()
    for pos in positions:
        nstruct.append("H", pos, coords_are_cartesian=True)
    return nstruct


def calculate_relative_energies(output_dictionary: dict) -> dict:
    """Get the relative energy dictionary from the output dictionary of a simulator / DFT run.
    this dictionary is given as {'{phase}_{h_conc}_{scale}': total_energy}. this method converts the total energy into relative energy (compared to the minimum energy of phase-h_conc combo."""
    phases = ["fcc", "hcp"]
    concs = [25, 50, 75, 100, 125]
    # going through the dictionary to find minimal values
    min_values = {"{}_{}".format(phase, conc): 1000 for phase, conc in product(phases, concs)}
    for key, value in output_dictionary.items():
        for family, min_value in min_values.items():
            if family in key and value < min_value:
                min_values[family] = value
    # updating dictionary based on minimal values
    res = {}
    for key, value in output_dictionary.items():
        for family, min_value in min_values.items():
            if family in key:
                res[key] = value - min_value
    return res


if __name__ == "__main__":
    # testing of the utils
    with open("parameters.json", "r") as f:
        d = json.load(f)
        write_potential_files(d)
    struct = pmt_utils.read_structure_from_cif("C:\\Users\\asaf\\OneDrive\\Documents\\Shachar\\Gd Corrosion Project\\Potential fit\\dft\\cifs\\Gd.cif")
    run_lammps(struct, "test.in", "test.out")
    print("ENERGY", parse_lammps_output("test.out"))
    xyzs, fxyzs = parse_force_file("forces.dump")
    print("COORDS")
    for xyz in xyzs:
        print(xyz)
    print("FORCES")
    for fxyz in fxyzs:
        print(fxyz)