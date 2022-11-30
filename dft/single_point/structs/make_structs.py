PARENT_DIR = "../../../"
import sys; sys.path.append(PARENT_DIR)
import os
from pymatgen.core import Structure
from src.utils import make_hydride

def make_hcp_struct(base_struct: Structure, n_hydrogens: int) -> Structure:
    pass

def make_fcc_struct(base_struct: Structure, n_hydrogens: int) -> Structure:
    pass

def make_naked_fcc_struct() -> Structure:
    """Make the naked FCC structure ready to be loaded with hydrogens"""
    cif = os.path.join(PARENT_DIR, "structures", "base", "GdH2.cif")
    struct = Structure.from_file(cif)
    struct.remove_species(["H"])
    struct.to("cif", "naked_fcc.cif")
    return struct

def make_naked_hcp_struct() -> Structure:
    """Make the naked HCP structure ready to be loaded with hydrogens"""
    cif = os.path.join(PARENT_DIR, "structures", "base", "Gd.cif")
    struct = Structure.from_file(cif)
    struct.make_supercell([2, 1, 1])
    struct.to("cif", "naked_hcp.cif")
    return struct

if __name__ == "__main__":
    # setting up positions in unit cell
    hcp_positions = {
        1: [[1.882, 0.977, -0.180], [-2.066, 0.834, -0.219]],   # = blue (in documentation)
        2: [[-0.187, -0.273, 2.942], [3.741, -0.088, 2.882]],   # = purple (in documentation)
        3: [[1.792, -0.052, 2.138], [5.489, 0.155, 1.723]],     # = green (in documentation)
        4: [[-0.227, 1.220, -1.054], [3.922, 1.147, -0.884]],   # = red (in documentation)
        5: [[-0.014, -0.570, 0.407], [3.934, -0.538, 0.421]]    # = gray (in documentation)
    }
    fcc_positions = {
        1: [[1.326, 3.998, 3.998], [1.326, 3.978, 1.326]],
        2: [[1.326, 1.326, 1.326], [1.326, 1.326, 3.978]],
        3: [[3.978, 1.326, 3.978], [3.978, 1.326, 1.326]],
        4: [[3.978, 3.978, 1.326], [3.978, 3.978, 3.978]],
        5: [[2.652, 2.652, 2.652], [2.652, 0, 0]]
    }
    hcp = make_naked_hcp_struct()
    fcc = make_naked_fcc_struct()
    hcp_h_pos = []
    fcc_h_pos = []
    for i in range(1, 6):
        print("making for {} hydrogens".format(i * 2))
        # making hcp structures
        hcp_h_pos += hcp_positions[i]
        s = make_hydride(hcp, hcp_h_pos)
        s.to("cif", "hcp_{}.cif".format(round(i * 2 / 8 * 100)))
        # making fcc structures
        fcc_h_pos += fcc_positions[i]
        s = make_hydride(fcc, fcc_h_pos)
        s.to("cif", "fcc_{}.cif".format(round(i * 2 / 8 * 100)))
