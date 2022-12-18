import sys
import os
import json
sys.path.append("/home/shaharpit/MeamGdH")
from src.utils import calculate_relative_energies
sys.path.append("/home/shaharpit/Personal/TorinaX")
from torinax.utils import pymatgen as pmt_utils
from torinax.io.QeOut import QeOut

def parse_forces(outfile: str):
    with open(outfile, "r") as f:
        read_block = False
        empty_counter = 0
        forces = []
        for line in f.readlines():
            if "Forces acting on atoms" in line:
                read_block = True
                empty_counter = 0
                forces = []
                continue
            if read_block:
                if len(line) == 1:
                    empty_counter += 1
                    if empty_counter == 2:
                        read_block = False
                        continue
                    continue
                line = line.split("=")[-1]
                # reading forces in (kcal/mol)/angstrom units
                forces.append([float(s) * 23.06 * 13.61 / 0.529 for s in line.split()])
        return forces

def parse_energy(outfile: str) -> float:
    outfile = QeOut(outfile)
    return outfile.read_scalar_data()["total_energy"] * 23.06 * 13.61 # converting Ry to kcal/mol

if __name__ == "__main__":
    d = {"energy": {}, "forces": {}}
    dft_dir = "/home/shaharpit/MeamGdH/dft/single_point/comps"
    for fname in os.listdir(dft_dir):
        if fname.endswith(".out"):
            name = os.path.splitext(fname)[0]
            print("Reading", name)
            outfile = os.path.join(dft_dir, fname)
            d["energy"][name] = parse_energy(outfile)
            d["forces"][name] = parse_forces(outfile)
    d["energy"] = calculate_relative_energies(d["energy"])
    with open("/home/shaharpit/MeamGdH/data/data.json", "w") as f:
        json.dump(d, f, indent=2)
