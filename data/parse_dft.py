import sys
import os
import json
sys.path.append("C:\\Users\\asaf\\Documents\\GitHub\\TorinaX")
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
                forces.append([float(s) for s in line.split()])
        return forces

def to_cif(outfile: str, cif: str):
    outfile = QeOut(outfile)
    struct = outfile.read_specie()
    pmt_struct = pmt_utils.structure_to_pmt_structure(struct)
    pmt_struct.to("cif", cif)

def parse_energy(outfile: str) -> float:
    outfile = QeOut(outfile)
    return outfile.read_scalar_data()["total_energy"]

if __name__ == "__main__":
    d = {}
    dft_dir = "C:\\Users\\asaf\\OneDrive\\Documents\\Shachar\\Gd Corrosion Project\\Potential fit\\dft\\bulk_modulus"
    for fname in os.listdir(dft_dir):
        if fname.endswith(".out"):
            name = os.path.splitext(fname)[0]
            print("Reading", name)
            outfile = os.path.join(dft_dir, fname)
            ajr = {}
            ajr["energy"] = parse_energy(outfile)
            ajr["forces"] = parse_forces(outfile)
            d[name] = ajr
            to_cif(outfile, os.path.join("..\\structures", name + ".cif"))
    with open("data.json", "w") as f:
        json.dump(d, f)
