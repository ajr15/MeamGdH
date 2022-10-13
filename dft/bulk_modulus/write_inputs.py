import sys
sys.path.append("/home/shaharpit/Personal/TorinaX")
from pymatgen.core import Structure
from torinax.io import QeIn
from torinax.utils.pymatgen import pmt_struct_to_structure

kwdict = {
  "CONTROL": {
    	"pseudo_dir": '/home/shaharpit/AcHProject/meam_fit/dft/qe_potentials',
    	"outdir": '/home/shaharpit/tmp',
    	"calculation": 'relax',
      "nstep": 150
      },
  "SYSTEM": {
  	  "occupations": 'smearing',
  	  "degauss": 0.01,
      "ecutwfc": 50
     },
  "IONS": {
      "ion_dynamics": 'bfgs',
     },
  "ATOMIC_SPECIES": {
      "Gd": "Gd.GGA-PBE-paw-v1.0.UPF",
      "H": "H.pbe-rrkjus_psl.1.0.0.UPF",
      },
  'ELECTRONS': {
      'adaptive_thr': True,
      'electron_maxstep': 500, 
      #'mixing_beta': 0.3, 
      #'mixing_mode': 'local-TF'
      },
  'K_POINTS': {
      'type': 'automatic', 
      'vec': [4, 4, 4, 0, 0, 0]
      }
  }

def make_inputs(struct: Structure, kwds_dict: dict, prefix: str):
    V = struct.volume
    # writing inputs with scaling
    scalars = [0.8, 0.9, 1, 1.1, 1.2]
    for s in scalars:
        print("Writing with {}".format(s))
        struct.scale_lattice(s * V)
        infile = QeIn("./{}_{}.in".format(prefix, s))
        infile.write_file(pmt_struct_to_structure(struct), kwds_dict)
    print("ALL DONE !")


if __name__ == "__main__":
    gd_struct = Structure.from_file("../cifs/Gd.cif")
    make_inputs(gd_struct, kwdict, "Gd")
    gdh2_struct = Structure.from_file("../cifs/GdH2.cif")
    make_inputs(gdh2_struct, kwdict, "GdH2")

