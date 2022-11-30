import os
import sys
sys.path.append("/home/shaharpit/Personal/TorinaX")
from pymatgen.core import Structure
from torinax.io import QeIn
from torinax.utils.pymatgen import pmt_struct_to_structure

kwdict = {
  "CONTROL": {
    	"pseudo_dir": '/home/shaharpit/MeamGdH/dft/qe_potentials/',
    	"outdir": '/home/shaharpit/tmp',
    	"calculation": 'scf',
      "nstep": 150
      },
  "SYSTEM": {
  	  "occupations": 'smearing',
  	  "degauss": 0.01,
      "ecutwfc": 50,
      "lspinorb": True,
      "noncolin": True
     },
  "IONS": {
      "ion_dynamics": 'bfgs',
     },
  "ATOMIC_SPECIES": {
      "Gd": "Gd.rel-pbe-spdn-kjpaw_psl.1.0.0.UPF",
      "H": "H.pbe-kjpaw_psl.0.1.UPF",
      },
  'ELECTRONS': {
      'adaptive_thr': True,
      'electron_maxstep': 500, 
      },
  'K_POINTS': {
      'type': 'automatic', 
      'vec': [4, 4, 4, 0, 0, 0]
      }
  }


if __name__ == "__main__":
    for fname in os.listdir("../structs"):
        if fname.endswith(".cif") and not fname.startswith("naked"):
            name = os.path.splitext(fname)[0]
            print(name)
            struct = Structure.from_file(os.path.join("../structs", fname))
            V = struct.volume
            for scale in [0.9, 1, 1.1]:
                struct.scale_lattice(scale * V)
                infile = QeIn("./{}_{}.in".format(name, scale))
                infile.write_file(pmt_struct_to_structure(struct), kwdict)


