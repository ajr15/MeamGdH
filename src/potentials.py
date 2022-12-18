from dataclasses import dataclass
import json

@dataclass
class EmbeddedAtomElementPotential:

    """Object for handling the LAMMPS MEAM potential for an element"""

    name: str       # potential name
    symbol: str     # element symbol (H, Gd...)
    lat: str        # lattice of the structure
    Z: int          # coordination number (n nearest neighbors) in reference structure
    ielement: int   # atomic number
    atwt: float     # atomic weight
    alat: float     # lattice constant of reference structure
    esub: float     # cohesive energy (?)
    alpha: float    # the "alpha" parameter in the Rose equation of states (for ref structure)
    asub: float=1   # "A" parameter for MEAM (in density function)
    ibar: int=1     # shape of gamma function in electron density (see LAMMPS manual for more details)
    rozero: float=1 # element density scaling
    b0: float=1     # beta 0 parameter in embedding function
    b1: float=1     # beta 1 parameter in embedding function
    b2: float=1     # beta 2 parameter in embedding function
    b3: float=1     # beta 3 parameter in embedding function
    t0: float=1     # t 0 parameter in embedding function
    t1: float=1     # t 1 parameter in embedding function
    t2: float=1     # t 2 parameter in embedding function
    t3: float=1     # t 3 parameter in embedding function
    
    def to_json(self, path: str):
        """Method to save the potential to a json file"""
        with open(path, "w") as f:
            json.dump(self.__dict__, f)

    @staticmethod
    def from_json(path: str):
        """Method to load the potential from a json file"""
        with open(path, "r") as f:
            d = json.load(f)
            return EmbeddedAtomElementPotential(**d)

    def write_lammps_string(self) -> str:
        """Method to write the potential to string (to be written in LAMMPS files)"""
        # writing the block describing the potential in the library.meam LAMMPS file
        l = [
            "'{}'        '{}'   {}      {}           {}".format(self.symbol, self.lat, self.Z, self.ielement, self.atwt), 
            "{}        {}     {}     {}          {}   {}   {}    {}".format(self.alpha, self.b0, self.b1, self.b2, self.b3, self.alat, self.esub, self.asub), 
            "{}         {}            {}        {}         {}      {}".format(self.t0, self.t1, self.t2, self.t3, self.rozero, self.ibar)
            ]
        return "\n".join(l)
    

@dataclass
class EmbeddedAtomInteractionPotential:

    """Object for handling the LAMMPS MEAM potential for interactions"""

    name: str           # potential name
    lattce: str        # lattice of the binary reference structure of the 2 elements
    Ec: float           # cohesive energy of ref binary structure
    alpha: float        # alpha parameter of ref binary structure (calculated by bulk modulus)
    re11: float         # nearest distance between element 1 and element 1 in ref binary structure
    re12: float         # nearest distance between element 1 and element 2 in ref binary structure
    re22: float         # nearest distance between element 2 and element 2 in ref binary structure
    delr: float=0.1     # smoothing length of cutoff function
    Cmax121: float=2.8  # Cmax screening parameter of between 2 element 1 atoms screened by another element 1
    Cmax122: float=2.8  # Cmax screening parameter of between 2 element 1 atoms screened by another element 2
    Cmax112: float=2.8  # Cmax screening parameter of between element 1 and element 2 atoms screened by another element 1
    Cmax221: float=2.8  # Cmax screening parameter of between element 1 and element 2 atoms screened by another element 2
    Cmin121: float=2    # Cmin screening parameter of between 2 element 1 atoms screened by another element 1
    Cmin122: float=2    # Cmin screening parameter of between 2 element 1 atoms screened by another element 2
    Cmin112: float=2    # Cmin screening parameter of between element 1 and element 2 atoms screened by another element 1
    Cmin221: float=2    # Cmin screening parameter of between element 1 and element 2 atoms screened by another element 2

    def to_json(self, path: str):
        """Method to save the potential to a json file"""
        with open(path, "w") as f:
            json.dump(self.__dict__, f)

    @staticmethod
    def from_json(path: str):       
        """Method to load the potential from a json file"""
        with open(path, "r") as f:
            d = json.load(f)
            return EmbeddedAtomElementPotential(**d)

    def get_name(self, key: str):
        """Method to get proper name of a key"""
        if "re" in key:
            return "re({})".format(",".join(key[2:]))
        elif "Cmax" in key:
            return "Cmax({})".format(",".join(key[4:]))
        elif "Cmin" in key:
            return "Cmin({})".format(",".join(key[4:]))
        elif key == "Ec":
            return "Ec(1, 2)"
        elif key == "alpha":
            return "alpha(1, 2)"
        elif key == "lattce":
            return "lattce(1, 2)"
        else:
            return key


    def write_lammps_string(self) -> str:
        """Method to write the potential to string (to be written in LAMMPS files)"""
        res = ""
        for k, v in self.__dict__.items():
            if k == "name": 
                continue
            if type(v) is str:
                s = "'{}'".format(v)
            else:
                s = str(v)
            res += "{} = {}\n".format(self.get_name(k), s)
        return res
