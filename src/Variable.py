from typing import List, Union
from dataclasses import dataclass
from .potentials import EmbeddedAtomElementPotential, EmbeddedAtomInteractionPotential


@dataclass
class Variable:

    """A simulation variable"""

    name: str
    potential: Union[EmbeddedAtomElementPotential, EmbeddedAtomInteractionPotential]
    min_value: float
    max_value: float
    value: float

    def set_value(self, value: float):
        self.value = value

    def __eq__(self, obj):
        if isinstance(obj, Variable):
            return obj.value == self.value
        else:
            raise ValueError("Variable can be compared only to another Variable")

    def __lt__(self, obj):
        if isinstance(obj, Variable):
            return self.value < obj.value
        else:
            raise ValueError("Variable can be compared only to another Variable")

    def __gt__(self, obj):
        if isinstance(obj, Variable):
            return self.value > obj.value
        else:
            raise ValueError("Variable can be compared only to another Variable")
