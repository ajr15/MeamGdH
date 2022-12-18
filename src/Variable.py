from typing import List, Union
import numpy as np
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

    def random_init(self):
        """set a random value (within the limits)"""
        self.value = np.random.rand() * (self.max_value - self.min_value) + self.min_value

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
