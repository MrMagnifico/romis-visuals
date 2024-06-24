from utils import *

import numpy as np
import random
from abc import ABC, abstractmethod
from matplotlib import pyplot as plt
from typing import NamedTuple


class Spike(NamedTuple):
    begin: float
    end: float
    peak: float


class UnitUniform(ABC):
    """Abstract class representing distributions defined across the domain [0:1]"""
    @abstractmethod
    def evaluate(self, x: float) -> float:
        """
        Evaluate this distribution at a given x

        Args:
            x: Value for which to evaluate the distribution

        Returns:
            PDF at the given x
        """

    @abstractmethod
    def sample(self) -> float:
        """
        Draw a sample from this distribution

        Returns:
            Value randomly drawn from the support of this distribution proportional to PDF
        """


class SpikeyDistribution(UnitUniform):
    def __init__(self, spikes: list[Spike]):
        """
        Define a spikey distribution

        Args:
            spikes: Spikes that define the shape of the distribution
        """
        # Sort spikes based on beginning point and verify that spikes are non overlapping
        self._spikes = sorted(spikes, key=lambda spike: spike.begin)
        for idx in range(len(self._spikes) - 1):
            if self._spikes[idx].end > self._spikes[idx + 1].begin:
                raise Exception("Spikes are overlapping in instantiation of SpikeyDistribution")
            
        # Precompute PDFs of each spike
        self._spikes_pdfs = []
        for spike in self._spikes:
            self._spikes_pdfs.append(0.5 * (spike.end - spike.begin) * spike.peak)
        self._spikes_pdfs = normalise(self._spikes_pdfs)

    def _sample_spike(self, spike: Spike, x: float):
        # Determine which end of the spike the given x lies on and construct linear function appropriately
        middle      = (spike.begin + spike.end) / 2
        gradient    = None
        intercept   = None
        if x <= middle:
            gradient    = (spike.peak) / (middle - spike.begin)
            intercept   = -gradient * spike.begin
        else:
            gradient    = (-spike.peak) / (spike.end - middle)
            intercept   = -gradient * spike.end
        return gradient * x + intercept
    
    def evaluate(self, x: float) -> float:
        super().evaluate(x)
        for spike in self._spikes:
            # Determine if given x lies within bounds of this spike
            if spike.begin <= x <= spike.end:
                return self._sample_spike(spike, x)
        return 0
    
    def sample(self) -> float:
        chosen_idx: int = np.random.choice(range(len(self._spikes)), size=1, p=self._spikes_pdfs)[0]
        chosen_spike    = self._spikes[chosen_idx]
        return random.triangular(chosen_spike.begin, chosen_spike.end)
    

class ConstantDistribution(UnitUniform):
    def __init__(self, lower: float, upper: float) -> None:
        if lower < 0 or upper > 1 or lower > upper:
            raise Exception(f"Input parameters of uniform distribution [{lower}, {upper}] are invalid")
        self._lower = lower
        self._upper = upper

    def evaluate(self, x: float) -> float:
        if self._lower <= x <= self._upper:
            return 1 / (self._upper - self._lower) 
        return 0
    
    def sample(self) -> float:
        return random.uniform(self._lower, self._upper)


if __name__ == "__main__":
    x_axis = np.linspace(0, 1, num=256)

    # Spikey
    spikey      = SpikeyDistribution([Spike(0.1, 0.3, 0.5), Spike(0.6, 0.9, 0.2)])
    spikey_vals = list(map(spikey.evaluate, x_axis))
    plt.plot(x_axis, spikey_vals, label="Spikey")

    # Constant
    constant        = ConstantDistribution(0, 0.5)
    constant_vals   = list(map(constant.evaluate, x_axis))
    plt.plot(x_axis, constant_vals, label="Constant")

    plt.title("Distribution Showcase")
    plt.show()
