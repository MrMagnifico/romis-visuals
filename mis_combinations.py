from sampling_functions import *

import numpy as np
import sys
from functools import reduce
from typing import Callable, NamedTuple


class SampleDrawn(NamedTuple):
    value: float
    gen_dist: UnitUniform
    target_dist: UnitUniform

class SampleRIS(NamedTuple):
    value: float
    weight: float


def balance_heuristic_mis_weights(samples: list[SampleDrawn]) -> list[float]:
    weights = []
    for sample in samples:
        denom = reduce(lambda lhs, rhs: lhs + rhs.target_dist.evaluate(sample.value), 
                       samples, sys.float_info.min)
        weights.append(sample.target_dist.evaluate(sample.value) / denom)
    return weights


def pairwise_mis_weights(samples: list[SampleDrawn], canonical_idx: int) -> list[float]:
    weights             = []
    canonical_sample    = samples[canonical_idx]
    for idx, sample in enumerate(samples):
        const = 1 / (len(samples) - 1)
        if idx == canonical_idx:
            sum                     = 0
            target_val_canonical    = sample.target_dist.evaluate(sample.value)
            for nc_idx, non_canonical_sample in enumerate(samples):
                if nc_idx == canonical_idx:
                    continue
                sum += target_val_canonical / (non_canonical_sample.target_dist.evaluate(sample.value) + target_val_canonical) if target_val_canonical > 0 else 0
            weights.append(const * sum)
        else:
            target_val_sample   = sample.target_dist.evaluate(sample.value)
            prob_val            = target_val_sample / (target_val_sample + canonical_sample.target_dist.evaluate(sample.value)) if target_val_sample > 0 else 0
            weights.append(const * prob_val)
    return weights


def ris(samples: list[SampleDrawn], mis_weights: list[float], target_function: Callable[[float], float],
        num_samples: int = 1) -> list[SampleRIS]:
    # Evaluate resampling weights
    ris_weights = []
    for idx, sample in enumerate(samples):
        ris_weights.append(mis_weights[idx] * target_function(sample.value) * (1 / sample.gen_dist.evaluate(sample.value)))
    
    # Choose sample(s) proportional to RIS weights and return them alongside their unbiased contribution weight(s)
    ris_weights                     = normalise(ris_weights)
    chosen_indices: np.ndarray[int] = np.random.choice(range(len(samples)), size=num_samples, p=ris_weights, replace=True)
    chosen_samples                  = [samples[idx] for idx in chosen_indices]
    sum_ris_weights                 = sum(ris_weights)
    unbiased_contribution_weights   = [(1 / target_function(chosen.value)) * sum_ris_weights for chosen in chosen_samples]
    return [SampleRIS(sample.value, weight) for sample, weight in zip(chosen_samples, unbiased_contribution_weights)]


if __name__ == "__main__":
    # Define distributions to draw from and to target
    draw1   = SpikeyDistribution([Spike(0.0, 0.3, 4), Spike(0.3, 0.4, 3)])
    draw2   = SpikeyDistribution([Spike(0.5, 0.8, 3), Spike(0.8, 0.9, 7)])
    target1 = ConstantDistribution(0.1, 0.3)
    target2 = ConstantDistribution(0.6, 0.8)

    # Plot distributions
    x_axis  = np.linspace(0, 1, num=256)
    plt.plot(x_axis, list(map(draw1.evaluate, x_axis)),     linestyle="--", label="Draw1")
    plt.plot(x_axis, list(map(draw2.evaluate, x_axis)),     linestyle="--", label="Draw2")
    plt.plot(x_axis, list(map(target1.evaluate, x_axis)),   linestyle="--", label="Target1")
    plt.plot(x_axis, list(map(target2.evaluate, x_axis)),   linestyle="--", label="Target2")

    # Draw a number of samples from each distribution
    SAMPLES_PER_DIST                    = 8
    dists                               = list(zip([draw1, draw2], [target1, target2]))
    drawn_samples: list[SampleDrawn]    = []
    for _ in range(SAMPLES_PER_DIST):
        for draw_dist, target_dist in dists:
            drawn_samples.append(SampleDrawn(draw_dist.sample(), draw_dist, target_dist))

    # Plot drawn samples
    x_vals = list(map(lambda x: x.value,                        drawn_samples))
    y_vals = list(map(lambda x: x.gen_dist.evaluate(x.value),   drawn_samples))
    plt.scatter(x_vals, y_vals, label="Drawn", s=20)

    # Apply RIS with different weights
    TARGET_FUNCTION     = target1.evaluate
    balance_weights     = balance_heuristic_mis_weights(drawn_samples)
    pairwise_weights    = pairwise_mis_weights(drawn_samples, 0)
    ris_balance         = ris(drawn_samples, balance_weights,   TARGET_FUNCTION)
    ris_pairwise        = ris(drawn_samples, pairwise_weights,  TARGET_FUNCTION)

    # Plot samples chosen with RIS
    ris_samples: list[tuple[list[SampleRIS], str]] = [(ris_balance, "Balance"), (ris_pairwise, "Pairwise")]
    for samples, label in ris_samples:
        x_vals = list(map(lambda x: x.value,                                samples))
        y_vals = list(map(lambda x: TARGET_FUNCTION(x.value) * x.weight,    samples))
        plt.scatter(x_vals, y_vals, label=label)

    # Render plot
    plt.legend()
    plt.show()
