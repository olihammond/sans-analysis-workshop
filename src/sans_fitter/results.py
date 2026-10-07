"""Result containers used by the workshop fitting helper."""

from dataclasses import dataclass, field

import numpy as np


@dataclass
class PosteriorSummary:
    """Posterior samples and summary statistics from a Bayesian fit."""

    labels: list[str]
    samples: np.ndarray
    logp: np.ndarray | None = None
    chains: np.ndarray | None = None
    best: dict[str, float] = field(default_factory=dict)
    mean: dict[str, float] = field(default_factory=dict)
    median: dict[str, float] = field(default_factory=dict)
    std: dict[str, float] = field(default_factory=dict)
    ci_68: dict[str, tuple[float, float]] = field(default_factory=dict)
    ci_95: dict[str, tuple[float, float]] = field(default_factory=dict)
    diagnostics: dict[str, dict[str, float]] = field(default_factory=dict)

    def index_of(self, parameter: str) -> int:
        """Return the posterior column for a parameter."""
        try:
            return self.labels.index(parameter)
        except ValueError:
            available = ', '.join(self.labels)
            raise KeyError(
                f"Parameter '{parameter}' is not sampled. Available: {available}"
            ) from None

    def save_posterior_csv(self, filename: str) -> None:
        """Write posterior samples and optional log probability to CSV."""
        columns = list(self.labels)
        arrays = [self.samples]
        if self.logp is not None:
            columns.append('logp')
            arrays.append(self.logp.reshape(-1, 1))
        np.savetxt(
            filename,
            np.hstack(arrays),
            delimiter=',',
            header=','.join(columns),
            comments='',
            fmt='%.8e',
        )
