"""Result containers used by the workshop fitting helper."""

import html
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


@dataclass
class FitReport:
    """Summary of a completed fit: parameters, goodness of fit and residual checks.

    ``print(report)`` gives a plain-text table, ``report`` alone renders an HTML
    table in a notebook, and ``report.to_markdown()`` returns Markdown.
    """

    model: str
    engine: str
    method: str
    n_points: int
    n_fitted: int
    chisq: float
    parameters: dict[str, dict[str, object]]
    residuals_within_2: float
    sign_changes: int

    @property
    def dof(self) -> int:
        """Degrees of freedom, N - p."""
        return self.n_points - self.n_fitted

    @property
    def reduced_chisq(self) -> float:
        """Reduced chi-squared, chi^2 / (N - p)."""
        return self.chisq / self.dof if self.dof > 0 else float('nan')

    @property
    def expected_spread(self) -> float:
        """Expected scatter of the reduced chi-squared for a good fit, sqrt(2 / (N - p))."""
        return float(np.sqrt(2 / self.dof)) if self.dof > 0 else float('nan')

    def summary_lines(self) -> list[str]:
        """Return the header lines shared by every output format."""
        return [
            f'Model: {self.model}   (engine: {self.engine}, method: {self.method})',
            f'N = {self.n_points} points, p = {self.n_fitted} fitted parameters, '
            f'ν = N − p = {self.dof}',
            f'χ² = {self.chisq:.4g};  reduced χ²ν = {self.reduced_chisq:.3f} '
            f'(expected 1 ± {self.expected_spread:.2f})',
            f'Normalized residuals: {self.residuals_within_2:.0%} within ±2; '
            f'{self.sign_changes} sign changes (random scatter gives about '
            f'{self.n_points // 2})',
        ]

    def table_rows(self) -> list[list[str]]:
        """Return the parameter table as strings: name, value, uncertainty, status, bounds."""
        rows = []
        for name, info in self.parameters.items():
            value = f'{info["value"]:.6g}'
            if info['vary']:
                stderr = info.get('stderr')
                uncertainty = f'{stderr:.2g}' if stderr else '—'
                status = 'fitted'
                bounds = f'[{info["min"]:.4g}, {info["max"]:.4g}]'
            else:
                uncertainty, status, bounds = '—', 'fixed', ''
            if info.get('note'):
                status = f'{status}; {info["note"]}'
            rows.append([name, value, uncertainty, status, bounds])
        return rows

    HEADERS = ('Parameter', 'Value', 'Uncertainty', 'Status', 'Bounds')

    def __str__(self) -> str:
        rows = self.table_rows()
        widths = [
            max(len(header), *(len(row[i]) for row in rows)) if rows else len(header)
            for i, header in enumerate(self.HEADERS)
        ]

        def line(cells):
            return '  '.join(cell.ljust(width) for cell, width in zip(cells, widths))

        table = [line(self.HEADERS), line(['-' * width for width in widths])]
        table += [line(row) for row in rows]
        return '\n'.join(self.summary_lines() + [''] + [row.rstrip() for row in table])

    __repr__ = __str__

    def to_markdown(self) -> str:
        """Return the report as Markdown (summary list and parameter table)."""
        summary = '\n'.join(f'- {line}' for line in self.summary_lines())
        header = '| ' + ' | '.join(self.HEADERS) + ' |'
        divider = '|' + '|'.join('---' for _ in self.HEADERS) + '|'
        body = '\n'.join('| ' + ' | '.join(row) + ' |' for row in self.table_rows())
        return f'{summary}\n\n{header}\n{divider}\n{body}\n'

    def _repr_html_(self) -> str:
        summary = '<br>'.join(html.escape(line) for line in self.summary_lines())
        header = ''.join(f'<th style="text-align:left">{h}</th>' for h in self.HEADERS)
        body = ''.join(
            '<tr>'
            + ''.join(f'<td style="text-align:left">{html.escape(c)}</td>' for c in row)
            + '</tr>'
            for row in self.table_rows()
        )
        return (
            f'<div><p style="font-family:monospace">{summary}</p>'
            f'<table><thead><tr>{header}</tr></thead><tbody>{body}</tbody></table></div>'
        )
