"""Teaching-oriented SANS fitting helper bundled with the workshop."""

from .results import PosteriorSummary
from .sans_fitter import SANSFitter

__version__ = '0.1.0'
__all__ = ['PosteriorSummary', 'SANSFitter']
