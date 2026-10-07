"""Small fitting interface used by the SANS data analysis workshop."""

import warnings
from typing import Any, Literal, Optional

import matplotlib.pyplot as plt
import numpy as np
from bumps.fitters import fit as bumps_fit

try:
    from bumps.util import format_uncertainty
except ImportError:  # BUMPS < 1.0
    from bumps.formatnum import format_uncertainty

# Fitting engine imports
from bumps.names import FitProblem
from sasdata.dataloader.loader import Loader
from sasmodels.bumps_model import Experiment
from sasmodels.bumps_model import Model as BumpsModel

# SasModels and SasData imports
from sasmodels.core import load_model
from sasmodels.direct_model import DirectModel

from .results import PosteriorSummary

try:
    from scipy.optimize import differential_evolution, least_squares, leastsq

    LMFIT_AVAILABLE = True
except ImportError:
    LMFIT_AVAILABLE = False
    warnings.warn('scipy not available. Only bumps engine will work.', stacklevel=2)


class SANSFitter:
    """
    A flexible SANS model fitter that works with any SasModels model.

    Features:
    - Loads data from various file formats (CSV, XML, HDF5)
    - Model-agnostic: works with any model from SasModels library
    - Supports multiple fitting engines (BUMPS, LMFit)
    - User-friendly parameter management

    Example:
        >>> fitter = SANSFitter()
        >>> fitter.load_data('my_sans_data.csv')
        >>> fitter.set_model('cylinder')
        >>> fitter.set_param('radius', value=20, min=1, max=100)
        >>> fitter.set_param('length', value=400, min=10, max=1000)
        >>> result = fitter.fit(engine='bumps')
        >>> fitter.plot_results()
    """

    def __init__(self):
        """Initialize the SANS fitter."""
        self.data = None
        self._base_mask = None
        self.kernel = None
        self.model_name = None
        self.params = {}
        self.fit_result = None
        self._fitted_model = None
        self._posterior = None
        self._posterior_base_params = None

        # Structure factor support
        self._structure_factor_name = None
        self._radius_effective_mode = 'unconstrained'
        self._form_factor_params = {}  # Store form factor params separately

    def load_data(self, filename: str) -> None:
        """
        Load SANS data from a file.

        Supports CSV, XML, and HDF5 formats through sasdata.

        Args:
            filename: Path to the data file

        Raises:
            FileNotFoundError: If the file doesn't exist
            ValueError: If the data cannot be loaded or is invalid
        """
        loader = Loader()
        try:
            data_list = loader.load(filename)
            if not data_list:
                raise ValueError(f'No data loaded from {filename}')

            self.data = data_list[0]

            # Setup required fields for sasmodels
            self.data.qmin = getattr(self.data, 'qmin', None) or self.data.x.min()
            self.data.qmax = getattr(self.data, 'qmax', None) or self.data.x.max()
            self._base_mask = (
                np.isnan(self.data.x)
                | np.isnan(self.data.y)
                | np.isnan(self.data.dy)
                | (self.data.dy <= 0)
            )
            self.data.mask = self._base_mask.copy()

            print(f'✓ Loaded data from {filename}')
            print(f'  Q range: {self.data.qmin:.4f} to {self.data.qmax:.4f} Å⁻¹')
            print(f'  Data points: {len(self.data.x)}')

        except Exception as e:
            raise ValueError(f'Failed to load data from {filename}: {str(e)}') from e

    def set_q_range(self, qmin: Optional[float] = None, qmax: Optional[float] = None) -> None:
        """
        Restrict fitting to a q range without discarding loaded data.

        Passing ``None`` for either limit uses the corresponding extent of the
        loaded dataset. Call ``set_q_range()`` with no arguments to restore the
        complete valid range.

        Args:
            qmin: Inclusive lower q limit.
            qmax: Inclusive upper q limit.

        Raises:
            ValueError: If no data is loaded, limits are invalid, or no points
                remain in the requested range.
        """
        if self.data is None or self._base_mask is None:
            raise ValueError('No data loaded. Use load_data() first.')

        data_qmin = float(np.nanmin(self.data.x))
        data_qmax = float(np.nanmax(self.data.x))
        selected_qmin = data_qmin if qmin is None else float(qmin)
        selected_qmax = data_qmax if qmax is None else float(qmax)

        if not np.isfinite(selected_qmin) or not np.isfinite(selected_qmax):
            raise ValueError('q-range limits must be finite.')
        if selected_qmin >= selected_qmax:
            raise ValueError('qmin must be less than qmax.')

        range_mask = (self.data.x < selected_qmin) | (self.data.x > selected_qmax)
        combined_mask = self._base_mask | range_mask
        if np.all(combined_mask):
            raise ValueError('The requested q range contains no valid data points.')

        self.data.mask = combined_mask
        self.data.qmin = selected_qmin
        self.data.qmax = selected_qmax
        self.fit_result = None
        self._fitted_model = None

        included = np.count_nonzero(~combined_mask)
        print(f'✓ Using q range {selected_qmin:.4g} to {selected_qmax:.4g} Å⁻¹ ({included} points)')

    def set_model(self, model_name: str, platform: str = 'cpu') -> None:
        """
        Set the SANS model to use for fitting.

        This resets any active structure factor to ensure a clean state.

        Args:
            model_name: Name of the model from SasModels (e.g., 'cylinder', 'sphere')
            platform: Computation platform ('cpu' or 'opencl')

        Raises:
            ValueError: If the model name is not valid
        """
        try:
            # Reset structure factor when changing form factor
            self._structure_factor_name = None
            self._radius_effective_mode = 'unconstrained'
            self._form_factor_params = {}

            # Force CPU platform to avoid OpenCL issues
            self.kernel = load_model(model_name, dtype='double', platform='dll')
            self.model_name = model_name

            # Initialize parameters with default values from the model
            self.params = {}
            for param in self.kernel.info.parameters.kernel_parameters:
                self.params[param.name] = {
                    'value': param.default,
                    'min': param.limits[0] if param.limits[0] > -np.inf else 0,
                    'max': param.limits[1] if param.limits[1] < np.inf else param.default * 10,
                    'vary': False,  # By default, parameters are fixed
                    'description': param.description,
                }

            # Add implicit scale and background parameters (present in all models)
            # These are not in kernel_parameters but are always available
            if 'scale' not in self.params:
                self.params['scale'] = {
                    'value': 1.0,
                    'min': 0.0,
                    'max': np.inf,
                    'vary': False,
                    'description': 'Scale factor for the model intensity',
                }

            if 'background' not in self.params:
                self.params['background'] = {
                    'value': 0.0,
                    'min': 0.0,
                    'max': np.inf,
                    'vary': False,
                    'description': 'Constant background level',
                }

            print(f"✓ Model '{model_name}' loaded successfully")
            print(f'  Available parameters: {len(self.params)}')

        except Exception as e:
            raise ValueError(f"Failed to load model '{model_name}': {str(e)}") from e

    def get_params(self) -> None:
        """Display current parameter values and settings in a readable format."""
        if not self.params:
            print('No model loaded. Use set_model() first.')
            return

        print(f'\n{"=" * 80}')
        print(f'Model: {self.model_name}')
        if self._structure_factor_name:
            print(f'Structure Factor: {self._structure_factor_name}')
            print(f'Radius Effective Mode: {self._radius_effective_mode}')
        print(f'{"=" * 80}')
        print(f'{"Parameter":<20} {"Value":<12} {"Min":<12} {"Max":<12} {"Vary":<8}')
        print(f'{"-" * 80}')

        for name, info in self.params.items():
            vary_str = '✓' if info['vary'] else '✗'
            # Show linked indicator for radius_effective in link_radius mode
            if name == 'radius_effective' and self._radius_effective_mode == 'link_radius':
                vary_str = '→radius'
            print(
                f'{name:<20} {info["value"]:<12.4g} {info["min"]:<12.4g} '
                f'{info["max"]:<12.4g} {vary_str:<8}'
            )
        print(f'{"=" * 80}\n')

    def set_param(
        self,
        name: str,
        value: Optional[float] = None,
        min: Optional[float] = None,
        max: Optional[float] = None,
        vary: Optional[bool] = None,
    ) -> None:
        """
        Configure a model parameter for fitting.

        Args:
            name: Parameter name
            value: Initial value (optional)
            min: Minimum bound (optional)
            max: Maximum bound (optional)
            vary: Whether to vary during fit (optional)

        Raises:
            KeyError: If parameter name doesn't exist for the current model
        """
        if name not in self.params:
            available = ', '.join(self.params.keys())
            raise KeyError(f"Parameter '{name}' not found. Available: {available}")

        if value is not None:
            self.params[name]['value'] = value
            # Sync radius_effective when radius is updated in link_radius mode
            if (
                name == 'radius'
                and self._radius_effective_mode == 'link_radius'
                and 'radius_effective' in self.params
            ):
                self.params['radius_effective']['value'] = value
        if min is not None:
            self.params[name]['min'] = min
        if max is not None:
            self.params[name]['max'] = max
        if vary is not None:
            self.params[name]['vary'] = vary

    def set_structure_factor(
        self, structure_factor_name: str, radius_effective_mode: str = 'unconstrained'
    ) -> None:
        """
        Apply a structure factor to the current model.

        This creates a product model (form_factor * structure_factor) to account
        for inter-particle interactions in concentrated systems.

        Supported structure factors:
        - 'hardsphere': Hard sphere structure factor (Percus-Yevick closure)
        - 'hayter_msa': Hayter-Penfold rescaled MSA for charged spheres
        - 'squarewell': Square well potential
        - 'stickyhardsphere': Sticky hard sphere (Baxter model)

        Args:
            structure_factor_name: Name of the structure factor (e.g., 'hardsphere')
            radius_effective_mode: How to handle the effective radius.
                - 'unconstrained': 'radius_effective' is a separate fitting parameter.
                - 'link_radius': 'radius_effective' is constrained to the form factor's 'radius'.

        Raises:
            ValueError: If no form factor model is set, or if the structure factor is invalid
        """
        if self.kernel is None or self.model_name is None:
            raise ValueError('No form factor model loaded. Use set_model() first.')

        # Validate structure factor name
        supported_sf = ['hardsphere', 'hayter_msa', 'squarewell', 'stickyhardsphere']
        if structure_factor_name not in supported_sf:
            raise ValueError(
                f"Unsupported structure factor '{structure_factor_name}'. "
                f'Supported: {", ".join(supported_sf)}'
            )

        # Validate radius_effective_mode
        if radius_effective_mode not in ['unconstrained', 'link_radius']:
            raise ValueError(
                f"Invalid radius_effective_mode '{radius_effective_mode}'. "
                "Use 'unconstrained' or 'link_radius'."
            )

        # Store form factor parameters before switching to product model
        if not self._form_factor_params:
            self._form_factor_params = {k: dict(v) for k, v in self.params.items()}

        # Create product model name
        full_model_name = f'{self.model_name}@{structure_factor_name}'

        try:
            # Load the product model
            self.kernel = load_model(full_model_name, dtype='double', platform='dll')
            self._structure_factor_name = structure_factor_name
            self._radius_effective_mode = radius_effective_mode

            # Rebuild parameters from product model
            new_params = {}
            for param in self.kernel.info.parameters.kernel_parameters:
                # Preserve existing values if parameter already exists
                if param.name in self._form_factor_params:
                    new_params[param.name] = dict(self._form_factor_params[param.name])
                else:
                    new_params[param.name] = {
                        'value': param.default,
                        'min': param.limits[0] if param.limits[0] > -np.inf else 0,
                        'max': param.limits[1] if param.limits[1] < np.inf else param.default * 10,
                        'vary': False,
                        'description': param.description,
                    }

            # Ensure scale and background are present
            if 'scale' not in new_params:
                if 'scale' in self._form_factor_params:
                    new_params['scale'] = dict(self._form_factor_params['scale'])
                else:
                    new_params['scale'] = {
                        'value': 1.0,
                        'min': 0.0,
                        'max': np.inf,
                        'vary': False,
                        'description': 'Scale factor for the model intensity',
                    }

            if 'background' not in new_params:
                if 'background' in self._form_factor_params:
                    new_params['background'] = dict(self._form_factor_params['background'])
                else:
                    new_params['background'] = {
                        'value': 0.0,
                        'min': 0.0,
                        'max': np.inf,
                        'vary': False,
                        'description': 'Constant background level',
                    }

            self.params = new_params

            if 'radius_effective_mode' in self.params:
                self.params['radius_effective_mode']['value'] = 0
                self.params['radius_effective_mode']['vary'] = False

            # Handle radius_effective linking
            if radius_effective_mode == 'link_radius':
                if 'radius' in self.params and 'radius_effective' in self.params:
                    # Link radius_effective to radius
                    self.params['radius_effective']['value'] = self.params['radius']['value']
                    self.params['radius_effective']['vary'] = False
                    print("  Note: 'radius_effective' linked to 'radius' value")
                else:
                    warnings.warn(
                        'Cannot link radius_effective to radius: one or both parameters not found. '
                        'Using unconstrained mode.',
                        stacklevel=2,
                    )
                    self._radius_effective_mode = 'unconstrained'

            print(f"✓ Structure factor '{structure_factor_name}' applied to '{self.model_name}'")
            print(f'  Product model: {full_model_name}')
            print(f'  Total parameters: {len(self.params)}')

        except Exception as e:
            raise ValueError(f"Failed to load model '{full_model_name}': {str(e)}") from e

    def remove_structure_factor(self) -> None:
        """
        Remove the current structure factor and revert to the form factor only.

        Raises:
            ValueError: If no structure factor is currently set
        """
        if self._structure_factor_name is None:
            raise ValueError('No structure factor is currently set.')

        # Reload the original form factor model
        try:
            self.kernel = load_model(self.model_name, dtype='single', platform='dll')

            # Restore form factor parameters
            self.params = {k: dict(v) for k, v in self._form_factor_params.items()}

            sf_name = self._structure_factor_name
            self._structure_factor_name = None
            self._radius_effective_mode = 'unconstrained'
            self._form_factor_params = {}

            print(f"✓ Structure factor '{sf_name}' removed")
            print(f'  Reverted to form factor: {self.model_name}')

        except Exception as e:
            raise ValueError(f'Failed to reload form factor model: {str(e)}') from e

    def get_structure_factor(self) -> Optional[str]:
        """
        Get the name of the currently applied structure factor.

        Returns:
            Name of the structure factor, or None if no structure factor is set
        """
        return self._structure_factor_name

    def fit(
        self,
        engine: Literal['bumps', 'lmfit'] = 'bumps',
        method: Optional[str] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Perform the fit using the specified engine.

        Args:
            engine: Fitting engine ('bumps' or 'lmfit')
            method: Optimization method (engine-specific)
                   - BUMPS: 'amoeba', 'lm', 'newton', 'de' (default: 'amoeba')
                   - LMFit: 'leastsq', 'least_squares', 'differential_evolution', etc.
            **kwargs: Additional arguments passed to the fitting engine

        Returns:
            Dictionary with fit results including chi-squared and parameter values

        Raises:
            ValueError: If data or model not loaded, or invalid engine
        """
        if self.data is None:
            raise ValueError('No data loaded. Use load_data() first.')
        if self.kernel is None:
            raise ValueError('No model loaded. Use set_model() first.')

        if engine == 'bumps':
            return self._fit_bumps(method or 'amoeba', **kwargs)
        elif engine == 'lmfit':
            if not LMFIT_AVAILABLE:
                raise ValueError("scipy is not installed. Use 'bumps' engine or install scipy.")
            return self._fit_lmfit(method or 'leastsq', **kwargs)
        else:
            raise ValueError(f"Unknown engine '{engine}'. Use 'bumps' or 'lmfit'.")

    def _fit_bumps(self, method: str = 'amoeba', **kwargs: Any) -> dict[str, Any]:
        """Fit using BUMPS engine."""
        _, _, problem = self._build_bumps_problem()

        print(f'\nInitial χ² = {problem.chisq():.4f}')
        print(f'Fitting with BUMPS (method: {method})...')

        # Perform fit
        result = bumps_fit(problem, method=method, **kwargs)

        # Store results
        self.fit_result = {
            'engine': 'bumps',
            'method': method,
            'chisq': problem.chisq(),
            'parameters': {},
            'problem': problem,
            'result': result,
        }

        # Extract fitted parameters
        for k, v, dv in zip(problem.labels(), result.x, result.dx):
            self.fit_result['parameters'][k] = {
                'value': v,
                'stderr': dv,
                'formatted': format_uncertainty(v, dv),
            }
            # Update internal parameter values
            if k in self.params:
                self.params[k]['value'] = v

        self._fitted_model = problem
        self._posterior = None
        self._posterior_base_params = None

        # Print results
        print('\n✓ Fit completed!')
        print(f'Final χ² = {self.fit_result["chisq"]:.4f}')
        print('\nFitted parameters:')
        for name, info in self.fit_result['parameters'].items():
            print(f'  {name}: {info["formatted"]}')

        return self.fit_result

    def _build_bumps_problem(self) -> tuple[BumpsModel, Experiment, FitProblem]:
        """Build a BUMPS model, experiment, and fit problem from current state."""
        # Prepare parameter dictionary for BumpsModel
        pars = {name: info['value'] for name, info in self.params.items()}

        # Create BUMPS model
        model = BumpsModel(self.kernel, **pars)

        # Set parameter ranges for fitting
        for name, info in self.params.items():
            if info['vary']:
                param_obj = getattr(model, name)
                param_obj.range(info['min'], info['max'])

        # Handle radius_effective linking in link_radius mode
        if (
            self._radius_effective_mode == 'link_radius'
            and hasattr(model, 'radius_effective')
            and hasattr(model, 'radius')
        ):
            # Constrain radius_effective to equal radius
            model.radius_effective = model.radius

        # Create experiment and fit problem
        experiment = Experiment(data=self.data, model=model)
        problem = FitProblem(experiment)
        return model, experiment, problem

    def fit_bayesian(
        self,
        samples: int = 2000,
        burn: int = 100,
        thin: int = 1,
        pop: int = 10,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Sample the posterior with BUMPS' DREAM algorithm."""
        if self.data is None:
            raise ValueError('No data loaded. Use load_data() first.')
        if self.kernel is None:
            raise ValueError('No model loaded. Use set_model() first.')

        _, experiment, problem = self._build_bumps_problem()
        labels = list(problem.labels())
        if not labels:
            raise ValueError(
                'Bayesian fitting requires a varying parameter. '
                'Use set_param(..., vary=True) first.'
            )

        print(f'\nInitial χ² = {problem.chisq():.4f}')
        print(f'Sampling with DREAM (samples={samples}, burn={burn}, thin={thin})...')
        result = bumps_fit(
            problem,
            method='dream',
            samples=samples,
            burn=burn,
            thin=thin,
            pop=pop,
            **kwargs,
        )
        state = getattr(result, 'state', None)
        if state is None or not hasattr(state, 'draw'):
            raise RuntimeError('BUMPS did not return a DREAM posterior state.')

        draw = state.draw()
        posterior_samples = np.asarray(draw.points, dtype=float)
        if posterior_samples.ndim != 2 or posterior_samples.shape[0] < 2:
            raise RuntimeError('DREAM returned too few posterior samples.')
        if posterior_samples.shape[1] != len(labels):
            raise RuntimeError('Posterior columns do not match varying parameters.')

        logp = getattr(draw, 'logp', None)
        logp = None if logp is None else np.asarray(logp, dtype=float)
        chains = None
        try:
            _, chain_values, _ = state.chains()
            chain_values = np.asarray(chain_values, dtype=float)
            if chain_values.ndim == 3 and chain_values.shape[-1] == len(labels):
                chains = chain_values
        except (AttributeError, TypeError, ValueError):
            pass

        means = posterior_samples.mean(axis=0)
        medians = np.median(posterior_samples, axis=0)
        stds = posterior_samples.std(axis=0, ddof=1)
        percentiles = np.percentile(posterior_samples, [16, 84, 2.5, 97.5], axis=0)
        r_hat = self._gelman_rubin(chains)
        diagnostics = {
            name: {'r_hat': float(r_hat[index])}
            for index, name in enumerate(labels)
            if r_hat is not None
        }

        self._posterior = PosteriorSummary(
            labels=labels,
            samples=posterior_samples,
            logp=logp,
            chains=chains,
            best={name: float(result.x[index]) for index, name in enumerate(labels)},
            mean={name: float(means[index]) for index, name in enumerate(labels)},
            median={name: float(medians[index]) for index, name in enumerate(labels)},
            std={name: float(stds[index]) for index, name in enumerate(labels)},
            ci_68={
                name: (float(percentiles[0, index]), float(percentiles[1, index]))
                for index, name in enumerate(labels)
            },
            ci_95={
                name: (float(percentiles[2, index]), float(percentiles[3, index]))
                for index, name in enumerate(labels)
            },
            diagnostics=diagnostics,
        )
        self._posterior_base_params = {name: info['value'] for name, info in self.params.items()}

        problem.setp(result.x)
        self.fit_result = {
            'engine': 'bumps',
            'method': 'dream',
            'chisq': problem.chisq(),
            'parameters': {},
            'problem': problem,
            'result': result,
            'posterior': self._posterior,
        }
        for index, name in enumerate(labels):
            value = float(result.x[index])
            stderr = float(stds[index])
            self.params[name]['value'] = value
            self._posterior_base_params[name] = value
            self.fit_result['parameters'][name] = {
                'value': value,
                'stderr': stderr,
                'formatted': format_uncertainty(value, stderr),
            }

        self._fitted_model = problem
        print('\n✓ Bayesian fit completed!')
        print(f'Final χ² = {self.fit_result["chisq"]:.4f}')
        for name in labels:
            low, high = self._posterior.ci_95[name]
            print(f'  {name}: 95% credible interval [{low:.6g}, {high:.6g}]')
        return self.fit_result

    @staticmethod
    def _gelman_rubin(chains: Optional[np.ndarray]) -> Optional[np.ndarray]:
        """Calculate split-chain R-hat when enough DREAM history is available."""
        if chains is None or chains.shape[0] < 4 or chains.shape[1] < 2:
            return None
        generations, chain_count, _ = chains.shape
        values = chains[generations // 2 :, :, :]
        sample_count = values.shape[0]
        chain_means = values.mean(axis=0)
        between = sample_count * chain_means.var(axis=0, ddof=1)
        within = values.var(axis=0, ddof=1).mean(axis=0)
        variance = ((sample_count - 1) / sample_count) * within + between / sample_count
        with np.errstate(divide='ignore', invalid='ignore'):
            return np.sqrt(variance / within)

    def _fit_lmfit(self, method: str = 'leastsq', **kwargs: Any) -> dict[str, Any]:
        """Fit using scipy.optimize (leastsq/least_squares) engine."""
        # Get initial parameter values and build bounds
        param_names = [name for name, info in self.params.items() if info['vary']]
        x0 = np.array([self.params[name]['value'] for name in param_names])
        bounds_lower = np.array([self.params[name]['min'] for name in param_names])
        bounds_upper = np.array([self.params[name]['max'] for name in param_names])

        # Create direct model calculator (kernel already set to CPU in set_model)
        calculator = DirectModel(self.data, self.kernel)

        # Capture instance attributes for use in residual closure
        radius_effective_mode = self._radius_effective_mode

        # Define residual function
        def residual(x):
            # Build full parameter dictionary
            par_dict = {name: info['value'] for name, info in self.params.items()}
            # Update with fitted parameters
            for i, name in enumerate(param_names):
                par_dict[name] = x[i]

            # Handle radius_effective linking in link_radius mode
            if (
                radius_effective_mode == 'link_radius'
                and 'radius' in par_dict
                and 'radius_effective' in par_dict
            ):
                par_dict['radius_effective'] = par_dict['radius']

            # Calculate model
            I_calc = calculator(**par_dict)
            # Return weighted residuals
            valid = ~self.data.mask
            return (self.data.y[valid] - I_calc) / self.data.dy[valid]

        print(f'\nFitting with scipy.optimize (method: {method})...')

        # Perform fit based on method
        if method == 'leastsq':
            # Levenberg-Marquardt (no bounds support)
            result = leastsq(residual, x0, full_output=True, **kwargs)
            fitted_params = result[0]
            cov_matrix = result[1]
            result[2]

            # Calculate parameter errors from covariance matrix
            if cov_matrix is not None:
                param_errors = np.sqrt(np.diag(cov_matrix))
            else:
                param_errors = np.zeros_like(fitted_params)

            # Calculate chi-squared
            final_residuals = residual(fitted_params)
            chisq = np.sum(final_residuals**2)

        elif method == 'least_squares':
            # Trust Region Reflective (supports bounds)
            bounds = (bounds_lower, bounds_upper)
            result = least_squares(residual, x0, bounds=bounds, **kwargs)
            fitted_params = result.x

            # Estimate parameter errors from Jacobian
            jacobian = result.jac
            covariance = np.linalg.pinv(jacobian.T @ jacobian)
            degrees_of_freedom = max(1, len(result.fun) - len(fitted_params))
            residual_variance = np.sum(result.fun**2) / degrees_of_freedom
            param_errors = np.sqrt(np.maximum(0, np.diag(covariance) * residual_variance))

            chisq = np.sum(result.fun**2)

        elif method == 'differential_evolution':
            # Global optimizer (supports bounds)
            bounds_list = list(zip(bounds_lower, bounds_upper))

            def objective(x):
                return np.sum(residual(x) ** 2)

            result = differential_evolution(objective, bounds_list, **kwargs)
            fitted_params = result.x
            param_errors = np.zeros_like(fitted_params)  # DE doesn't provide errors
            chisq = result.fun

        else:
            raise ValueError(
                f"Unknown method '{method}'. Use 'leastsq', 'least_squares', or 'differential_evolution'."
            )

        # Store results
        self.fit_result = {
            'engine': 'lmfit',
            'method': method,
            'chisq': chisq,
            'parameters': {},
            'result': result,
        }

        # Extract fitted parameters
        for i, name in enumerate(param_names):
            self.fit_result['parameters'][name] = {
                'value': fitted_params[i],
                'stderr': param_errors[i],
                'formatted': f'{fitted_params[i]:.6g} ± {param_errors[i]:.6g}'
                if param_errors[i] > 0
                else f'{fitted_params[i]:.6g}',
            }
            # Update internal parameter values
            self.params[name]['value'] = fitted_params[i]

        # Add fixed parameters to results
        for name, info in self.params.items():
            if name not in param_names:
                self.fit_result['parameters'][name] = {
                    'value': info['value'],
                    'stderr': 0.0,
                    'formatted': f'{info["value"]:.6g} (fixed)',
                }

        self._fitted_model = result

        # Print results
        print('\n✓ Fit completed!')
        print(f'Final χ² = {self.fit_result["chisq"]:.4f}')
        print('\nFitted parameters:')
        for name, info in self.fit_result['parameters'].items():
            print(f'  {name}: {info["formatted"]}')

        return self.fit_result

    def plot_results(self, show_residuals: bool = True, log_scale: bool = True) -> None:
        """
        Plot experimental data and fitted model.

        Args:
            show_residuals: If True, show residuals in a separate panel
            log_scale: If True, use log scale for both axes
        """
        if self.data is None:
            raise ValueError('No data to plot. Use load_data() first.')

        if self.fit_result is None:
            print('No fit results available. Plotting data only.')
            plt.figure(figsize=(10, 6))
            plt.errorbar(
                self.data.x, self.data.y, yerr=self.data.dy, fmt='o', label='Data', alpha=0.6
            )
            plt.xlabel('Q (Å⁻¹)')
            plt.ylabel('I(Q)')
            plt.title('SANS Data')
            if log_scale:
                plt.xscale('log')
                plt.yscale('log')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()
            return

        q, intensity, uncertainty, I_fit = self._fit_arrays()
        residuals = (intensity - I_fit) / uncertainty

        # Create plot
        if show_residuals:
            fig, (ax1, ax2) = plt.subplots(
                2, 1, figsize=(10, 10), gridspec_kw={'height_ratios': [3, 1]}
            )
        else:
            fig, ax1 = plt.subplots(1, 1, figsize=(10, 6))

        # Main plot
        ax1.errorbar(
            self.data.x,
            self.data.y,
            yerr=self.data.dy,
            fmt='o',
            label='Experimental Data',
            alpha=0.6,
            markersize=4,
        )
        ax1.plot(q, I_fit, 'r-', label='Fitted Model', linewidth=2)
        ax1.set_xlabel('Q (Å⁻¹)', fontsize=12)
        ax1.set_ylabel('I(Q)', fontsize=12)
        ax1.set_title(
            f'SANS Fit: {self.model_name} (χ² = {self.fit_result["chisq"]:.4f})', fontsize=14
        )
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        if log_scale:
            ax1.set_xscale('log')
            ax1.set_yscale('log')

        if show_residuals:
            ax2.axhline(0, color='gray', linestyle='--', linewidth=1)
            ax2.plot(q, residuals, 'o', markersize=4, alpha=0.6)
            ax2.set_xlabel('Q (Å⁻¹)', fontsize=12)
            ax2.set_ylabel('Residuals (σ)', fontsize=12)
            ax2.grid(True, alpha=0.3)
            if log_scale:
                ax2.set_xscale('log')

        plt.tight_layout()
        plt.show()

    def _fit_arrays(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return selected q, intensity, uncertainty, and fitted intensity arrays."""
        valid = ~self.data.mask
        if self.fit_result['engine'] == 'bumps':
            experiment = next(iter(self._fitted_model.models))
            I_fit = np.asarray(experiment.theory())
        else:
            calculator = DirectModel(self.data, self.kernel)
            par_dict = {name: info['value'] for name, info in self.fit_result['parameters'].items()}
            I_fit = np.asarray(calculator(**par_dict))

        selected_q = self.data.x[valid]
        selected_intensity = self.data.y[valid]
        selected_uncertainty = self.data.dy[valid]
        if I_fit.shape == self.data.y.shape:
            I_fit = I_fit[valid]
        if I_fit.shape != selected_intensity.shape:
            raise ValueError('Fitted curve shape does not match the selected data range.')
        return selected_q, selected_intensity, selected_uncertainty, I_fit

    def get_posterior(self) -> PosteriorSummary:
        """Return samples and summaries from the latest Bayesian fit."""
        if self._posterior is None:
            raise ValueError('No posterior available. Run fit_bayesian() first.')
        return self._posterior

    def plot_posterior_pairs(
        self, params: Optional[list[str]] = None, show_contours: bool = True
    ) -> plt.Figure:
        """Plot marginal and pairwise posterior distributions."""
        posterior = self.get_posterior()
        labels = posterior.labels if params is None else params
        indices = [posterior.index_of(name) for name in labels]
        count = len(labels)
        fig, axes = plt.subplots(count, count, figsize=(3 * count, 3 * count), squeeze=False)
        samples = posterior.samples[:, indices]
        stride = max(1, len(samples) // 2000)

        for row in range(count):
            for column in range(count):
                axis = axes[row, column]
                if row == column:
                    axis.hist(samples[:, column], bins=40, density=True, alpha=0.75)
                elif row > column:
                    if show_contours:
                        axis.hist2d(samples[:, column], samples[:, row], bins=35, cmap='Blues')
                    else:
                        axis.plot(
                            samples[::stride, column],
                            samples[::stride, row],
                            '.',
                            alpha=0.2,
                        )
                else:
                    axis.set_visible(False)
                    continue
                if row == count - 1:
                    axis.set_xlabel(labels[column])
                if column == 0 and row > 0:
                    axis.set_ylabel(labels[row])

        fig.suptitle('Posterior parameter pairs')
        fig.tight_layout()
        plt.show()
        return fig

    def plot_param_distribution(self, param: str, bins: int = 50) -> plt.Figure:
        """Plot one marginal posterior distribution and its credible intervals."""
        posterior = self.get_posterior()
        values = posterior.samples[:, posterior.index_of(param)]
        low_68, high_68 = posterior.ci_68[param]
        low_95, high_95 = posterior.ci_95[param]

        fig, axis = plt.subplots(figsize=(8, 5))
        axis.hist(values, bins=bins, density=True, alpha=0.7)
        axis.axvspan(low_95, high_95, alpha=0.15, label='95% credible interval')
        axis.axvspan(low_68, high_68, alpha=0.3, label='68% credible interval')
        axis.axvline(posterior.median[param], color='black', linestyle='--', label='median')
        axis.set(xlabel=param, ylabel='Posterior density', title=f'Posterior: {param}')
        axis.legend()
        fig.tight_layout()
        plt.show()
        return fig

    def plot_param_correlations(self, threshold: float = 0.0) -> plt.Figure:
        """Plot the posterior parameter correlation matrix."""
        posterior = self.get_posterior()
        correlation = np.atleast_2d(np.corrcoef(posterior.samples, rowvar=False))
        displayed = correlation.copy()
        displayed[np.abs(displayed) < threshold] = np.nan

        fig, axis = plt.subplots(figsize=(7, 6))
        image = axis.imshow(displayed, vmin=-1, vmax=1, cmap='coolwarm')
        positions = np.arange(len(posterior.labels))
        axis.set_xticks(positions, posterior.labels, rotation=45, ha='right')
        axis.set_yticks(positions, posterior.labels)
        axis.set_title('Posterior parameter correlations')
        fig.colorbar(image, ax=axis, label='Correlation')
        fig.tight_layout()
        plt.show()
        return fig

    def plot_trace(self, params: Optional[list[str]] = None) -> plt.Figure:
        """Plot DREAM chains to support a visual convergence check."""
        posterior = self.get_posterior()
        labels = posterior.labels if params is None else params
        indices = [posterior.index_of(name) for name in labels]
        fig, axes = plt.subplots(len(labels), 1, figsize=(10, 2.5 * len(labels)), squeeze=False)

        for axis, name, index in zip(axes[:, 0], labels, indices):
            if posterior.chains is None:
                axis.plot(posterior.samples[:, index], linewidth=0.8)
            else:
                for chain in range(posterior.chains.shape[1]):
                    axis.plot(posterior.chains[:, chain, index], linewidth=0.6, alpha=0.7)
            axis.set(ylabel=name)
            axis.grid(alpha=0.2)
        axes[-1, 0].set_xlabel('Generation')
        fig.suptitle('DREAM chain traces')
        fig.tight_layout()
        plt.show()
        return fig

    def plot_posterior_predictive(
        self, style: str = 'band', n_draws: int = 50, log_scale: bool = True
    ) -> plt.Figure:
        """Plot model curves propagated from posterior parameter samples."""
        if style not in {'band', 'draws', 'band+draws'}:
            raise ValueError("style must be 'band', 'draws', or 'band+draws'.")
        posterior = self.get_posterior()
        if self._posterior_base_params is None:
            raise ValueError('Posterior model state is unavailable.')
        if n_draws < 1:
            raise ValueError('n_draws must be at least 1.')

        calculator = DirectModel(self.data, self.kernel)
        draw_count = min(n_draws, len(posterior.samples))
        draw_indices = np.linspace(0, len(posterior.samples) - 1, draw_count, dtype=int)
        curves = []
        for sample_index in draw_indices:
            parameters = dict(self._posterior_base_params)
            parameters.update(
                {
                    name: posterior.samples[sample_index, column]
                    for column, name in enumerate(posterior.labels)
                }
            )
            if (
                self._radius_effective_mode == 'link_radius'
                and 'radius' in parameters
                and 'radius_effective' in parameters
            ):
                parameters['radius_effective'] = parameters['radius']
            curves.append(np.asarray(calculator(**parameters)))

        predictions = np.asarray(curves)
        low, median, high = np.percentile(predictions, [2.5, 50, 97.5], axis=0)
        valid = ~self.data.mask
        q = self.data.x[valid]

        fig, axis = plt.subplots(figsize=(9, 6))
        axis.errorbar(
            q,
            self.data.y[valid],
            yerr=self.data.dy[valid],
            fmt='o',
            ms=3,
            alpha=0.6,
            label='Data',
        )
        if style in {'draws', 'band+draws'}:
            for curve in predictions:
                axis.plot(q, curve, color='tab:blue', alpha=0.08)
        if style in {'band', 'band+draws'}:
            axis.fill_between(q, low, high, alpha=0.25, label='95% credible band')
        axis.plot(q, median, color='tab:red', label='Posterior median')
        axis.set(xlabel='Q (Å⁻¹)', ylabel='I(Q)', title='Posterior predictive check')
        if log_scale:
            axis.set_xscale('log')
            axis.set_yscale('log')
        axis.legend()
        fig.tight_layout()
        plt.show()
        return fig

    def save_results(self, filename: str) -> None:
        """
        Save fit results to a file.

        Args:
            filename: Output file path (CSV format)
        """
        if self.fit_result is None:
            raise ValueError('No fit results to save. Run fit() first.')

        # Prepare data
        with open(filename, 'w') as f:
            f.write('# SANS Fit Results\n')
            f.write(f'# Model: {self.model_name}\n')
            f.write(f'# Engine: {self.fit_result["engine"]}\n')
            f.write(f'# Method: {self.fit_result["method"]}\n')
            f.write(f'# Chi-squared: {self.fit_result["chisq"]:.6f}\n')
            f.write('#\n')
            f.write('# Fitted Parameters:\n')
            for name, info in self.fit_result['parameters'].items():
                f.write(f'# {name}: {info["formatted"]}\n')
                if self._posterior is not None and name in self._posterior.ci_95:
                    low, high = self._posterior.ci_95[name]
                    f.write(f'# {name} 95% credible interval: [{low:.6g}, {high:.6g}]\n')
            f.write('#\n')
            f.write('Q,I_exp,dI_exp,I_fit,Residuals\n')

            q, intensity, uncertainty, I_fit = self._fit_arrays()
            residuals = (intensity - I_fit) / uncertainty

            for q_value, i_exp, di_exp, i_fit, res in zip(
                q, intensity, uncertainty, I_fit, residuals
            ):
                f.write(f'{q_value:.6e},{i_exp:.6e},{di_exp:.6e},{i_fit:.6e},{res:.6e}\n')

        print(f'✓ Results saved to {filename}')
