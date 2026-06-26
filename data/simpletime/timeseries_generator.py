import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Dict, Union
from scipy import signal, stats
import math
from tqdm import tqdm

def register_distribution(name: str):
    """Decorator to register a generation method with a distribution name."""

    def decorator(func):
        func._dist_name = name
        return func

    return decorator


class TimeSeriesGenerator:
    """
    A class to generate synthetic time series samples from different distributions.
    """

    def __init__(
        self,
        random_state: int = None,
        start_date: str = "2000-01-01",
        frequency: str = "H",
    ):
        """
        Initializes the TimeSeriesGenerator with an optional random state for reproducibility.

        Args:
            random_state (int, optional): Seed for the random number generator. Defaults to None.
        """
        self.random_state = random_state
        self.rng = np.random.default_rng(random_state)
        self.start_date = pd.to_datetime(start_date)
        self.frequency = frequency

        self._distribution_map = {}
        for attr_name in dir(self):
            attr = getattr(self, attr_name)
            if callable(attr) and hasattr(attr, "_dist_name"):
                self._distribution_map[attr._dist_name] = attr

        print(
            f"TimeSeriesGenerator initialized with random_state={random_state}, start_date={start_date}, frequency={frequency}"
        )

    def _seed_rng(self, sample_idx: int):
        """Re-seeds the RNG for a specific sample index if a base seed is provided."""
        if self.random_state is not None:
            # Use SeedSequence to derive a robust child seed
            ss = np.random.SeedSequence(self.random_state)
            child_seeds = ss.spawn(sample_idx + 1)
            self.rng = np.random.default_rng(child_seeds[sample_idx])

    def add_effects(self, data: np.ndarray, params: Dict) -> np.ndarray:
        """
        Adds optional effects like noise, seasonality, and trend to the generated data.

        Args:
            data (np.ndarray): The original time series data.
            params (Dict): A dictionary of parameters specifying which effects to add and their configurations.
        Returns:
            np.ndarray: The modified time series data with added effects.
        """
        length = len(data)

        if params.get("normalize", False):
            if params.get("normalization_method", "minmax") == "minmax":
                data = (data - np.min(data)) / (np.max(data) - np.min(data) + 1e-8)
            elif params.get("normalization_method") == "meanstd":
                data = (data - np.mean(data)) / (np.std(data) + 1e-8)
            else:
                raise ValueError(
                    f"Unsupported normalization method: {params.get('normalization_method')}"
                )

        if params.get("add_seasonality", False):
            seasonality_period = params.get("seasonality_period", 16)
            seasonality_amplitude = params.get("seasonality_amplitude", 0.1)
            seasonal_component = seasonality_amplitude * np.sin(
                2 * np.pi * np.arange(length) / seasonality_period
            )
            data = data + seasonal_component

        if params.get("add_noise", False):
            noise_level = params.get("noise_level", 0.05)
            noise = self.rng.normal(loc=0, scale=noise_level, size=length)
            data = data + noise

        if params.get("add_trend", False):
            trend_slope = params.get("trend_slope", self.rng.choice([1, 1]))
            trend = np.arange(length) / length * trend_slope
            data = data + trend

        if params.get("add_spikes", False):
            spike_magnitude = params.get("spike_magnitude", 0.5)
            average_spike_interval = params.get("average_spike_interval", 16)
            num_spikes = int(
                self.rng.normal(
                    length / average_spike_interval, np.sqrt(average_spike_interval)
                )
            )

            spike_indices = self.rng.choice(length, size=num_spikes, replace=False)
            data[spike_indices] += spike_magnitude * self.rng.uniform(
                0.5, 1.5, size=num_spikes
            )

        return data

    # --- 1. Deterministic Trends ---

    @register_distribution("constant")
    def _generate_constant(self, length: int, params: Dict) -> np.ndarray:
        """Generates a constant value time series."""
        value = params.get("value", self.rng.uniform(-1, 1))
        return np.full(length, value)

    @register_distribution("linear")
    def _generate_linear(self, length: int, params: Dict) -> np.ndarray:
        """Generates a linear trend time series."""
        slope = params.get("slope", self.rng.uniform(-1, 1))
        intercept = params.get("intercept", self.rng.uniform(-1, 1))
        return np.arange(length) * slope + intercept

    @register_distribution("staircase")
    def _generate_staircase(self, length: int, params: Dict) -> np.ndarray:
        """Generates a staircase-like step function."""
        min_run = max(2, int(length * 0.025))
        max_run = max(min_run + 2, int(length * 0.2))

        run = params.get("run", self.rng.integers(min_run, max_run))
        down = params.get("down", self.rng.choice([True, False]))
        n_steps = int(np.ceil(length / run))
        data = np.repeat(np.arange(n_steps), run)[:length]
        if down:
            data = data[::-1]
        return data

    @register_distribution("power_law")
    def _generate_power_law(self, length: int, params: Dict) -> np.ndarray:
        """Generates a power law trend: c * x^k where k is an integer."""
        # Draw the integer exponent k (defaulting to 2, 3, or 4)
        exponent = params.get("exponent", self.rng.integers(2, 5))
        
        # Coefficient c (scaling factor)
        coefficient = params.get("coefficient", self.rng.uniform(0, 1))
        
        # Normalized x range [0, 1] to keep the series within 
        # a manageable scale regardless of length.
        x = np.linspace(0, 1, length)
        return coefficient * (x ** exponent)

    @register_distribution("exponential")
    def _generate_exponential(self, length: int, params: Dict) -> np.ndarray:
        """Generates an increasing exponential growth time series."""
        base = params.get("base", self.rng.uniform(1.1, 3))

        # If the user manually passed a base < 1, you might want to invert it
        if base < 1:
            base = 1 / base

        min_exponent = max(3, params.get("min_exponent", 3))
        max_exponent = max(min_exponent + 3, params.get("max_exponent", 10))
        exponent = self.rng.uniform(min_exponent, max_exponent)
        return base ** np.linspace(0, exponent, length)

    # --- 2. Harmonic & Periodic ---

    @register_distribution("sine")
    def _generate_sine(self, length: int, params: Dict) -> np.ndarray:
        """Generates a simple sine wave."""
        amplitude = params.get("amplitude", self.rng.uniform(1.0, 10.0))
        min_periods = params.get("min_periods", 5)
        
        # Nyquist limit is 0.5; but still hits resolution issue. 
        # e.q. freq=0.25 is only 4 points per cycle.
        nyquist_limit = 0.15

        # 1. Calculate and validate min_freq
        min_freq = min_periods / length
        if min_freq > nyquist_limit:
            print(f"--- WARNING: Length {length} is too short for {min_periods} periods. ---")
            print(f"Clamping frequency to {nyquist_limit} to prevent aliasing.")
            min_freq = nyquist_limit

        # 2. Validate max_freq
        user_max = params.get("max_freq", 0.15)
        max_freq = max(min_freq, min(user_max, nyquist_limit))

        # 3. Generate frequency and signal
        frequency = params.get("frequency", self.rng.uniform(min_freq, max_freq))
        phase = params.get("phase", 0.0)
        
        t = np.arange(length)
        return amplitude * np.sin(2 * np.pi * frequency * t + phase)

    @register_distribution("sawtooth")
    def _generate_sawtooth(self, length: int, params: Dict) -> np.ndarray:
        """Generates a sawtooth wave."""
        min_periods = params.get("min_periods", 5)

        # Period must be at most (length / min_periods)
        max_period = int(length / min_periods)
        if max_period < 2:
            max_period = 2

        period = params.get("period", self.rng.integers(2, max_period + 1))
        num_periods = int(np.ceil(length / period))
        return np.tile(np.linspace(-1, 1, period), num_periods)[:length]

    @register_distribution("square")
    def _generate_square(self, length: int, params: Dict) -> np.ndarray:
        """Generates a square wave."""
        min_periods = params.get("min_periods", 5)

        min_freq = min_periods / length
        max_freq = max(min_freq, params.get("max_freq", 0.2))

        frequency = params.get("frequency", self.rng.uniform(min_freq, max_freq))
        duty = params.get("duty", 0.5)
        phase = params.get("phase", 0.0)
        t = np.arange(length)
        return signal.square(2 * np.pi * frequency * t + phase, duty=duty)

    @register_distribution("triangle")
    def _generate_triangle(self, length: int, params: Dict) -> np.ndarray:
        """Generates a triangle wave with resolution safety checks."""
        min_periods = params.get("min_periods", 5)
        
        # 0.125 = 8 points per cycle. Anything higher starts losing the shape.
        nyquist_safety_limit = 0.125
        
        # Calculate the requested minimum frequency
        calculated_min_freq = min_periods / length

        # Check for Aliasing/Resolution issues
        if calculated_min_freq > nyquist_safety_limit:
            # If the requested min_periods forces a frequency that is too high
            # for the given length, we clamp it to the safety limit.
            min_freq = nyquist_safety_limit
        else:
            min_freq = calculated_min_freq

        # Ensure user_max doesn't exceed the safety limit either
        user_max = params.get("max_freq", 0.2)
        max_freq = min(user_max, nyquist_safety_limit)
        
        # Ensure max isn't lower than min (if min got clamped high)
        max_freq = max(min_freq, max_freq)

        frequency = params.get("frequency", self.rng.uniform(min_freq, max_freq))
        width = params.get("width", 0.5)
        phase = params.get("phase", 0.0)
        t = np.arange(length)
        
        return signal.sawtooth(2 * np.pi * frequency * t + phase, width=width)

    @register_distribution("noise_patch")
    def _generate_noise_patch(self, length: int, params: Dict) -> np.ndarray:
        """Generates a series by repeating a short patch of noise."""
        min_periods = params.get("min_periods", 5)

        max_period = int(length / min_periods)
        if max_period < 2:
            max_period = 2

        period = params.get("period", self.rng.integers(2, max_period + 1))
        num_patches = params.get("num_patches", math.ceil(length / period))
        return np.tile(self.rng.uniform(0, 1, size=period), num_patches)[:length]

    @register_distribution("fourier")
    def _generate_fourier(self, length: int, params: Dict) -> np.ndarray:
        """Generates a repeating complex signal using harmonic Fourier components."""
        min_periods = params.get("min_periods", 5)

        # 1. Define the Fundamental Frequency (Base Rhythm)
        # To repeat at least 'min_periods' times, the base freq must be >= min_periods / length.
        # We allow it to be slightly faster (up to 2x) to add variety.
        min_fund_freq = min_periods / length
        max_fund_freq = (min_periods * 2) / length

        # Safety: Ensure fundamental doesn't exceed global max_freq (0.2)
        # If length is very small, this might clamp, but prevents aliasing.
        max_allowed = 0.2
        if min_fund_freq > max_allowed:
            min_fund_freq = max_allowed
        if max_fund_freq > max_allowed:
            max_fund_freq = max_allowed

        fundamental_freq = self.rng.uniform(min_fund_freq, max_fund_freq)

        # 2. Decide harmonics (Integer multiples)
        # We can only use harmonics that stay below the Nyquist-safe limit (0.2 or 0.5)
        # e.g. if fundamental is 0.05, we can use harmonics 1, 2, 3, 4 (up to 0.20)
        max_harmonic_idx = int(0.2 / fundamental_freq)
        if max_harmonic_idx < 1:
            max_harmonic_idx = 1

        num_components = self.rng.integers(3, 6)

        # Randomly select which harmonics to use (e.g., 1st, 3rd, 5th)
        # We use replacement=False so we don't pick the same harmonic twice
        available_harmonics = np.arange(
            1, max_harmonic_idx + 2
        )  # +2 ensures at least [1] exists
        chosen_harmonics = self.rng.choice(
            available_harmonics,
            size=min(num_components, len(available_harmonics)),
            replace=False,
        ).reshape(-1, 1)

        # 3. Generate parameters
        freqs = chosen_harmonics * fundamental_freq
        amps = self.rng.uniform(0, 1, size=freqs.shape)
        phases = self.rng.uniform(0, 1, size=freqs.shape)

        # 4. Vectorized Sum
        t = np.arange(length).reshape(1, -1)
        data = np.sum(amps * np.sin(2 * np.pi * freqs * t + phases), axis=0)

        return data

    # --- 3. Transient Dynamics ---

    @register_distribution("damped_sine")
    def _generate_damped_sine(self, length: int, params: Dict) -> np.ndarray:
        """Generates a sine wave with exponential decay or growth."""
        amplitude = params.get("amplitude", self.rng.uniform(1, 10))
        min_freq = params.get("min_periods", 5) / length
        frequency = params.get("frequency", self.rng.uniform(min_freq, 0.2))

        min_cycles = params.get("min_cycles", 8)
        min_freq = min_cycles / length

        # Cap max_freq to 0.25 (Nyquist safety) or higher if length is very short
        max_freq = max(min_freq * 2, 0.25)

        frequency = params.get("frequency", self.rng.uniform(min_freq, max_freq))

        # Decay scaling: Ensure signal lasts for a significant portion of the window
        # roughly 3 to 5 time constants within the length
        min_decay = 2.0 / length
        max_decay = 6.0 / length
        decay = params.get("decay", self.rng.uniform(min_decay, max_decay))

        if params.get("growth", False):
            decay = -abs(decay)
        phase = params.get("phase", 0.0)
        t = np.arange(length)
        return (
            amplitude * np.exp(-decay * t) * np.sin(2 * np.pi * frequency * t + phase)
        )

    @register_distribution("growing_sine")
    def _generate_growing_sine(self, length: int, params: Dict) -> np.ndarray:
        """Generates a sine wave with exponential growth."""
        return self._generate_damped_sine(length, {**params, "growth": True})

    @register_distribution("chirp")
    def _generate_chirp(self, length: int, params: Dict) -> np.ndarray:
        """Generates a frequency-swept cosine signal."""
        f_min = 2.0 / length
        f_max = 0.25  # Nyquist safety

        f0 = params.get("f0", self.rng.uniform(f_min, f_max * 0.3))
        t1 = length
        f1 = params.get(
            "f1", self.rng.uniform(f0 * 2, f_max)
        )  # Ensure sweep is significant

        method = "logarithmic" if params.get("logarithmic", False) else "linear"
        t = np.arange(length)
        return signal.chirp(t, f0=f0, t1=t1, f1=f1, method=method)

    # --- 4. Autocorrelated Processes ---

    @register_distribution("ar")
    def _generate_ar(self, length: int, params: Dict) -> np.ndarray:
        """Generates an Autoregressive (AR) process."""
        coeffs = params.get("coeffs", [self.rng.uniform(0.1, 0.9)])
        if not isinstance(coeffs, list):
            coeffs = [coeffs]
        sigma = params.get("sigma", 1)
        noise = self.rng.normal(0, sigma, size=length)
        a = [1] + [-c for c in coeffs]
        b = [1]
        return signal.lfilter(b, a, noise)

    @register_distribution("ma")
    def _generate_ma(self, length: int, params: Dict) -> np.ndarray:
        """Generates a Moving Average (MA) process."""
        coeffs = params.get("coeffs", [self.rng.uniform(0.1, 0.9)])
        if not isinstance(coeffs, list):
            coeffs = [coeffs]
        sigma = params.get("sigma", 1)
        noise = self.rng.normal(0, sigma, size=length)
        b = [1] + coeffs
        a = [1]
        return signal.lfilter(b, a, noise)

    @register_distribution("arma")
    def _generate_arma(self, length: int, params: Dict) -> np.ndarray:
        """Generates an Autoregressive Moving Average (ARMA) process."""
        ar_coeffs = params.get("ar_coeffs", [self.rng.uniform(0.1, 0.9)])
        ma_coeffs = params.get("ma_coeffs", [self.rng.uniform(0.1, 0.9)])
        if not isinstance(ar_coeffs, list):
            ar_coeffs = [ar_coeffs]
        if not isinstance(ma_coeffs, list):
            ma_coeffs = [ma_coeffs]
        sigma = params.get("sigma", 1)
        noise = self.rng.normal(0, sigma, size=length)
        a = [1] + [-c for c in ar_coeffs]
        b = [1] + ma_coeffs
        return signal.lfilter(b, a, noise)

    @register_distribution("sarima")
    def _generate_sarima(self, length: int, params: Dict) -> np.ndarray:
        """Generates a Seasonal Autoregressive Integrated Moving Average (SARIMA) process."""
        # (p, d, q) x (P, D, Q)s
        p, d, q = params.get("order", (1, 1, 1))
        P, D, Q, s = params.get("seasonal_order", (0, 1, 0, 12))

        if s >= length:
            s = max(2, length // 2)

        # Coefficients (random if not provided)
        phi = params.get("ar_coeffs", self.rng.uniform(-0.3, 0.3, p).tolist())
        theta = params.get("ma_coeffs", self.rng.uniform(-0.3, 0.3, q).tolist())
        Phi = params.get("sar_coeffs", self.rng.uniform(-0.2, 0.2, P).tolist())
        Theta = params.get("sma_coeffs", self.rng.uniform(-0.2, 0.2, Q).tolist())

        sigma = params.get("sigma", 1.0)
        noise = self.rng.normal(0, sigma, size=length)

        # Non-seasonal polynomials
        poly_phi = np.array([1.0] + [-c for c in phi])
        poly_theta = np.array([1.0] + theta)

        # Seasonal polynomials
        poly_Phi = np.zeros(P * s + 1)
        poly_Phi[0] = 1.0
        for i, c in enumerate(Phi):
            poly_Phi[(i + 1) * s] = -c

        poly_Theta = np.zeros(Q * s + 1)
        poly_Theta[0] = 1.0
        for i, c in enumerate(Theta):
            poly_Theta[(i + 1) * s] = c

        # Combine
        combined_a = np.convolve(poly_phi, poly_Phi)
        combined_b = np.convolve(poly_theta, poly_Theta)

        # ARMA filtering
        data = signal.lfilter(combined_b, combined_a, noise)

        # Integration (d) - Cumulative sum
        for _ in range(d):
            data = np.cumsum(data)

        # Seasonal Integration (D) - Recurring cumulative sum
        # Y_t = X_t + Y_{t-s}
        for _ in range(D):
            # We use lfilter to undifference seasonally
            diff_s_a = np.zeros(s + 1)
            diff_s_a[0] = 1.0
            diff_s_a[s] = -1.0
            data = signal.lfilter([1.0], diff_s_a, data)

        # Normalize to prevent explosion from integration if desired
        if params.get("normalize_result", True):
            data = (data - np.mean(data)) / (np.std(data) + 1e-8)

        return data

    @register_distribution("garch")
    def _generate_garch(self, length: int, params: Dict) -> np.ndarray:
        """Generates a Generalized Autoregressive Conditional Heteroskedasticity (GARCH) process."""
        # GARCH(1,1)
        omega = params.get("omega", self.rng.uniform(0.1, 0.9))
        alpha = params.get("alpha", self.rng.uniform(0.1, 0.9))
        beta = params.get("beta", self.rng.uniform(0.1, 0.9))

        vols = np.zeros(length)
        errors = np.zeros(length)
        vols[0] = omega / (1 - alpha - beta) if (alpha + beta) < 1 else omega

        for t in range(1, length):
            vols[t] = omega + alpha * (errors[t - 1] ** 2) + beta * vols[t - 1]
            errors[t] = np.sqrt(vols[t]) * self.rng.normal()
        return errors

    @register_distribution("pink_noise")
    def _generate_pink_noise(self, length: int, params: Dict) -> np.ndarray:
        """Generates pink noise (1/f noise) using Real-FFT."""
        beta = params.get("beta", 1.0)

        # optimization: rfft takes real input and returns only positive frequencies
        white = self.rng.normal(size=length)
        f_white = np.fft.rfft(white)
        freqs = np.fft.rfftfreq(length)

        # Avoid divide by zero at DC component (index 0)
        with np.errstate(divide="ignore", invalid="ignore"):
            s_scale = 1 / np.power(np.abs(freqs), beta / 2.0)
        s_scale[0] = 0

        f_pink = f_white * s_scale

        # irfft returns real output directly
        data = np.fft.irfft(f_pink, n=length)
        return (data - np.mean(data)) / (np.std(data) + 1e-8)

    @register_distribution("fractional_brownian")
    def _generate_fractional_brownian(self, length: int, params: Dict) -> np.ndarray:
        """Generates fractional Brownian motion (fBm) using spectral approximation."""
        # Approximate fBm using Davies-Harte or similar logic is complex,
        # using a simpler power-law spectral method for synthetic research.
        hurst = params.get("hurst", 0.7)
        # Generate white noise
        white = self.rng.normal(size=length)
        # Apply fractional integration in frequency domain
        freqs = np.fft.fftfreq(length)
        # Power law filter: 1/f^(H + 0.5)
        with np.errstate(divide="ignore", invalid="ignore"):
            kernel = np.abs(freqs) ** -(hurst + 0.5)
        kernel[0] = 0
        f_white = np.fft.fft(white)
        f_fbm = f_white * kernel
        data = np.fft.ifft(f_fbm).real
        return data / np.std(data)  # Normalize

    @register_distribution("markov_chain")
    def _generate_markov_chain(self, length: int, params: Dict) -> np.ndarray:
        """Generates a simple discrete-state Markov chain."""
        # Simple two-state Markov chain by default
        # states: [0, 1]
        t_matrix = params.get("transition_matrix", [[0.9, 0.1], [0.2, 0.8]])
        values = params.get("state_values", [0.0, 1.0])

        current_state = 0
        data = np.zeros(length)
        for t in range(length):
            data[t] = values[current_state]
            current_state = self.rng.choice(len(values), p=t_matrix[current_state])
        return data

    # --- 5. I.I.D. Noise & Shocks ---

    @register_distribution("normal")
    def _generate_normal(self, length: int, params: Dict) -> np.ndarray:
        """Generates i.i.d. Gaussian (normal) samples."""
        loc = params.get("loc", 0)
        scale = params.get("scale", 1)
        return self.rng.normal(loc=loc, scale=scale, size=length)

    @register_distribution("uniform")
    def _generate_uniform(self, length: int, params: Dict) -> np.ndarray:
        """Generates i.i.d. Uniformly distributed samples."""
        low = params.get("low", -1)
        high = params.get("high", 1)
        return self.rng.uniform(low, high, size=length)

    @register_distribution("poisson")
    def _generate_poisson(self, length: int, params: Dict) -> np.ndarray:
        """Generates i.i.d. Poisson distributed samples."""
        lam = params.get("lam", self.rng.uniform(0, 3))
        return self.rng.poisson(lam=lam, size=length)

    @register_distribution("binary")
    def _generate_binary(self, length: int, params: Dict) -> np.ndarray:
        """Generates i.i.d. binary (0/1) samples."""
        p = params.get("p", self.rng.uniform(0, 1))
        return self.rng.choice([0, 1], size=length, p=[1 - p, p]).astype(float)

    @register_distribution("student_t")
    def _generate_student_t(self, length: int, params: Dict) -> np.ndarray:
        """Generates i.i.d. Student's t-distributed samples."""
        df = params.get("df", 3)
        loc = params.get("loc", 0)
        scale = params.get("scale", 1)
        return self.rng.standard_t(df, size=length) * scale + loc

    @register_distribution("lognormal")
    def _generate_lognormal(self, length: int, params: Dict) -> np.ndarray:
        """Generates i.i.d. Log-normally distributed samples."""
        mean = params.get("mean", 0.0)
        sigma = params.get("sigma", self.rng.uniform(0.1, 1))
        return self.rng.lognormal(mean, sigma, length)

    @register_distribution("laplace")
    def _generate_laplace(self, length: int, params: Dict) -> np.ndarray:
        """Generates i.i.d. Laplace distributed samples."""
        loc = params.get("loc", 0.0)
        scale = params.get("scale", self.rng.uniform(0.1, 1))
        return self.rng.laplace(loc, scale, length)

    @register_distribution("cauchy")
    def _generate_cauchy(self, length: int, params: Dict) -> np.ndarray:
        """Generates i.i.d. Cauchy distributed samples."""
        return self.rng.standard_cauchy(length)

    @register_distribution("skew_normal")
    def _generate_skew_normal(self, length: int, params: Dict) -> np.ndarray:
        """Generates i.i.d. Skew-normal distributed samples."""
        a = params.get("a", 5.0)  # Skewness parameter
        loc = params.get("loc", 0.0)
        scale = params.get("scale", self.rng.uniform(0.1, 1))
        return stats.skewnorm.rvs(
            a, loc=loc, scale=scale, size=length, random_state=self.rng
        )

    # --- 6. Stochastic Drift/Shocks ---

    @register_distribution("random_walk")
    def _generate_random_walk(self, length: int, params: Dict) -> np.ndarray:
        """Generates a simple random walk."""
        loc = params.get("loc", 0)
        scale = params.get("scale", self.rng.uniform(0.1, 1))
        initial_value = params.get("initial_value", 0)
        steps = self.rng.normal(loc=loc, scale=scale, size=length - 1)
        return np.concatenate(([initial_value], steps)).cumsum()

    @register_distribution("piecewise_constant")
    def _generate_piecewise_constant(self, length: int, params: Dict) -> np.ndarray:
        """Generates a piecewise constant function using indexing."""
        min_seg_len = max(2, int(length * 0.05))
        max_seg_len = max(min_seg_len + 2, int(length * 0.2))
        avg_len = self.rng.integers(min_seg_len, max_seg_len)

        # Determine number of segments
        num_segments = min(
            params.get("num_segments", max(1, length // avg_len)), length
        )

        if num_segments <= 1:
            return np.full(length, self.rng.uniform(-1, 1))

        # 1. Pick change points
        change_points = np.sort(
            self.rng.choice(np.arange(1, length), num_segments - 1, replace=False)
        )

        # 2. Create an index mask: 0 0 0 1 0 0 1 0 ...
        # Cumsum turns this into:  0 0 0 1 1 1 2 2 ... (Segment IDs)
        indices = np.zeros(length, dtype=int)
        indices[change_points] = 1
        segment_ids = np.cumsum(indices)

        # 3. Generate random values for each segment and map them
        values = self.rng.uniform(-1, 1, size=num_segments)
        return values[segment_ids]

    @register_distribution("gbm")
    def _generate_gbm(self, length: int, params: Dict) -> np.ndarray:
        """Generates Geometric Brownian Motion (GBM)."""
        mu = params.get("mu", 0.1)
        sigma = params.get("sigma", 0.2)
        initial_value = params.get("initial_value", 1.0)
        dt = 1.0
        t = np.arange(length)
        w = self.rng.normal(0, np.sqrt(dt), size=length).cumsum()
        w = np.concatenate(([0], w[:-1]))
        return initial_value * np.exp((mu - 0.5 * sigma**2) * t + sigma * w)

    @register_distribution("intermittent")
    def _generate_intermittent(self, length: int, params: Dict) -> np.ndarray:
        """Generates an intermittent time series (many zeros, random spikes)."""
        p = params.get("probability", self.rng.uniform(0.02, 0.2))
        magnitude_dist = params.get("magnitude_dist", "poisson")

        occurrences = self.rng.binomial(1, p, length)
        if magnitude_dist == "poisson":
            magnitudes = self.rng.poisson(
                params.get("lam", self.rng.uniform(1, 10)), length
            )
        else:
            magnitudes = self.rng.uniform(1, 10, length)

        return (occurrences * magnitudes).astype(float)

    @register_distribution("impulse")
    def _generate_impulse(self, length: int, params: Dict) -> np.ndarray:
        """Generates a series with sparse random impulses."""
        min_impulses = max(1, int(length * 0.01))
        max_impulses = max(min_impulses + 1, int(length * 0.1))

        default_impulses = self.rng.integers(min_impulses, max_impulses)
        num_impulses = params.get("num_impulses", default_impulses)
        num_impulses = min(num_impulses, length)  # Safety cap

        magnitude = params.get("magnitude", self.rng.uniform(0, 10))
        data = np.zeros(length)
        indices = self.rng.choice(length, num_impulses, replace=False)
        data[indices] = magnitude
        return data

    @register_distribution("logistic")
    def _generate_logistic(self, length: int, params: Dict) -> np.ndarray:
        """Generates a deterministic chaotic series using the logistic map."""
        r = params.get("r", 3.9)
        x0 = params.get("initial_value", self.rng.uniform(0.1, 0.9))
        x = np.zeros(length)
        x[0] = x0
        for i in range(1, length):
            x[i] = r * x[i - 1] * (1 - x[i - 1])
        return x

    def create_univariate_sample(
        self, length: int, distribution: str, params: Dict = None
    ) -> pd.DataFrame:
        """
        Generates a single univariate time series sample.

        Args:
            length (int): The number of time steps in the series.
            distribution (str): The name of the distribution (e.g., 'normal', 'poisson', 'random_walk', 'sine').
            params (Dict, optional): A dictionary of parameters specific to the distribution. Defaults to None.

        Returns:
            pd.DataFrame: A pandas DataFrame containing the generated time series.
        """

        if params is None:
            params = {}

        if distribution not in self._distribution_map:
            raise ValueError(f"Unsupported distribution: {distribution}")

        generator_func = self._distribution_map[distribution]
        data = generator_func(length, params)

        # add effects
        data = self.add_effects(data, params)

        df = pd.Series(data, name=f"V_{0}").to_frame()
        df.index = pd.date_range(self.start_date, periods=length, freq=self.frequency)
        return df

    def create_univariate_dataset(
        self, num_samples: int, length: int, distribution: str, params: Dict = None
    ) -> List[pd.Series]:
        """
        Generates a dataset of multiple univariate time series samples.

        Args:
            num_samples (int): The number of series to generate in the dataset.
            length (int): The number of time steps in each series.
            distribution (str): The name of the distribution.
            params (Dict, optional): Parameters specific to the distribution. Defaults to None.

        Returns:
            List[pd.Series]: A list of pandas Series, each representing a time series sample.
        """
        samples = []
        for i in range(num_samples):
            self._seed_rng(i)
            samples.append(self.create_univariate_sample(length, distribution, params))
        return samples

    def create_multivariate_sample(
        self,
        length: int,
        num_series: int,
        correlation_matrix: np.ndarray = None,
        distributions: Union[str, List[str]] = "normal",
        params: Union[Dict, List[Dict]] = None,
    ) -> pd.DataFrame:
        """
        Generates a single multivariate time series sample.

        Args:
            length (int): The number of time steps in the series.
            num_series (int): The number of individual series to generate.
            correlation_matrix (np.ndarray, optional): The correlation matrix for the series.
                                            If None, independent series are generated.
                                            Defaults to None.
            distributions (Union[str, List[str]], optional): The distribution(s) for the series. Can be a single
                                                              string or a list of strings. Defaults to 'normal'.
            params (Union[Dict, List[Dict]], optional): The parameter(s) for the distribution(s).
                                                         Can be a single dict or a list of dicts.
                                                         Defaults to None.

        Returns:
            pd.DataFrame: A pandas DataFrame containing the generated multivariate time series.
        """
        if not isinstance(distributions, list):
            distributions = [distributions] * num_series
        if params is None:
            params = [{}] * num_series
        if not isinstance(params, list):
            params = [params] * num_series

        final_data = np.zeros((length, num_series))

        if correlation_matrix is not None and all(
            [i in ["normal", "poisson"] for i in distributions]
        ):
            mean_vector = [p.get("loc", 0) for p in params]

            # Generate correlated normal data using the Cholesky decomposition of the correlation matrix
            try:
                chol = np.linalg.cholesky(correlation_matrix)
            except np.linalg.LinAlgError:
                raise ValueError(
                    "The provided correlation matrix is not positive semi-definite."
                )

            uncorrelated_data = self.rng.normal(size=(length, num_series))
            correlated_data = np.dot(uncorrelated_data, chol.T) + mean_vector

            # Transform to other distributions if specified
            for i in range(num_series):
                dist = distributions[i]
                if dist == "normal":
                    final_data[:, i] = correlated_data[:, i]
                elif dist == "poisson":
                    lam = params[i].get("lam", 1)
                    # Note: This is a simplification. Converting correlated normals to poisson isn't trivial.
                    # Here, we scale the normal data and apply a Poisson distribution.
                    scaled_data = correlated_data[:, i] - correlated_data[:, i].min()
                    scaled_data = scaled_data * (lam / scaled_data.mean())
                    final_data[:, i] = self.rng.poisson(lam=scaled_data)
                else:
                    raise ValueError(
                        f"Unsupported distribution for multivariate sample: {dist}"
                    )
        else:  # No correlation matrix provided or unsuitable distribution requested, generate independent samples
            for i in range(num_series):
                final_data[:, i] = self.create_univariate_sample(
                    length, distributions[i], params[i]
                )["V_0"].values

        return pd.DataFrame(final_data, columns=[f"V_{i}" for i in range(num_series)])

    def create_lagged_multivariate_sample(
        self,
        length: int,
        num_series: int,
        lag_amount: int,
        distribution: str,
        params: Dict = None,
    ) -> pd.DataFrame:
        """
        Generates a multivariate time series where each series is a lagged version of the first.

        This implementation shifts the series "back in time" relative to the index,
        meaning NaN values will appear at the end of the series.

        Args:
            length (int): The number of time steps in the series.
            num_series (int): The number of individual series to generate.
            lag_amount (int): The number of steps to lag each subsequent series.
            distribution (str): The distribution for the initial univariate series.
            params (Dict, optional): A dictionary of parameters for the initial distribution.

        Returns:
            pd.DataFrame: A pandas DataFrame containing the generated lagged time series.
        """
        if num_series <= 0:
            raise ValueError("num_series must be a positive integer.")
        if lag_amount < 0:
            raise ValueError("lag_amount cannot be negative.")

        # Create the initial, unlagged univariate series
        df = self.create_univariate_sample(
            length + lag_amount * (num_series - 1), distribution, params
        )
        base_series = df["V_0"]

        # Create lagged versions of the base series by shifting backwards
        for i in range(1, num_series):
            # A negative shift moves the data UP, creating NaNs at the end
            # These covariates will not be forcasted, so NaNs in the prediction window have no effect.
            lagged_series = base_series.shift(-i * lag_amount)

            df[f"V_{i}"] = lagged_series

        df = df.head(length)

        return df

    def create_multivariate_dataset(
        self,
        num_samples: int,
        length: int,
        num_series: int,
        lagged: bool = False,
        lag_amount: int = 5,
        correlation_matrix: np.ndarray = None,
        distributions: Union[str, List[str]] = "normal",
        params: Union[Dict, List[Dict]] = None,
    ) -> List[pd.DataFrame]:
        """
        Generates a dataset of multiple multivariate time series samples.

        Args:
            num_samples (int): The number of series to generate in the dataset.
            length (int): The number of time steps in each series.
            num_series (int): The number of individual series in each sample.
            lagged (bool): Create lagged timeseries or independent/correlated one
            lag_amount (int): The number of steps to lag each subsequent series.
            correlation_matrix (np.ndarray, optional): The correlation matrix for the series.
            distributions (Union[str, List[str]], optional): The distribution(s) for the series.
            params (Union[Dict, List[Dict]], optional): The parameter(s) for the distribution(s).

        Returns:
            List[pd.DataFrame]: A list of pandas DataFrames, each a multivariate time series sample.
        """
        if not lagged:
            samples = []
            for i in range(num_samples):
                self._seed_rng(i)
                samples.append(
                    self.create_multivariate_sample(
                        length, num_series, correlation_matrix, distributions, params
                    )
                )
            return samples

        if lagged:
            samples = []
            for i in range(num_samples):
                self._seed_rng(i)
                samples.append(
                    self.create_lagged_multivariate_sample(
                        length, num_series, lag_amount, distributions, params
                    )
                )
            return samples


if __name__ == "__main__":
    # --- Example Usage ---
    # Define common parameters at the top for easy modification
    length = 128

    univariate_distributions = [
        # 1. Extrapolate Trend
        "constant",
        "linear",
        "exponential",
        "staircase",
        "piecewise_constant",
        # 2. Periodic (Copy-Paste)
        "sine",
        "sawtooth",
        "square",
        "triangle",
        "noise_patch",
        "fourier",
        # 3. Evolving Patterns
        "damped_sine",
        "growing_sine",
        "chirp",
        # 4. Predictable (Noise)
        "ar",
        "ma",
        "arma",
        "sarima",
        "garch",
        "pink_noise",
        "fractional_brownian",
        "markov_chain",
        # 5. Unpredictable
        "normal",
        "uniform",
        "poisson",
        "binary",
        "student_t",
        "lognormal",
        "laplace",
        "cauchy",
        "skew_normal",
        "random_walk",
        "gbm",
        "intermittent",
        "impulse",
        "logistic",
    ]

    univariate_default_distribution = "random_walk"
    multivariate_default_distribution = "random_walk"

    generator = TimeSeriesGenerator(random_state=13579)
    print("\n--- Generating Samples ---")

    # 1. Create univariate samples
    for distribution in univariate_distributions:
        print(
            f"\nGenerating a single univariate {distribution} sample with length={length}..."
        )
        univariate_sample = generator.create_univariate_sample(
            length=length,
            distribution=distribution,
            # params={'lam': 1}
        )
        print(univariate_sample.head())
        plt.figure(figsize=(10, 4))
        plt.plot(univariate_sample)
        plt.title(f"Univariate {distribution} Sample")
        plt.xlabel("Time Step")
        plt.ylabel("Value")
        plt.grid(True)

    # 1. Create univariate samples with noise and trend
    for distribution in univariate_distributions:
        print(
            f"\nGenerating a single univariate {distribution} sample with length={length}..."
        )
        univariate_sample = generator.create_univariate_sample(
            length=length,
            distribution=distribution,
            params={
                "normalize": True,
                "normalization_method": "minmax",
                "add_seasonality": True,
                "add_noise": True,
                "noise_level": 0.01,
                "add_trend": True,
                "trend_slope": 0.5,
            },
        )
        print(univariate_sample.head())
        plt.figure(figsize=(10, 4))
        plt.plot(univariate_sample)
        plt.title(
            f"Univariate {distribution} Sample with Noise, Seasonality, and Trend"
        )
        plt.xlabel("Time Step")
        plt.ylabel("Value")
        plt.grid(True)

    # 2. Create a dataset of univariate samples
    print(
        f"\nGenerating a dataset of 5 univariate samples with length={length} using '{univariate_default_distribution}' distribution..."
    )
    univariate_dataset = generator.create_univariate_dataset(
        num_samples=5,
        length=length,
        distribution=univariate_default_distribution,
        params={"lam": 5},
    )
    print(f"Dataset contains {len(univariate_dataset)} samples.")
    print("First sample from the dataset:")
    print(univariate_dataset[0].head())
    plt.figure(figsize=(10, 6))
    for i, sample in enumerate(univariate_dataset):
        plt.plot(sample, label=f"Sample {i+1}")
    plt.title(f"Univariate Dataset Samples ({univariate_default_distribution})")
    plt.xlabel("Time Step")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True)

    # 3. Create a single multivariate sample
    print(
        f"\nGenerating a single multivariate sample with length={length} using '{multivariate_default_distribution}' distribution..."
    )
    # Define a correlation matrix
    correlation = np.array([[1.0, 0.8], [0.8, 1.0]])
    multivariate_sample = generator.create_multivariate_sample(
        length=length,
        num_series=2,
        correlation_matrix=correlation,
        distributions=multivariate_default_distribution,
    )
    print(multivariate_sample.head())
    print("\nCorrelation matrix of generated data:\n", multivariate_sample.corr())
    plt.figure(figsize=(10, 4))
    for column in multivariate_sample.columns:
        plt.plot(multivariate_sample[column], label=column)
    plt.title(f"Multivariate Sample ({multivariate_default_distribution})")
    plt.xlabel("Time Step")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True)

    # 4. Create a single multivariate sample with correlation
    print(
        f"\nGenerating a single multivariate sample with length={length} using '{multivariate_default_distribution}' distribution..."
    )
    # Define a correlation matrix
    correlation = np.array([[1.0, 0.8], [0.8, 1.0]])
    multivariate_sample = generator.create_multivariate_sample(
        length=length,
        num_series=2,
        correlation_matrix=correlation,
        distributions="normal",
    )
    print(multivariate_sample.head())
    print("\nCorrelation matrix of generated data:\n", multivariate_sample.corr())
    plt.figure(figsize=(10, 4))
    for column in multivariate_sample.columns:
        plt.plot(multivariate_sample[column], label=column)
    plt.title("Multivariate Sample: normal with Correlation")
    plt.xlabel("Time Step")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True)

    # 5. Create lagged multivariate samples
    print(f"\nGenerating a lagged multivariate sample with length={length}...")

    multivariate_sample = generator.create_lagged_multivariate_sample(
        length=length,
        num_series=3,
        lag_amount=4,
        distribution=multivariate_default_distribution,
        # params={'lam': 1}
    )
    print(multivariate_sample.head())
    plt.figure(figsize=(10, 4))
    plt.plot(multivariate_sample)
    plt.title(f"Multivariate lagged {multivariate_default_distribution} Sample")
    plt.xlabel("Time Step")
    plt.ylabel("Value")
    plt.grid(True)

    # 6. Create a dataset of multivariate samples
    multivariate_dataset = generator.create_multivariate_dataset(
        num_samples=3,
        length=length,
        num_series=3,
        lagged=True,
        lag_amount=5,
        distributions=univariate_default_distribution,
    )
    lagged_sample = multivariate_dataset[0]
    print(lagged_sample.head(10))
    print("\nCorrelation matrix of lagged data:\n", lagged_sample.corr())
    plt.figure(figsize=(10, 4))
    for column in lagged_sample.columns:
        plt.plot(lagged_sample[column], label=column)
    plt.title(
        f"Lagged Multivariate Sample from dataset ({univariate_default_distribution} base)"
    )
    plt.xlabel("Time Step")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True)
    plt.show()

    # # 7. Create a dataset of multivariate sample with lagged covariates
    # print(f"\nGenerating a lagged multivariate sample with length={length}...")
    # multivariate_dataset = generator.create_multivariate_dataset(
    #     num_samples=3,
    #     length=length,
    #     num_series=3,
    #     lagged=True,
    #     lag_amount=5,
    #     distributions=univariate_default_distribution,
    # )
    # lagged_sample = multivariate_dataset[0]
    # print(lagged_sample.head(10))
    # print("\nCorrelation matrix of lagged data:\n", lagged_sample.corr())
    # plt.figure(figsize=(10, 4))
    # for column in lagged_sample.columns:
    #     plt.plot(lagged_sample[column], label=column)
    # plt.title(f'Lagged Multivariate Sample ({univariate_default_distribution} base)')
    # plt.xlabel('Time Step')
    # plt.ylabel('Value')
    # plt.legend()
    # plt.grid(True)
    # plt.show()

    plt.show()
