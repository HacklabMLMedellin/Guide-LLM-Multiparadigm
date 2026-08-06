"""
Paradigm-Aware Timing Decorators
================================
Estimates theoretical execution time on future hardware paradigms by
scaling measured simulation time with configurable hardware factors.

Research-backed default factors:
  • Classical:     1.00  (baseline)
  • Quantum:       0.08  (~12.5× speedup on quantum-friendly problems)
  https://postquantum.com/quantum-computing/quantum-classical/#:~:text=Shor%E2%80%99s%20algorithm%20for%20integer%20factorization.%20In%201994,%20mathematician%20Peter%20Shor%20discovered%20a%20quantum%20algorithm%20that%20could%20factor%20large%20numbers%20exponentially%20faster%20than%20any%20known%20classical%20method
  https://ethz.ch/en/news-and-events/eth-news/news/2023/05/for-very-small-problem-sizes-a-classical-computer-is-faster.html#:~:text=A%20quantum%20computer%20on%20the%20other%20hand%20can%20only%20execute%20hundreds%20of%20thousands%20up%20to%20maybe%20millions%20of%20steps%20per%20second
  • Photonic:      0.04  (~25× speedup via light-speed + WDM parallelism)
  https://www.photondelta.com/blog/how-do-photonic-chips-compare-to-traditional-processors/#:~:text=Photonic%20chips%20can%20process%20multiple%20data%20streams%20simultaneously%20using%20different%20wavelengths%20of%20light,%20a%20technique%20called%20wavelength-division%20multiplexing%20that%20electronic%20processors%20cannot%20replicate%20in%20the%20same%20way.Energy%20efficiency%20represents%20another%20major%20advantage.%20Photonic%20chips%20consume%20substan
  • Thermodynamic: 0.12  (~8.3× speedup approaching Landauer limit)
  https://quantumcomputing.stackexchange.com/questions/28327/thermodynamic-speed-limit-to-quantum-computing
  https://pmc.ncbi.nlm.nih.gov/articles/PMC8960662/
https://share.gemini.google/hvtU5rPw3l8b
"""

from functools import wraps
from time import perf_counter

# ----------------------------------------------------------------------
# CONFIGURABLE HARDWARE FACTORS
# ----------------------------------------------------------------------
# Modify these constants as hardware improves. All decorators read from
# this dictionary, so a single change propagates everywhere.
# ----------------------------------------------------------------------
HARDWARE_FACTORS = {
    "classical": 1.00,
    "quantum": 0.08,
    "photonic": 0.04,
    "thermodynamic": 0.12,
}


# ----------------------------------------------------------------------
# Base timer (measures real wall-clock time on current hardware)
# ----------------------------------------------------------------------
def timer(func):
    """Measure and print actual execution time on classical hardware."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        start = perf_counter()
        result = func(*args, **kwargs)
        elapsed = perf_counter() - start
        print(f"{func.__name__}: {elapsed:.6f} sec")
        return result

    return wrapper


# ----------------------------------------------------------------------
# Paradigm timer factory
# ----------------------------------------------------------------------
def _make_paradigm_timer(paradigm_name, default_factor):
    """
    Create a decorator that estimates theoretical execution time on a
    specific future hardware paradigm.

    real_time = simulation_time × hardware_factor

    Usage:
        @quantum_timer
        def my_func(): ...

        @quantum_timer(factor=0.05)   # override for a specific function
        def my_func(): ...
    """

    def decorator(func=None, *, factor=None):
        chosen_factor = default_factor if factor is None else factor

        def inner_decorator(f):
            @wraps(f)
            def wrapper(*args, **kwargs):
                start = perf_counter()
                result = f(*args, **kwargs)
                sim_time = perf_counter() - start
                real_time = sim_time * chosen_factor

                print(
                    f"[{paradigm_name.upper():13}] {f.__name__:20} "
                    f"sim={sim_time:.6f}s | "
                    f"factor={chosen_factor:6.3f} | "
                    f"est_real={real_time:.6f}s"
                )
                return result

            return wrapper

        if func is None:
            return inner_decorator
        return inner_decorator(func)

    return decorator


# ----------------------------------------------------------------------
# Pre-built decorators for each paradigm
# ----------------------------------------------------------------------
classical_timer = _make_paradigm_timer("classical", HARDWARE_FACTORS["classical"])
quantum_timer = _make_paradigm_timer("quantum", HARDWARE_FACTORS["quantum"])
photonic_timer = _make_paradigm_timer("photonic", HARDWARE_FACTORS["photonic"])
thermodynamic_timer = _make_paradigm_timer(
    "thermodynamic", HARDWARE_FACTORS["thermodynamic"]
)


# ----------------------------------------------------------------------
# Example usage
# ----------------------------------------------------------------------
if __name__ == "__main__":
    import time

    @timer
    def baseline_sort(data):
        time.sleep(0.1)
        return sorted(data)

    @quantum_timer
    def quantum_optimize(data):
        time.sleep(0.1)
        return min(data)

    @photonic_timer
    def photonic_fft(data):
        time.sleep(0.1)
        return [x * 2 for x in data]

    @thermodynamic_timer(factor=0.10)  # override default for this function
    def thermodynamic_search(data):
        time.sleep(0.1)
        return data[0]

    test_data = list(range(1000, 0, -1))

    print("--- Baseline (classical) ---")
    baseline_sort(test_data)

    print("\n--- Future-hardware estimates ---")
    quantum_optimize(test_data)
    photonic_fft(test_data)
    thermodynamic_search(test_data)
