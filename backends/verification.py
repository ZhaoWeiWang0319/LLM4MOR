"""Shared verification tolerances and timeouts."""
import math


DEFAULT_FEAS_TOL = 1e-6
CONTEXT_TIMEOUT = 300


def feasibility_tolerance(args):
    """Validate the numeric feasibility tolerance shared by both languages."""
    value = getattr(args, 'verify_feas_tol', DEFAULT_FEAS_TOL)
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise ValueError('--verify-feas-tol must be a finite nonnegative number')
    return value
