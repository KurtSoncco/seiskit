"""Tests for Ricker acceleration/velocity helpers used by ASDA -fx."""

import numpy as np

from seiskit.utils import acceleration_to_velocity, compute_ricker, compute_ricker_velocity


def test_velocity_is_integral_of_acceleration():
    dt = 0.001
    accel = compute_ricker(2.0, 0.5, 5.0, dt)
    vel = compute_ricker_velocity(2.0, 0.5, 5.0, dt)
    vel2 = acceleration_to_velocity(accel, dt)
    assert np.allclose(vel, vel2)
    # End velocity of a zero-mean Ricker accel should be near 0
    assert abs(vel[-1]) < 1e-3 * np.max(np.abs(vel))


def test_velocity_starts_at_zero():
    vel = compute_ricker_velocity(1.0, 1.0, 10.0, 0.01)
    assert vel[0] == 0.0
