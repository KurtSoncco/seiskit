from seiskit.utils import compute_ricker
import numpy as np
import matplotlib.pyplot as plt
def test_compute_ricker():
    samples = compute_ricker(0.75, 1.4, 2.0, 0.01)
    assert len(samples) > 0

def test_compute_ricker_shape():
    samples = compute_ricker(0.75, 1.4, 2.0, 0.01)
    expected_shape = (int(2.0 / 0.01) + 1,)
    assert samples.shape == expected_shape

def test_compute_ricker_values():
    samples = compute_ricker(0.75, 1.4, 15.0, 0.01)
    assert samples[0] == -0.0003904965392413389
    assert samples[-1] == 0.0
    
    
    
def test_with_real_data():
    samples = compute_ricker(freq=0.75, t_shift=1.4, duration=15.0, dt=0.01)
    
    real_data = np.loadtxt("tests/ricker.in", skiprows=1)
    
    # COmpute the difference and plot it and save it to a png file
    difference = samples - real_data
    plt.plot(difference)
    plt.savefig("tests/ricker_difference.png")
    
    # Compute the MAE and MSE between the samples and the real data
    mae = np.mean(np.abs(samples - real_data))
    mse = np.mean(np.square(samples - real_data))
    
    assert mae < 1e-8
    assert mse < 1e-8