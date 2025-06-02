import numpy as np
import matplotlib.pyplot as plt
from ipywidgets import interact, FloatSlider
from numpy.linalg import eigh

class BellStateSensor:
    def __init__(self, gamma=1.0, T2=5.0, shots=100, sigma_noise=0.02):
        self.gamma = gamma
        self.T2 = T2
        self.shots = shots
        self.sigma_noise = sigma_noise

        self.times = np.linspace(0.1, 5, 100)

        # True field parameters
        self.B0_true = 1.0
        self.omega_true = 1.0
        self.phi0_true = 0.0

    def B_field(self, t, B0, omega):
        return B0 * np.sin(omega * t)

    def phase(self, t, B0, omega, phi0):
        return 2 * self.gamma * self.B_field(t, B0, omega) * t + phi0

    def density_matrix(self, t, B0, omega, phi0):
        """Returns a 4x4 density matrix for Bell state evolved under magnetic field."""
        θ = self.phase(t, B0, omega, phi0)
        decay = np.exp(-t / self.T2)
        rho = np.zeros((4, 4), dtype=complex)

        # |Φ⁺⟩ = (|00⟩ + |11⟩)/√2 -> evolves with phase θ
        rho[0, 0] = rho[3, 3] = 0.5
        rho[0, 3] = 0.5 * decay * np.exp(-1j * θ)
        rho[3, 0] = 0.5 * decay * np.exp(1j * θ)
        return rho

    def expectation(self, rho):
        """Measure observable Z⊗Z"""
        ZZ = np.diag([1, -1, -1, 1])
        return np.real(np.trace(rho @ ZZ))

    def simulate_measurement(self, exp_val):
        """Simulate noisy measurement with binomial sampling"""
        p = (1 + exp_val) / 2
        counts = np.random.binomial(self.shots, p)
        return 2 * (counts / self.shots) - 1 + np.random.normal(0, self.sigma_noise)

    def generate_data(self):
        self.measurements = []
        for t in self.times:
            rho = self.density_matrix(t, self.B0_true, self.omega_true, self.phi0_true)
            exp_val = self.expectation(rho)
            meas = self.simulate_measurement(exp_val)
            self.measurements.append(meas)
        self.measurements = np.array(self.measurements)

    def sqrtm(self, matrix):
        eigvals, eigvecs = eigh(matrix)
        sqrt_vals = np.sqrt(np.maximum(eigvals, 0))
        return eigvecs @ np.diag(sqrt_vals) @ eigvecs.T.conj()

    def fidelity(self, rho_true, rho_est):
        sqrt_rho = self.sqrtm(rho_true)
        inter = sqrt_rho @ rho_est @ sqrt_rho
        sqrt_inter = self.sqrtm(inter)
        return np.real(np.trace(sqrt_inter)) ** 2

    def compute_fidelity_curve(self, B0_est, omega_est, phi0_est):
        fids = []
        for t in self.times:
            r_true = self.density_matrix(t, self.B0_true, self.omega_true, self.phi0_true)
            r_est = self.density_matrix(t, B0_est, omega_est, phi0_est)
            fids.append(self.fidelity(r_true, r_est))
        return np.array(fids)

    def plot_interactive(self):
        @interact(
            B0=FloatSlider(min=0.5, max=1.5, step=0.01, value=1.0, description='B₀'),
            omega=FloatSlider(min=0.5, max=1.5, step=0.01, value=1.0, description='ω'),
            phi0=FloatSlider(min=-np.pi, max=np.pi, step=0.1, value=0.0, description='ϕ₀')
        )
        def update(B0, omega, phi0):
            preds = []
            for t in self.times:
                rho = self.density_matrix(t, B0, omega, phi0)
                preds.append(self.expectation(rho))

            preds = np.array(preds)
            fids = self.compute_fidelity_curve(B0, omega, phi0)

            plt.figure(figsize=(12, 4))

            plt.subplot(1, 2, 1)
            plt.plot(self.times, self.measurements, label='Noisy Data', alpha=0.7)
            plt.plot(self.times, preds, label='Model Prediction', color='crimson')
            plt.xlabel("Time")
            plt.ylabel("⟨Z⊗Z⟩")
            plt.title("Signal Matching")
            plt.legend()

            plt.subplot(1, 2, 2)
            plt.plot(self.times, fids, color='seagreen')
            plt.ylim(0, 1.05)
            plt.xlabel("Time")
            plt.ylabel("Fidelity")
            plt.title("Fidelity Between Estimated & True States")

            plt.tight_layout()
            plt.show()


sensor = BellStateSensor()
sensor.generate_data()
sensor.plot_interactive()
