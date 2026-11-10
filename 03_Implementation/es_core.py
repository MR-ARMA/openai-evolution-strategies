# File: 03_Implementation/es_core.py
import os
import numpy as np
import gymnasium as gym

# ==============================================================================
# 1. ENVIRONMENT CONFIGURATIONS (Table 2 & Section 2.2 / 4.1 Defaults)
# ==============================================================================
ENV_CONFIGS = {
    "Swimmer-v4": {
        "ac_bins": 10,       # Paper Section 2.2 & 4.1: 10 uniform bins encourage exploration
        "lr": 0.01,
        "sigma": 0.02,
        "l2coeff": 0.005,
        "pop_size": 64,
        "target_score": 365.0,
    },
    "Hopper-v4": {
        "ac_bins": 10,       # Paper Section 2.2 & 4.1: Discretized actions for Hopper
        "lr": 0.01,
        "sigma": 0.02,
        "l2coeff": 0.005,
        "pop_size": 64,
        "target_score": 3000.0,
    },
    "HalfCheetah-v4": {
        "ac_bins": None,     # Continuous actions work natively for HalfCheetah
        "lr": 0.01,
        "sigma": 0.02,
        "l2coeff": 0.005,
        "pop_size": 64,
        "target_score": 3000.0,
    },
    "Walker2d-v4": {
        "ac_bins": None,     # Continuous actions
        "lr": 0.01,
        "sigma": 0.02,
        "l2coeff": 0.005,
        "pop_size": 64,
        "target_score": 2500.0,
    },
    "Ant-v4": {
        "ac_bins": None,
        "lr": 0.01,
        "sigma": 0.02,
        "l2coeff": 0.005,
        "pop_size": 128,
        "target_score": 2500.0,
    },
    "Humanoid-v4": {
        "ac_bins": None,
        "lr": 0.01,
        "sigma": 0.02,
        "l2coeff": 0.005,
        "pop_size": 256,
        "target_score": 5000.0,
    },
}


# ==============================================================================
# 2. FITNESS SHAPING & ADAM OPTIMIZER (Exact match to es.py & optimizers.py)
# ==============================================================================
def compute_ranks(x: np.ndarray) -> np.ndarray:
    """Computes zero-indexed integer ranks for a 1D array of returns."""
    assert x.ndim == 1
    ranks = np.empty(len(x), dtype=int)
    ranks[x.argsort()] = np.arange(len(x))
    return ranks


def compute_centered_ranks(x: np.ndarray) -> np.ndarray:
    """Maps raw returns to symmetric centered ranks in [-0.5, +0.5]."""
    y = compute_ranks(x.ravel()).reshape(x.shape).astype(np.float32)
    y /= (x.size - 1)
    y -= 0.5
    return y


class Adam:
    """First-order Adam optimizer operating directly on flat parameter vector theta."""
    def __init__(self, theta: np.ndarray, stepsize: float, beta1: float = 0.9, beta2: float = 0.999, epsilon: float = 1e-08):
        self.theta = theta
        self.dim = len(theta)
        self.stepsize = stepsize
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = np.zeros(self.dim, dtype=np.float32)
        self.v = np.zeros(self.dim, dtype=np.float32)
        self.t = 0

    def update(self, globalg: np.ndarray) -> np.ndarray:
        self.t += 1
        a = self.stepsize * np.sqrt(1.0 - self.beta2 ** self.t) / (1.0 - self.beta1 ** self.t)
        self.m = self.beta1 * self.m + (1.0 - self.beta1) * globalg
        self.v = self.beta2 * self.v + (1.0 - self.beta2) * (globalg * globalg)
        step = -a * self.m / (np.sqrt(self.v) + self.epsilon)
        self.theta += step
        return self.theta


# ==============================================================================
# 3. RUNNING OBSERVATION NORMALIZATION
# ==============================================================================
class RunningStat:
    """Tracks running mean and standard deviation of state observations."""
    def __init__(self, shape, eps: float = 1e-2):
        self.sum = np.zeros(shape, dtype=np.float64)
        self.sumsq = np.full(shape, eps, dtype=np.float64)
        self.count = eps

    def increment(self, s: np.ndarray, ssq: np.ndarray, c: int):
        self.sum += s
        self.sumsq += ssq
        self.count += c

    @property
    def mean(self) -> np.ndarray:
        return (self.sum / self.count).astype(np.float32)

    @property
    def std(self) -> np.ndarray:
        return np.sqrt(np.maximum(self.sumsq / self.count - np.square(self.mean), 1e-2)).astype(np.float32)


# ==============================================================================
# 4. MUJOCO MLP POLICY (With Action Discretization per Section 2.2 & 4.1)
# ==============================================================================
def normc_initializer(shape, std: float = 1.0, rng=None) -> np.ndarray:
    """Normalized column initializer matching OpenAI's tf_util.normc_initializer."""
    if rng is None:
        rng = np.random.RandomState()
    out = rng.randn(*shape).astype(np.float32)
    out *= std / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
    return out


class MujocoMLPPolicy:
    """
    2-layer 64-unit Tanh MLP Policy for MuJoCo continuous control.
    Supports action discretization into uniform bins per Section 2.2 & 4.1.
    """
    def __init__(self, obs_dim: int, act_dim: int, hidden_dims=(64, 64), seed: int = 0, ac_bins=None):
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.hidden_dims = hidden_dims
        self.ac_bins = ac_bins
        rng = np.random.RandomState(seed)

        dims = [obs_dim] + list(hidden_dims) + [act_dim]
        self.shapes = []
        params = []
        for i in range(len(dims) - 1):
            w_shape = (dims[i], dims[i + 1])
            b_shape = (dims[i + 1],)
            self.shapes.append((w_shape, b_shape))
            std = 0.01 if i == len(dims) - 2 else 1.0
            w = normc_initializer(w_shape, std=std, rng=rng)
            b = np.zeros(b_shape, dtype=np.float32)
            params.extend([w.ravel(), b.ravel()])

        self.theta = np.concatenate(params).astype(np.float32)
        self.num_params = len(self.theta)

        if self.ac_bins is not None and self.ac_bins > 1:
            self.bin_edges = np.linspace(-1.0, 1.0, self.ac_bins)
        else:
            self.bin_edges = None

    def act(self, obs: np.ndarray, theta: np.ndarray, ob_mean: np.ndarray, ob_std: np.ndarray) -> np.ndarray:
        x = np.clip((obs - ob_mean) / ob_std, -5.0, 5.0).astype(np.float32)
        idx = 0
        for i, (w_shape, b_shape) in enumerate(self.shapes):
            w_size = int(np.prod(w_shape))
            b_size = int(np.prod(b_shape))
            w = theta[idx : idx + w_size].reshape(w_shape)
            idx += w_size
            b = theta[idx : idx + b_size].reshape(b_shape)
            idx += b_size
            x = x @ w + b
            if i < len(self.shapes) - 1:
                x = np.tanh(x)

        if self.bin_edges is not None:
            # Non-smooth step discretization to force diverse exploration (Section 2.2)
            x = np.digitize(x, self.bin_edges) / float(self.ac_bins) * 2.0 - 1.0
        else:
            x = np.clip(x, -1.0, 1.0)
        return x


# ==============================================================================
# 5. CHASING CAMERA UTILITIES (Fixes Out-of-Frame Video Bug)
# ==============================================================================
def make_tracked_env(env_id: str, render_mode: str = "rgb_array"):
    """
    Creates a Gymnasium MuJoCo environment configured with the built-in 'track' camera
    so the camera automatically follows the agent's center of mass.
    """
    try:
        env = gym.make(env_id, render_mode=render_mode, camera_name="track")
    except Exception:
        env = gym.make(env_id, render_mode=render_mode)
    return env


def render_tracked_frame(env) -> np.ndarray:
    """
    Renders an RGB frame while dynamically locking the camera's lookat coordinates
    onto the agent's torso/root position as a fallback if the free camera is active.
    """
    unwrapped = env.unwrapped
    try:
        # Attempt to lock free camera lookat onto torso x/y coordinates
        renderer = getattr(unwrapped, "mujoco_renderer", None)
        if renderer is not None:
            viewer = getattr(renderer, "viewer", None)
            if viewer is not None and hasattr(viewer, "cam"):
                # If camera is in free mode (type 0), manually track root x position
                if getattr(viewer.cam, "type", 0) == 0:
                    viewer.cam.lookat[0] = float(unwrapped.data.qpos[0])
                    if env.spec.id.startswith("Swimmer") and len(unwrapped.data.qpos) > 1:
                        viewer.cam.lookat[1] = float(unwrapped.data.qpos[1])
    except Exception:
        pass
    return env.render()