"""Reinforcement-learning part of the cell runner (needs requirements-rl.txt).

``tools/run_cell.py`` imports this module only for RL cells, so that the non-RL
cells run without PyTorch. The model construction is the same as in the
research runner of the paper (``tools/run_experiment_cell.py`` in the private
development repository); functions copied from it say so.
"""

from __future__ import annotations

import copy
import os
import random
from typing import Any, Dict

import numpy as np
import torch
from stable_baselines3 import A2C, DDPG, PPO, SAC, TD3
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.vec_env import DummyVecEnv

from meta.policies.eiie_extractor import EIIEExtractor

SB3_MODELS = {"ppo": PPO, "a2c": A2C, "sac": SAC, "td3": TD3, "ddpg": DDPG}


def algo_family(cfg: Dict[str, Any]) -> str:
    algo_cfg = cfg["algo"]
    family = algo_cfg.get("family", algo_cfg["name"].split("_")[0]).lower()
    if family not in SB3_MODELS:
        raise ValueError(f"Unsupported algo family: {family}")
    return family


# Copied from tools/run_experiment_cell.py (set_seed).
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def configure_torch_threads(threads: int | None) -> Dict[str, int]:
    """Set the PyTorch thread counts before the model is built.

    ``threads`` (``--threads``) wins; otherwise ``TORCH_NUM_THREADS`` or
    ``OMP_NUM_THREADS`` is used, as in the research runner. Inter-op threads are
    always 1. The thread count changes the order of floating-point reductions,
    so a different count can give different (equally valid) training results.
    The paper's RL cells used 2 threads.
    """
    if threads is None:
        raw = os.environ.get("TORCH_NUM_THREADS") or os.environ.get("OMP_NUM_THREADS")
        threads = int(raw) if raw else None
    if threads is not None:
        torch.set_num_threads(max(1, int(threads)))
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass  # already set in this process
    return {
        "torch_num_threads": torch.get_num_threads(),
        "torch_num_interop_threads": torch.get_num_interop_threads(),
    }


# Copied from tools/run_experiment_cell.py (build_model).
def build_model(cfg: Dict[str, Any], env, seed: int, device: str):
    algo_cfg = cfg["algo"]
    family = algo_family(cfg)
    model_cls = SB3_MODELS[family]

    policy_kwargs = dict(algo_cfg.get("policy_kwargs", {}))
    if policy_kwargs.get("features_extractor_class") == "eiie":
        sample_env = env.envs[0] if hasattr(env, "envs") else env
        eiie_kwargs = dict(policy_kwargs.get("features_extractor_kwargs", {}))
        eiie_kwargs.setdefault("n_assets", sample_env.stock_dim)
        eiie_kwargs.setdefault(
            "tech_features_per_asset",
            sample_env.tech_array.shape[1] // sample_env.stock_dim,
        )
        policy_kwargs["features_extractor_class"] = EIIEExtractor
        policy_kwargs["features_extractor_kwargs"] = eiie_kwargs

    model_kwargs = copy.deepcopy(algo_cfg.get("model_kwargs", {}))
    noise_std = model_kwargs.pop("action_noise_std", None)
    if family == "td3" and noise_std is not None and "action_noise" not in model_kwargs:
        sample_env = env.envs[0] if hasattr(env, "envs") else env
        action_dim = int(np.prod(sample_env.action_space.shape))
        model_kwargs["action_noise"] = NormalActionNoise(
            mean=np.zeros(action_dim),
            sigma=np.full(action_dim, float(noise_std)),
        )

    return model_cls(
        algo_cfg.get("policy", "MlpPolicy"),
        env,
        verbose=0,
        seed=seed,
        device=device,
        policy_kwargs=policy_kwargs if policy_kwargs else None,
        **model_kwargs,
    )


def train_model(cfg: Dict[str, Any], train_env, *, seed: int, device: str, total_timesteps: int):
    """Build a fresh model on the training split and train it."""
    vec = DummyVecEnv([lambda e=train_env: e])
    model = build_model(cfg, vec, seed=seed, device=device)
    model.learn(total_timesteps=int(total_timesteps))
    return model


def load_model(cfg: Dict[str, Any], path: str, train_env, *, device: str):
    """Load a saved model (evaluation only, no training)."""
    vec = DummyVecEnv([lambda e=train_env: e])
    return SB3_MODELS[algo_family(cfg)].load(str(path), env=vec, device=device)


def software_versions() -> Dict[str, str]:
    import gymnasium
    import stable_baselines3

    return {
        "torch": torch.__version__,
        "stable-baselines3": stable_baselines3.__version__,
        "gymnasium": gymnasium.__version__,
    }
