# RL hyperparameters

This page lists the complete reinforcement-learning (RL) configuration used in
the CIFEr 2027 paper. The tables are generated from the configuration files in
this repository. They are not typed by hand.

- Selected values: `configs/tuned_stage1/<algo>__<market>.yaml` (key `algo.model_kwargs`,
  `algo.policy_kwargs` and `environment.reward_scaling`). Each file also keeps the
  full tuning provenance: trial seeds, checkpoint values and SHA-256 fingerprints of
  the inputs.
- Fixed settings: `configs/algo/<algo>.yaml`. A tuned file overrides the keys
  that it contains; all other keys come from the base file.
- Markets: `nyse` is the U.S. proxy, `pse` is the Philippine Stock Exchange proxy.

## Calibration protocol

| Item | Value |
|---|---|
| Optimiser | Optuna 4.8.0, `TPESampler` |
| Studies | 8 (4 configurations × 2 markets) |
| Trials per study | 20, all completed (0 pruned, 0 failed) |
| Seeds per trial | 3 |
| Friction level during calibration | F3 |
| Split for selection | validation (2018–2019) |
| Checkpoints per trial | 5 (20k, 40k, 60k, 80k, 100k requested steps) |
| Objective | maximum over common checkpoints of the median validation risk-free-adjusted Sharpe across the 3 seeds |
| Final training | 300,000 requested steps, fresh policies, evaluation seeds 1–21 |
| Runtime | Python 3.10.20, Stable-Baselines3 2.8.0, PyTorch 2.12.0, Gymnasium 1.2.3, NumPy 2.2.6, pandas 2.2.3, CPU |

## Search space

| Parameter | Configurations | Range or choices | Scale |
|---|---|---|---|
| `reward_scaling` | all | [0.1, 3.0] | log |
| `learning_rate` | PPO-MLP, PPO-EIIE, SAC | [1e-5, 1e-3] | log |
| `learning_rate` | TD3 | [1e-5, 1e-2] | log |
| `gamma` | all | [0.90, 0.999] | uniform |
| `n_steps` | PPO-MLP, PPO-EIIE | {512, 1024, 2048, 4096} | categorical |
| `batch_size` | PPO-MLP, PPO-EIIE | {64, 128, 256, 512}, only values that divide `n_steps` | categorical |
| `n_epochs` | PPO-MLP, PPO-EIIE | {3, 5, 10} | categorical |
| `gae_lambda` | PPO-MLP, PPO-EIIE | [0.8, 1.0] | uniform |
| `ent_coef` | PPO-MLP, PPO-EIIE | [1e-4, 0.1] | log |
| `clip_range` | PPO-MLP, PPO-EIIE | [0.1, 0.4] | uniform |
| `hidden_dim` (EIIE extractor width) | PPO-EIIE | {16, 32, 64} | categorical |
| `batch_size` | SAC | {128, 256} | categorical |
| `tau` | SAC, TD3 | [0.001, 0.05] | log |
| `batch_size` | TD3 | {64, 128, 256, 512} | categorical |
| `buffer_size` | TD3 | {10,000, 50,000, 100,000} | categorical |
| `action_noise_std` | TD3 | [0.05, 1.0] | log |

Search dimensions: PPO-MLP 9, PPO-EIIE 10, SAC 5, TD3 7. Equal trial counts
therefore do not give equal search coverage.

## Selected values

| Parameter | PPO-MLP U.S. | PPO-MLP PSE | PPO-EIIE U.S. | PPO-EIIE PSE | SAC U.S. | SAC PSE | TD3 U.S. | TD3 PSE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `learning_rate` | 0.000325287 | 8.83963e-05 | 0.000318956 | 0.000416994 | 0.000963134 | 4.92546e-05 | 3.47097e-05 | 4.15195e-05 |
| `gamma` | 0.940309 | 0.959779 | 0.953537 | 0.925585 | 0.973408 | 0.982564 | 0.941234 | 0.996515 |
| `n_steps` | 1024 | 512 | 512 | 512 | — | — | — | — |
| `batch_size` | 256 | 64 | 64 | 512 | 128 | 256 | 256 | 128 |
| `n_epochs` | 10 | 5 | 3 | 3 | — | — | — | — |
| `gae_lambda` | 0.811612 | 0.811568 | 0.906494 | 0.800023 | — | — | — | — |
| `ent_coef` | 0.0882673 | 0.00221958 | 0.00398812 | 0.0167669 | — | — | — | — |
| `clip_range` | 0.353709 | 0.297571 | 0.340337 | 0.150807 | — | — | — | — |
| `hidden_dim` | — | — | 64 | 32 | — | — | — | — |
| `tau` | — | — | — | — | 0.0019014 | 0.00127837 | 0.0123095 | 0.0403128 |
| `buffer_size` | — | — | — | — | — | — | 100000 | 10000 |
| `action_noise_std` | — | — | — | — | — | — | 0.0723486 | 0.0663908 |
| `reward_scaling` | 0.127939 | 0.305987 | 1.00505 | 0.172168 | 0.11285 | 0.106155 | 0.329024 | 0.40185 |
| Winning trial | 10 | 15 | 13 | 15 | 12 | 3 | 0 | 1 |
| Selected checkpoint (steps) | 100000 | 100000 | 60000 | 100000 | 80000 | 40000 | 60000 | 60000 |

The selected checkpoint is the calibration checkpoint with the best objective.
Final policies were trained again from the start for 300,000 steps; calibration
checkpoints were not reused. The TD3 winners are trials 0 and 1, which are in
the sampler's random start-up phase.

## Fixed settings and networks

| Configuration | Fixed settings | Network |
|---|---|---|
| PPO-MLP | Stable-Baselines3 defaults for all keys that are not tuned (for example `vf_coef` 0.5, `max_grad_norm` 0.5) | `MlpPolicy` defaults: separate policy and value networks of [64, 64], tanh |
| PPO-EIIE | As PPO-MLP | `meta/policies/eiie_extractor.py`: two shared 1×1 convolutions (input channels → `hidden_dim` → `hidden_dim`) with ReLU, flattened with 2 cash features, then the `MlpPolicy` head |
| SAC | `buffer_size` 10,000, `learning_starts` 1,000, `train_freq` 1, `gradient_steps` 1, `ent_coef` "auto" | `MlpPolicy` defaults: actor and two critics of [256, 256], ReLU |
| TD3 | `learning_starts` 1,000, `train_freq` 1, `gradient_steps` 1; Gaussian action noise with mean 0 and standard deviation `action_noise_std` on each action dimension; Stable-Baselines3 defaults `policy_delay` 2, `target_policy_noise` 0.2, `target_noise_clip` 0.5 | `MlpPolicy` defaults: actor and two critics of [400, 300], ReLU |

The `release_commit` field in each tuned file identifies the commit of the
private development repository that ran the study.
