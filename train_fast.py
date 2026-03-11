# train_fast.py — Windows + CUDA  v7
# Fixes: obs_rms shared-reference contamination, SaveNorm timing, reward noise,
#        missing TimeLimit on train env, insufficient training length
# Run as: python train_fast.py

import multiprocessing
multiprocessing.set_start_method('spawn', force=True)

import os
import copy
import gymnasium as gym
import numpy as np
from collections import deque
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import (
    SubprocVecEnv, DummyVecEnv, VecFrameStack, VecNormalize
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
from gymnasium.wrappers import TimeLimit
from simglucose.simulation.env import T1DSimEnv
from simglucose.patient.t1dpatient import T1DPatient
from simglucose.sensor.cgm import CGMSensor
from simglucose.actuator.pump import InsulinPump
from simglucose.simulation.scenario_gen import RandomScenario
from simglucose.controller.base import Action
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────────────────────
# ENVIRONMENT
# ─────────────────────────────────────────────────────────────────────────────

class SimglucoseGymEnv(gym.Env):
    def __init__(self, patient_name='adult#001', seed=0):
        super().__init__()
        self.patient_name = patient_name
        self._seed        = seed
        self._build_env(seed)

        self.observation_space = gym.spaces.Box(
            low=np.array( [0.0, -5.0, 0.0], dtype=np.float32),
            high=np.array([10.0,  5.0, 5.0], dtype=np.float32),
            dtype=np.float32
        )
        self.action_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=(1,), dtype=np.float32
        )
        self._prev_bg      = None
        self._prev_insulin = 0.0
        self._iob          = 0.0
        self._iob_decay    = 0.97

    def _build_env(self, seed):
        patient  = T1DPatient.withName(self.patient_name)
        sensor   = CGMSensor.withName('Dexcom', seed=seed)
        pump     = InsulinPump.withName('Insulet')
        scenario = RandomScenario(
            start_time=datetime(2025, 1, 1, 0, 0, 0),
            seed=seed
        )
        self.t1d_env = T1DSimEnv(patient, sensor, pump, scenario)

    def _get_obs(self, bg):
        delta = (bg - self._prev_bg) / 100.0 if self._prev_bg is not None else 0.0
        return np.array([
            np.clip(bg / 100.0,       0.0, 10.0),
            np.clip(delta,           -5.0,  5.0),
            np.clip(self._iob / 5.0,  0.0,  5.0),
        ], dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None and seed != self._seed:
            self._seed = seed
            self._build_env(seed)
        obs, _, _, _       = self.t1d_env.reset()
        bg                 = obs.CGM
        self._prev_bg      = bg
        self._iob          = 0.0
        self._prev_insulin = 0.0
        return self._get_obs(bg), {}

    def step(self, action):
        insulin_dose       = float(np.clip(action[0], 0.0, 1.0)) * 0.15
        self._iob          = self._iob * self._iob_decay + insulin_dose
        act                = Action(basal=insulin_dose, bolus=0)
        obs, _, done, info = self.t1d_env.step(act)
        bg                 = obs.CGM
        reward             = self._compute_reward(bg, insulin_dose)
        if bg < 40:
            done    = True
            reward -= 5.0
        self._prev_bg      = bg
        self._prev_insulin = insulin_dose
        return self._get_obs(bg), reward, done, False, info

    def _compute_reward(self, bg, insulin_dose):
        # ── Zone reward: smooth gaussian centred at 120, range [-4, +1] ──────
        if 70 <= bg <= 180:
            reward = float(np.exp(-0.5 * ((bg - 120.0) / 40.0) ** 2))
        elif bg < 70:
            reward = -1.0 - 3.0 * ((70.0 - bg) / 30.0)
        else:   # bg > 180
            reward = -0.5 - 2.5 * ((bg - 180.0) / 220.0)

        # ── Laziness penalty: not dosing when clearly hyperglycaemic ─────────
        if bg > 180 and insulin_dose < 0.005:
            reward -= 0.5

        # ── FIX: Smoothness penalty halved — was drowning the zone signal ─────
        # Original: 0.3 * delta/0.15 → up to 0.3 per step (30 % of max reward)
        # Fixed:    0.1 * delta/0.15 → up to 0.1 per step (10 % of max reward)
        delta_insulin = abs(insulin_dose - self._prev_insulin)
        reward -= 0.1 * (delta_insulin / 0.15)

        # ── Rate-of-change risk: penalise rapid drop toward hypo ─────────────
        if self._prev_bg is not None:
            delta_bg = bg - self._prev_bg
            if bg < 100 and delta_bg < -10:
                reward -= 0.2 * abs(delta_bg) / 10.0  # was 0.3, reduced

        return float(reward)


# ─────────────────────────────────────────────────────────────────────────────
# ENV FACTORIES
# ─────────────────────────────────────────────────────────────────────────────

def make_train_env(rank, seed=42):
    def _init():
        env = SimglucoseGymEnv('adult#001', seed=seed + rank)
        # FIX: TimeLimit on training env prevents runaway episodes from
        # skewing VecNormalize running stats with unrepresentative long tails.
        # 2000 steps ≈ 167 min simulated — long enough to see meals + overnight
        env = TimeLimit(env, max_episode_steps=2000)
        env = Monitor(env)
        return env
    return _init


def make_eval_env(rank, seed=999):
    def _init():
        env = SimglucoseGymEnv('adult#001', seed=seed + rank)
        env = TimeLimit(env, max_episode_steps=1000)   # bumped from 800
        env = Monitor(env)
        return env
    return _init


# ─────────────────────────────────────────────────────────────────────────────
# CALLBACKS
# ─────────────────────────────────────────────────────────────────────────────

class TIRCallback(BaseCallback):
    """Logs Time-in-Range (70–180 mg/dL) to TensorBoard."""
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self._bg_history = deque(maxlen=20000)

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if "bg" in info:
                self._bg_history.append(info["bg"])
        if len(self._bg_history) >= 1000:
            tir = float(np.mean([70 <= bg <= 180 for bg in self._bg_history]))
            self.logger.record("custom/time_in_range", tir)
        return True


class BestModelNormSaver(EvalCallback):
    """
    FIX: Extends EvalCallback to save VecNormalize stats ONLY when a new
    best model is found — not every step like the old SaveNormCallback.

    Also fixes the shared-reference bug:
      OLD (broken): eval_vn.obs_rms = train_vn.obs_rms
        → both point to the same RunningMeanStd object
        → training updates mutate eval stats → normalization contamination
        → value function collapses → reward tanks from 5k to 500

      NEW (correct): deepcopy on every eval sync
        → eval and train have completely independent stat objects
        → training normalization is never touched by eval
    """
    def __init__(self, train_vn: VecNormalize, norm_save_path: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.train_vn       = train_vn
        self.norm_save_path = norm_save_path
        self._last_best     = -np.inf

    def _on_step(self) -> bool:
        # Sync eval obs_rms from a DEEP COPY of train stats before each eval
        # This gives eval accurate normalisation without any shared references
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            self.eval_env.obs_rms = copy.deepcopy(self.train_vn.obs_rms)

        result = super()._on_step()

        # Save norm stats only when a new best model was actually recorded
        if self.best_mean_reward > self._last_best:
            self._last_best = self.best_mean_reward
            norm_file = os.path.join(self.norm_save_path, "best_model_vecnormalize.pkl")
            self.train_vn.save(norm_file)
            print(f"\n  ✓ New best reward {self.best_mean_reward:.1f} — "
                  f"saved VecNormalize → {norm_file}")

        return result


def linear_schedule(start: float, end: float = 5e-5):
    """Decays learning rate from start → end over training (SB3 passes progress_remaining)."""
    def fn(progress_remaining: float) -> float:
        return end + progress_remaining * (start - end)
    return fn


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':

    os.makedirs("logs", exist_ok=True)
    NUM_ENVS       = 8
    TOTAL_STEPS    = 5_000_000   # was 3M — more steps = higher peak reward
    RESUME_FROM_V5 = False       # set True + fill paths below to continue v5

    V5_MODEL_PATH = "ppo_t1d_v5_final.zip"
    V5_NORM_PATH  = "ppo_t1d_v5_vecnormalize.pkl"

    print(f"Spawning {NUM_ENVS} SubprocVecEnv training workers...")

    # ── Training env ─────────────────────────────────────────────────────────
    train_env = SubprocVecEnv([make_train_env(i) for i in range(NUM_ENVS)])
    train_env = VecFrameStack(train_env, n_stack=4)
    train_vn  = VecNormalize(
        train_env,
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.0,
        clip_reward=10.0,
        gamma=0.995
    )

    # ── Eval env — fully independent VecNormalize ─────────────────────────────
    eval_env = DummyVecEnv([make_eval_env(0)])
    eval_env = VecFrameStack(eval_env, n_stack=4)
    eval_vn  = VecNormalize(
        eval_env,
        norm_obs=True,
        norm_reward=False,
        training=False,       # NEVER update eval stats from observations
        gamma=0.995
    )
    # Seed eval with a deep copy — NOT a reference — so they never share state
    eval_vn.obs_rms = copy.deepcopy(train_vn.obs_rms)
    eval_vn.ret_rms = copy.deepcopy(train_vn.ret_rms)

    # ── PPO ──────────────────────────────────────────────────────────────────
    policy_kwargs = dict(
        net_arch=dict(pi=[256, 256], vf=[256, 256, 128]),
        ortho_init=True,
    )

    if RESUME_FROM_V5 and os.path.exists(V5_MODEL_PATH):
        print(f"Resuming from v5 checkpoint: {V5_MODEL_PATH}")
        # Reload v5 norm stats into train_vn so the running mean is warm
        v5_norm = VecNormalize.load(V5_NORM_PATH, train_env)
        train_vn.obs_rms = copy.deepcopy(v5_norm.obs_rms)
        train_vn.ret_rms = copy.deepcopy(v5_norm.ret_rms)
        eval_vn.obs_rms  = copy.deepcopy(v5_norm.obs_rms)
        model = PPO.load(V5_MODEL_PATH, env=train_vn, device="cuda")
        print("  ✓ v5 model + norm stats loaded. Continuing training...")
    else:
        model = PPO(
            "MlpPolicy",
            train_vn,
            verbose=1,
            learning_rate=linear_schedule(6e-4),
            n_steps=1024,
            batch_size=512,
            n_epochs=8,
            gamma=0.995,
            gae_lambda=0.95,
            clip_range=0.2,
            target_kl=0.02,
            ent_coef=0.02,
            vf_coef=0.75,
            max_grad_norm=0.5,
            device="cuda",
            policy_kwargs=policy_kwargs,
            tensorboard_log="./ppo_t1d_tensorboard/"
        )

    # ── Callbacks ────────────────────────────────────────────────────────────
    tir_cb = TIRCallback()

    eval_cb = BestModelNormSaver(
        train_vn        = train_vn,
        norm_save_path  = './logs/',
        eval_env        = eval_vn,
        best_model_save_path = './logs/',
        log_path        = './logs/',
        # Every 150k training steps — more frequent checkpoints, catch peak faster
        eval_freq       = 150_000 // NUM_ENVS,
        n_eval_episodes = 3,       # was 1 — average over 3 episodes, more stable
        deterministic   = True,
        render          = False,
    )

    print(f"Starting training — {TOTAL_STEPS:,} steps — ETA ~150 min on RTX Ada 2000")
    model.learn(
        total_timesteps = TOTAL_STEPS,
        callback        = [eval_cb, tir_cb],
        progress_bar    = True
    )

    # Save final model + normalization stats
    model.save("ppo_t1d_v7_final")
    train_vn.save("ppo_t1d_v7_vecnormalize.pkl")
    print("\nTraining complete.")
    print("  Final model : ppo_t1d_v7_final.zip")
    print("  VecNormalize: ppo_t1d_v7_vecnormalize.pkl")
    print("  Best model  : logs/best_model.zip")
    print("  Best norm   : logs/best_model_vecnormalize.pkl")