# app.py — Streamlit inference viewer for PPO T1D model
# Run as: streamlit run app.py

import streamlit as st
import numpy as np
import gymnasium as gym
from gymnasium.wrappers import TimeLimit
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecNormalize
from stable_baselines3.common.monitor import Monitor
from simglucose.simulation.env import T1DSimEnv
from simglucose.patient.t1dpatient import T1DPatient
from simglucose.sensor.cgm import CGMSensor
from simglucose.actuator.pump import InsulinPump
from simglucose.simulation.scenario_gen import RandomScenario
from simglucose.controller.base import Action
from datetime import datetime
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────────────────────
# Must be identical to train_fast.py — model was trained on this env
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
        """Rebuild the simglucose env with a specific seed (for new scenarios)."""
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

    # ── FIX 1: Accept seed in reset() and rebuild env so scenario changes ────
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None and seed != self._seed:
            self._seed = seed
            self._build_env(seed)           # rebuild with new seed/scenario
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
        if bg < 40:
            done = True
        self._prev_bg      = bg
        self._prev_insulin = insulin_dose

        # ── FIX 2: Proper reward function (was hardcoded to 0.0) ─────────────
        reward = self._compute_reward(bg, insulin_dose)

        return self._get_obs(bg), reward, done, False, info

    def _compute_reward(self, bg: float, insulin: float) -> float:
        """
        Reward shaping:
          +1.0  tight range  (80–140 mg/dL)
          +0.5  normal range (70–180 mg/dL)
          -1.0  hypo         (< 70 mg/dL)
          -2.0  severe hypo  (< 54 mg/dL)
          -0.5+ hyper        (> 180 mg/dL, scales with severity)
        """
        if bg < 54:
            return -2.0
        elif bg < 70:
            return -1.0
        elif 80 <= bg <= 140:
            return 1.0
        elif bg <= 180:
            return 0.5
        else:
            return -0.5 - 0.005 * (bg - 180)


# ─────────────────────────────────────────────────────────────────────────────
# SIMULATION RUNNER
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource
def load_model(model_path: str, norm_path: str):
    """Load model + VecNormalize stats. Cached so reload only on path change."""
    def _make():
        env = SimglucoseGymEnv('adult#001', seed=0)   # seed overridden at reset
        env = TimeLimit(env, max_episode_steps=1500)
        env = Monitor(env)
        return env

    raw_env = DummyVecEnv([_make])
    raw_env = VecFrameStack(raw_env, n_stack=4)
    vn_env  = VecNormalize.load(norm_path, raw_env)
    vn_env.training = False
    vn_env.norm_reward = False

    model = PPO.load(model_path, env=vn_env, device="cuda")
    return model, vn_env


def run_episode(model, env, seed: int) -> pd.DataFrame:
    """Run one full episode with the given seed and return a DataFrame."""

    # Pass seed through VecEnv stack so the underlying SimglucoseGymEnv rebuilds
    try:
        env.env_method("reset", seed=seed, indices=[0])
    except Exception:
        pass
    obs = env.reset()

    bg_list, insulin_list, iob_list, reward_list, tir_list = [], [], [], [], []

    while True:
        action, _ = model.predict(obs, deterministic=True)
        insulin_dose = float(np.clip(action[0][0], 0.0, 1.0)) * 0.15

        obs, reward, done, info = env.step(action)

        raw_obs = env.get_original_obs()   # shape: (1, 4*3)
        bg  = float(raw_obs[0][0]) * 100.0
        iob = float(raw_obs[0][2]) * 5.0

        in_range = 70 <= bg <= 180
        bg_list.append(bg)
        insulin_list.append(insulin_dose)
        iob_list.append(iob)
        reward_list.append(float(reward[0]))
        tir_list.append(1 if in_range else 0)

        if done[0]:
            break

    time_axis = [i * 5 for i in range(len(bg_list))]

    return pd.DataFrame({
        "time_min":   time_axis,
        "bg":         bg_list,
        "insulin":    insulin_list,
        "iob":        iob_list,
        "reward":     reward_list,
        "in_range":   tir_list,
    })


# ─────────────────────────────────────────────────────────────────────────────
# STREAMLIT UI
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="T1D RL Controller",
    page_icon="💉",
    layout="wide"
)

st.title("💉 T1D Reinforcement Learning Controller")
st.caption("PPO-based autonomous insulin dosing agent — UVA/Padova Simulator")

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Configuration")

    model_path = st.text_input(
        "Model path (.zip)",
        value="ppo_t1d_v6_final.zip",
        help="Path to your saved PPO model"
    )
    norm_path = st.text_input(
        "VecNormalize path (.pkl)",
        value="ppo_t1d_v6_vecnormalize.pkl",
        help="Must match the model — saved alongside it during training"
    )

    st.divider()
    seed = st.number_input(
        "Episode seed",
        min_value=0, max_value=9999, value=42,
        help="Controls the random meal scenario. Change to test different days."
    )

    run_btn = st.button("▶ Run Episode", type="primary", use_container_width=True)

    st.divider()
    st.markdown("**Glucose Zones**")
    st.markdown("🟢 In Range: 70–180 mg/dL")
    st.markdown("🔴 Hypo: < 70 mg/dL")
    st.markdown("🟡 Hyper: > 180 mg/dL")

# ── Main panel ───────────────────────────────────────────────────────────────
if run_btn:
    with st.spinner("Loading model and normalization stats..."):
        try:
            model, env = load_model(model_path, norm_path)
        except FileNotFoundError as e:
            st.error(f"File not found: {e}\n\nMake sure the model and VecNormalize "
                     f"pkl are in the same folder as app.py, or enter the full path.")
            st.stop()

    with st.spinner(f"Running episode (seed={seed})..."):
        df = run_episode(model, env, seed=seed)

    # ── Metrics row ──────────────────────────────────────────────────────────
    tir      = df["in_range"].mean() * 100
    hypo     = (df["bg"] < 70).mean() * 100
    hyper    = (df["bg"] > 180).mean() * 100
    mean_bg  = df["bg"].mean()
    total_ins= df["insulin"].sum()
    duration = df["time_min"].max()

    col1, col2, col3, col4, col5, col6 = st.columns(6)
    col1.metric("⏱ Duration",       f"{duration:.0f} min")
    col2.metric("🟢 Time in Range",  f"{tir:.1f}%",
                delta=f"{tir - 70:.1f}% vs 70% target",
                delta_color="normal")
    col3.metric("🔴 Time Hypo",      f"{hypo:.1f}%",
                delta=f"{hypo:.1f}%",
                delta_color="inverse")
    col4.metric("🟡 Time Hyper",     f"{hyper:.1f}%",
                delta=f"{hyper:.1f}%",
                delta_color="inverse")
    col5.metric("📊 Mean BG",        f"{mean_bg:.1f} mg/dL")
    col6.metric("💧 Total Insulin",  f"{total_ins:.3f} U")

    st.divider()

    # ── Blood Glucose plot ────────────────────────────────────────────────────
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.06,
        subplot_titles=(
            "Blood Glucose (mg/dL)",
            "Insulin Dose (U/min)",
            "Insulin on Board Proxy"
        ),
        row_heights=[0.5, 0.25, 0.25]
    )

    fig.add_hrect(y0=70,  y1=180, fillcolor="green", opacity=0.08,
                  line_width=0, row=1, col=1)
    fig.add_hline(y=70,  line_dash="dash", line_color="red",
                  line_width=1, row=1, col=1)
    fig.add_hline(y=180, line_dash="dash", line_color="orange",
                  line_width=1, row=1, col=1)
    fig.add_hline(y=120, line_dash="dot", line_color="green",
                  line_width=1, row=1, col=1)

    colors = ["red" if bg < 70 else ("orange" if bg > 180 else "green")
              for bg in df["bg"]]

    fig.add_trace(go.Scatter(
        x=df["time_min"], y=df["bg"],
        mode="lines", line=dict(color="royalblue", width=2),
        name="BG", showlegend=True
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=df["time_min"], y=df["bg"],
        mode="markers", marker=dict(color=colors, size=4),
        name="BG Zone", showlegend=False
    ), row=1, col=1)

    fig.add_trace(go.Bar(
        x=df["time_min"], y=df["insulin"],
        marker_color="steelblue", opacity=0.7, name="Insulin",
    ), row=2, col=1)

    fig.add_trace(go.Scatter(
        x=df["time_min"], y=df["iob"],
        mode="lines", line=dict(color="purple", width=1.5),
        fill="tozeroy", fillcolor="rgba(128,0,128,0.1)", name="IOB",
    ), row=3, col=1)

    fig.update_layout(
        height=700, template="plotly_dark",
        legend=dict(orientation="h", y=1.02),
        margin=dict(l=50, r=30, t=60, b=40),
    )
    fig.update_xaxes(title_text="Time (minutes)", row=3, col=1)
    fig.update_yaxes(title_text="mg/dL", row=1, col=1)
    fig.update_yaxes(title_text="U/min",  row=2, col=1)
    fig.update_yaxes(title_text="Units",  row=3, col=1)

    st.plotly_chart(fig, use_container_width=True)

    # ── Reward plot ───────────────────────────────────────────────────────────
    st.subheader("📈 Reward Over Time")
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=df["time_min"], y=df["reward"].cumsum(),
        mode="lines", line=dict(color="gold", width=2),
        name="Cumulative Reward"
    ))
    fig2.add_trace(go.Scatter(
        x=df["time_min"],
        y=df["reward"].rolling(20, min_periods=1).mean(),
        mode="lines", line=dict(color="orange", width=1.5, dash="dot"),
        name="Rolling Mean (20 steps)"
    ))
    fig2.update_layout(
        height=300, template="plotly_dark",
        xaxis_title="Time (minutes)",
        yaxis_title="Reward",
        margin=dict(l=50, r=30, t=30, b=40)
    )
    st.plotly_chart(fig2, use_container_width=True)

    # ── Raw data expander ─────────────────────────────────────────────────────
    with st.expander("📋 Raw episode data"):
        st.dataframe(
            df.style.background_gradient(subset=["bg"], cmap="RdYlGn")
                    .format({
                        "bg":      "{:.1f}",
                        "insulin": "{:.4f}",
                        "iob":     "{:.4f}",
                        "reward":  "{:.3f}",
                    }),
            use_container_width=True
        )
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇ Download CSV",
            data=csv,
            file_name=f"episode_seed{seed}.csv",
            mime="text/csv"
        )

else:
    st.info(
        "👈 Enter your model paths in the sidebar and click **Run Episode** to start.",
        icon="💡"
    )
    st.markdown("""
    ### What this app shows
    - **Blood Glucose trace** with hypo/hyper zone highlighting
    - **Insulin dosing decisions** made by the RL agent each 5-minute step
    - **Insulin on Board proxy** — the agent's internal estimate of active insulin
    - **Cumulative reward** and rolling mean reward over the episode
    - **Key metrics**: Time-in-Range, Hypo %, Hyper %, Mean BG, Total Insulin

    ### Files needed (same folder as app.py)
    ```
    ppo_t1d_v6_final.zip
    ppo_t1d_v6_vecnormalize.pkl
    ```
    These are saved automatically at the end of `train_fast.py`.
    """)