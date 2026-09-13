# -*- coding: utf-8 -*-
"""Cooperative navigation RL with MaCA self-defense, Python 3.7 / torch 1.13.

Paper-inspired multilevel counterfactual credit assignment, not a reproduction
of the paper's benchmarks or CMA-ES. See super_5_instrucion.md for details.
Run this file with --help. Training is explicit; Agent.get_action is inference.
Agent defaults to embedded Agent4 pursuit, salvo management and radar/jamming.
Combat training/evaluation use original armed maps; navigation stays compatible.
NavigationAgent is the heading-only interface with an untrained waypoint fallback.
See super_agent_5_pro_updates.md for scope, navigation checks and legacy markers.
"""
import argparse
from contextlib import contextmanager
from collections import Counter
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import pickle
import random
import sys
import tempfile
import time
import unittest
import warnings

# Conda's Windows native dependencies must be visible BEFORE numpy/torch import.
_DLL_HANDLES = []
if os.name == "nt":
    _dll = Path(sys.prefix) / "Library" / "bin"
    if _dll.is_dir():
        os.environ["PATH"] = str(_dll) + os.pathsep + os.environ.get("PATH", "")
        if hasattr(os, "add_dll_directory"):
            _DLL_HANDLES.append(os.add_dll_directory(str(_dll)))
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
DEFAULT_RUN = HERE / "super5_runs" / "default"
FORMAT_VERSION = 2
HEADINGS = np.arange(0, 360, 15, dtype=np.int32)


@dataclass
class Config:
    hidden: int = 64
    heads: int = 4
    grid: int = 8
    neighbors: int = 4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip: float = 0.2
    entropy: float = 0.015
    learning_rate: float = 0.0003
    value_coef: float = 0.5
    grad_clip: float = 0.5
    epochs: int = 4
    batch_steps: int = 64
    corr_threshold: float = 0.05
    # Joint / individual / correlated-set. Fixed, not claimed to be CMA-ES.
    credit_weights: tuple = (0.5, 0.25, 0.25)
    episode_steps: int = 512
    curriculum_updates: int = 200
    seed: int = 17
    backend: str = "maca"
    map_path: str = "maps/1000_1000_fighter10v10.map"
    toy_units: int = 6
    mode: str = "navigation"
    random_pos: bool = False
    survival_weight: float = 1.0
    task_weight: float = 0.1
    kill_weight: float = 1.0
    resource_weight: float = 0.02
    outcome_weight: float = 1.0
    combat_control: str = "legacy"

    @property
    def features(self):
        return 19 + 4 * self.neighbors + self.grid * self.grid

    def validate(self):
        if self.combat_control not in ('legacy', 'agent4', 'residual'):
            raise ValueError('Invalid combat_control')
        if self.mode not in ("navigation", "self-defense", "adversarial"):
            raise ValueError("Invalid mode")
        if self.mode != "navigation" and self.backend != "maca":
            raise ValueError("Combat requires the native maca backend")
        for name in ("survival_weight", "task_weight", "kill_weight", "resource_weight", "outcome_weight"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError(name + " must be finite and nonnegative")
        for name in ("hidden", "heads", "grid", "neighbors", "epochs",
                     "batch_steps", "episode_steps", "toy_units"):
            if getattr(self, name) < 1:
                raise ValueError(name + " must be positive")
        if self.hidden % self.heads or self.grid > 32 or self.toy_units > 64:
            raise ValueError("hidden must be a multiple of heads; grid <= 32; units <= 64")
        if not (0 <= self.gamma < 1 and 0 <= self.gae_lambda <= 1):
            raise ValueError("Invalid discount / GAE lambda")
        if not (0 < self.clip < 1 and 0 <= self.corr_threshold <= 1):
            raise ValueError("Invalid PPO clip / correlation threshold")
        w = np.asarray(self.credit_weights, dtype=np.float64)
        if w.shape != (3,) or not np.isfinite(w).all() or (w < 0).any() or w.sum() <= 0:
            raise ValueError("credit_weights must contain three nonnegative weights")
        for name in ("learning_rate", "grad_clip", "value_coef"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(name + " must be finite and positive")
        if self.entropy < 0 or not math.isfinite(self.entropy):
            raise ValueError("entropy must be finite and nonnegative")
        if self.curriculum_updates < 0 or self.backend not in ("maca", "toy"):
            raise ValueError("Invalid curriculum / backend")
        if not 0 <= self.seed < 2 ** 31:
            raise ValueError("seed must be in [0, 2**31)")
        return self


def resolve_map(path):
    p = Path(path).expanduser()
    return (p if p.is_absolute() else ROOT / p).resolve()


def map_digest(cfg):
    if cfg.backend == "toy":
        return "toy-v1-{}".format(cfg.toy_units)
    return hashlib.sha256(resolve_map(cfg.map_path).read_bytes()).hexdigest()


def select_device(name):
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(name)
    if device.type not in ("cpu", "cuda"):
        raise ValueError("Only CPU and CUDA are supported")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; use --device cpu")
    return device


# [LEGACY COMBAT] Existing targeting/jamming rules; unchanged in this update.
class SelfDefenseModule:
    """Rule-based responses using MaCA's discrete game actions and map units.

    Only the fighter's current radar observations can authorize a response.
    Raw observations do not expose an incoming-attack alarm: nearby visible
    enemies are the trigger, not a claim that an enemy has already fired.
    """
    SHORT_RANGE = 50.0
    LONG_RANGE = 120.0
    FREQUENCIES = 10

    @staticmethod
    def commands(fighters, enemy_count):
        if not isinstance(enemy_count, (int, np.integer)) or not 0 <= enemy_count <= 64:
            raise ValueError("enemy_count must be an integer in [0, 64]")
        commands = np.zeros(len(fighters), dtype=np.int32)
        for i, unit in enumerate(fighters):
            if not unit["alive"]:
                continue
            short = unit.get("s_missile_left", 0) > 0
            long = unit.get("l_missile_left", 0) > 0
            if not (short or long):
                continue
            candidates = []
            for target in unit.get("r_visible_list") or []:
                eid = target.get("id")
                if (not isinstance(eid, (int, np.integer)) or
                        not 1 <= eid <= enemy_count or not target.get("alive", True)):
                    continue
                try:
                    distance = math.hypot(float(target["pos_x"]) - float(unit["pos_x"]),
                                          float(target["pos_y"]) - float(unit["pos_y"]))
                except (KeyError, TypeError, ValueError):
                    continue
                if math.isfinite(distance):
                    candidates.append((distance, int(eid)))
            for distance, eid in sorted(candidates):
                if short and distance <= SelfDefenseModule.SHORT_RANGE:
                    commands[i] = eid + enemy_count
                    break
                if long and distance <= SelfDefenseModule.LONG_RANGE:
                    commands[i] = eid
                    break
        return commands

    @staticmethod
    def jam_frequency(unit):
        frequencies = [s.get("r_fp") for s in unit.get("j_recv_list") or []]
        frequencies = [int(f) for f in frequencies
                       if isinstance(f, (int, np.integer)) and
                       1 <= f <= SelfDefenseModule.FREQUENCIES]
        if not frequencies:
            return 0
        values, counts = np.unique(frequencies, return_counts=True)
        return int(values[counts.argmax()]) if counts.max() * 2 > len(frequencies) else 11


def safe_actions(indices, alive, detector_num, obs=None, enemy_count=None):
    """Build navigation actions, adding self-defense when raw obs are supplied."""
    indices, alive = np.asarray(indices), np.asarray(alive, dtype=bool)
    if indices.shape != alive.shape or indices.ndim != 1:
        raise ValueError("Action / alive shapes differ")
    if not np.issubdtype(indices.dtype, np.integer):
        raise ValueError("Heading indices must be integers")
    if ((indices < 0) | (indices >= len(HEADINGS))).any():
        raise ValueError("Invalid heading index")
    if not 0 <= detector_num <= len(alive):
        raise ValueError("Invalid detector count")
    d = np.zeros((detector_num, 2), dtype=np.int32)
    f = np.zeros((len(alive) - detector_num, 4), dtype=np.int32)
    headings = HEADINGS[indices] * alive.astype(np.int32)
    d[:, 0], f[:, 0] = headings[:detector_num], headings[detector_num:]
    # [LEGACY COMBAT BRIDGE] Supplying obs enables non-navigation channels.
    # NavigationAgent and navigation training never supply obs here.
    if obs is not None:
        detectors = obs.get("detector_obs_list", [])
        fighters = obs.get("fighter_obs_list", [])
        if len(detectors) != len(d) or len(fighters) != len(f):
            raise ValueError("Observation / action counts differ")
        units = list(detectors) + list(fighters)
        if not np.array_equal(alive, [bool(u["alive"]) for u in units]):
            raise ValueError("Observation / action alive masks differ")
        # The standard Agent interface supplies own counts only; bundled maps
        # are symmetric. Callers on asymmetric maps must supply enemy_count.
        if enemy_count is None:
            enemy_count = len(alive)
        f[:, 3] = SelfDefenseModule.commands(fighters, enemy_count)
        for i in np.flatnonzero(alive):
            frequency = int(i % SelfDefenseModule.FREQUENCIES) + 1
            if i < detector_num:
                d[i, 1] = frequency
            else:
                j = i - detector_num
                f[j, 1] = frequency
                f[j, 2] = SelfDefenseModule.jam_frequency(fighters[j])
    assert_safe_actions(d, f, enemy_count if obs is not None else None)
    return d, f


def assert_safe_actions(d, f, enemy_count=None):
    """Validate navigation-only or full MaCA actions, including channel bounds."""
    if enemy_count is not None and (
            not isinstance(enemy_count, (int, np.integer)) or not 0 <= enemy_count <= 64):
        raise ValueError("enemy_count must be an integer in [0, 64]")
    for a, columns in ((d, 2), (f, 4)):
        if a.ndim != 2 or a.shape[1] != columns or a.dtype != np.int32:
            raise ValueError("MaCA actions must be int32 arrays with 2 / 4 columns")
        if ((a[:, 0] < 0) | (a[:, 0] >= 360)).any():
            raise ValueError("Invalid heading")
        if enemy_count is None and (a[:, 1:] != 0).any():
            raise ValueError("Non-navigation command rejected")
        if ((a[:, 1] < 0) | (a[:, 1] > SelfDefenseModule.FREQUENCIES)).any():
            raise ValueError("Invalid radar frequency")
    if enemy_count is not None:
        if ((f[:, 2] < 0) | (f[:, 2] > SelfDefenseModule.FREQUENCIES + 1)).any():
            raise ValueError("Invalid jammer frequency")
        if ((f[:, 3] < 0) | (f[:, 3] > 2 * enemy_count)).any():
            raise ValueError("Invalid attack target encoding")


class NavigationState:
    """Only reads own-team position, course, type and alive status.

    Goals are exogenous navigation waypoints, never extracted from other teams.
    Shared coverage and neighbor observations use the own-team raw observation.
    """
    def __init__(self, cfg, size_x, size_y, detectors, fighters):
        if not np.isfinite([size_x, size_y]).all() or min(size_x, size_y) <= 0:
            raise ValueError("Map dimensions must be finite and positive")
        if any(int(x) != x or x < 0 for x in (detectors, fighters)):
            raise ValueError("Unit counts must be nonnegative integers")
        self.cfg = cfg
        self.scale = np.asarray([size_x, size_y], dtype=np.float32)
        self.detectors, self.fighters = int(detectors), int(fighters)
        self.n = self.detectors + self.fighters
        if self.n > 64:
            raise ValueError("This implementation supports at most 64 units per team")
        self.reset()

    def reset(self, goals=None):
        self.visited = np.zeros((self.cfg.grid, self.cfg.grid), dtype=bool)
        self.previous = None
        self.last_step = None
        self.goals = self.default_goals() if goals is None else self.check_goals(goals)

    def default_goals(self):
        angle = np.arange(self.n) * (2 * math.pi / max(1, self.n))
        radius = 0.2 if self.n > 1 else 0.0
        return np.stack([0.5 + radius * np.cos(angle),
                         0.5 + radius * np.sin(angle)], axis=-1).astype(np.float32)

    def check_goals(self, goals):
        goals = np.asarray(goals, dtype=np.float32)
        if goals.shape != (self.n, 2) or not np.isfinite(goals).all():
            raise ValueError("Goals must be finite normalized coordinates [units, 2]")
        if ((goals < 0.05) | (goals > 0.95)).any():
            raise ValueError("Normalized goals must lie inside [0.05, 0.95]")
        return goals.copy()

    def read(self, obs):
        detectors = obs.get("detector_obs_list", [])
        fighters = obs.get("fighter_obs_list", [])
        if len(detectors) != self.detectors or len(fighters) != self.fighters:
            raise ValueError("Observation counts do not match set_map_info")
        units = list(detectors) + list(fighters)
        alive = np.asarray([bool(u["alive"]) for u in units], dtype=bool)
        pos = np.zeros((self.n, 2), dtype=np.float32)
        course = np.zeros(self.n, dtype=np.float32)
        for i, u in enumerate(units):
            if alive[i]:
                pos[i] = [float(u["pos_x"]), float(u["pos_y"])]
                course[i] = float(u["course"])
        if not np.isfinite(pos).all() or not np.isfinite(course).all():
            raise ValueError("Non-finite observation")
        return np.clip(pos / self.scale, 0, 1), np.deg2rad(course % 360), alive

    def observe(self, obs, step):
        pos, course, alive = self.read(obs)
        if self.last_step is not None and step <= self.last_step:
            self.reset(self.goals)
        self.last_step = step
        cell = np.minimum((pos * self.cfg.grid).astype(int), self.cfg.grid - 1)
        self.visited[cell[alive, 0], cell[alive, 1]] = True
        velocity = np.zeros_like(pos) if self.previous is None else pos - self.previous
        self.previous = pos.copy()
        x = np.zeros((self.n, self.cfg.features), dtype=np.float32)
        center = pos[alive].mean(0) if alive.any() else np.zeros(2)
        for i in np.flatnonzero(alive):
            delta = self.goals[i] - pos[i]
            distance = float(np.linalg.norm(delta))
            others = np.flatnonzero(alive & (np.arange(self.n) != i))
            if len(others):
                others = others[np.argsort(np.linalg.norm(pos[others] - pos[i], axis=1), kind="stable")]
            local = [*pos[i], math.sin(course[i]), math.cos(course[i]),
                     *np.clip(velocity[i] * 20, -1, 1), *delta, distance,
                     float(distance < 0.05), *pos[i], *(1 - pos[i]),
                     float(i < self.detectors), i / max(1, self.n - 1),
                     *(center - pos[i]), min(1.0, step / self.cfg.episode_steps)]
            for k in range(self.cfg.neighbors):
                if k < len(others):
                    rel = pos[others[k]] - pos[i]
                    local.extend([*rel, float(np.linalg.norm(rel)), 1.0])
                else:
                    local.extend([0.0, 0.0, 0.0, 0.0])
            x[i] = np.concatenate([np.asarray(local), self.visited.ravel()])
        return x, alive

    def reward(self, before, after):
        p0, _, a0 = self.read(before)
        p1, _, a1 = self.read(after)
        active = a0 & a1
        if not active.any():
            return 0.0, {"progress": 0.0, "goal_fraction": 0.0,
                         "coverage": float(self.visited.mean()), "separation_penalty": 0.0}
        distance = np.linalg.norm(self.goals - p1, axis=1)
        progress = np.linalg.norm(self.goals - p0, axis=1) - distance
        reached = (distance < 0.05).astype(np.float32)
        margin = np.minimum(p1, 1 - p1).min(1)
        boundary = np.clip((0.03 - margin) / 0.03, 0, 1)
        proximity = np.zeros(self.n, dtype=np.float32)
        live = np.flatnonzero(active)
        if len(live) > 1:
            pd = np.linalg.norm(p1[live, None, :] - p1[None, live, :], axis=-1)
            np.fill_diagonal(pd, np.inf)
            proximity[live] = np.clip((0.04 - pd.min(1)) / 0.04, 0, 1)
        cell = np.minimum((p1 * self.cfg.grid).astype(int), self.cfg.grid - 1)
        new_cells = set((int(cell[i, 0]), int(cell[i, 1])) for i in live)
        new_count = sum(not self.visited[c] for c in new_cells)
        per_unit = 20 * progress + 0.05 * reached - 0.1 * boundary - 0.05 * proximity
        shared = float(per_unit[active].mean() + 0.08 * new_count / len(live))
        metrics = {"progress": float(progress[active].mean()),
                   "goal_fraction": float(reached[active].mean()),
                   "coverage": float((self.visited.sum() + new_count) / self.visited.size),
                   "separation_penalty": float(proximity[active].mean())}
        return shared, metrics


class Actor(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(cfg.features, cfg.hidden), nn.Tanh(),
                                 nn.Linear(cfg.hidden, cfg.hidden), nn.Tanh(),
                                 nn.Linear(cfg.hidden, len(HEADINGS)))
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, math.sqrt(2))
                nn.init.zeros_(layer.bias)
        nn.init.orthogonal_(self.net[-1].weight, 0.01)

    def forward(self, observation):
        return self.net(observation)


class CreditCritic(nn.Module):
    """State-only attention, then Q affine in joint action distributions.

    This makes policy marginalization exact for the represented Q (not for the
    true environment Q). State-conditioned action coefficients improve capacity
    while preserving this property. It cannot represent arbitrary action synergy.
    """
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        h = cfg.hidden
        self.embed = nn.Linear(cfg.features, h)
        self.attn = nn.MultiheadAttention(h, cfg.heads, dropout=0.0, batch_first=True)
        self.norm1, self.norm2 = nn.LayerNorm(h), nn.LayerNorm(h)
        self.ff = nn.Sequential(nn.Linear(h, h * 2), nn.Tanh(), nn.Linear(h * 2, h))
        self.coefficient = nn.Linear(h, len(HEADINGS))
        self.bias = nn.Linear(h, 1)

    def encode(self, obs, alive):
        n = obs.shape[1]
        if n == 0:
            raise ValueError("Critic requires at least one unit slot")
        keys = alive.clone()
        keys[~keys.any(1), 0] = True  # all-dead batch rows: avoid softmax(-inf)
        z = torch.tanh(self.embed(obs))
        attended, weights = self.attn(z, z, z, key_padding_mask=~keys)
        z = self.norm1(z + attended)
        z = self.norm2(z + self.ff(z)) * alive.unsqueeze(-1)
        count = alive.sum(1).clamp_min(1).to(z.dtype)
        pooled = z.sum(1) / count[:, None]
        bias = self.bias(pooled).squeeze(-1) * alive.any(1)
        coefficients = self.coefficient(z) * alive.unsqueeze(-1) / count[:, None, None]
        # A single attention block's residual rollout; state-only CorrSet.
        eye = torch.eye(n, device=obs.device).unsqueeze(0)
        rollout = (weights + eye) * alive[:, None, :] * alive[:, :, None]
        rollout = rollout / rollout.sum(-1, keepdim=True).clamp_min(1e-8)
        corr = ((rollout >= self.cfg.corr_threshold) | eye.bool())
        corr = corr & alive[:, :, None] & alive[:, None, :]
        return bias, coefficients, corr

    def values(self, obs, alive, actions, probabilities):
        bias, c, corr = self.encode(obs, alive)
        taken = c.gather(-1, actions.unsqueeze(-1)).squeeze(-1)
        expected = (c * probabilities).sum(-1)
        q = bias + taken.sum(-1)
        v = bias + expected.sum(-1)
        difference = expected - taken
        individual = q[:, None] + difference
        correlated = q[:, None] + (corr.to(c.dtype) * difference[:, None, :]).sum(-1)
        baselines = torch.stack([v[:, None].expand_as(individual), individual, correlated], -1)
        w = torch.as_tensor(self.cfg.credit_weights, dtype=c.dtype, device=c.device)
        baseline = (baselines * (w / w.sum())).sum(-1)
        return q, v, baseline, corr


def lambda_returns(rewards, values, terminals, gamma, lam):
    """Finite navigation episodes; terminal horizon has no bootstrap value."""
    rewards, values = np.asarray(rewards), np.asarray(values)
    if len(values) != len(rewards) + 1 or len(terminals) != len(rewards):
        raise ValueError("Incorrect GAE trajectory lengths")
    advantage = np.zeros(len(rewards), dtype=np.float32)
    acc = 0.0
    for t in reversed(range(len(rewards))):
        continuation = 1.0 - float(terminals[t])
        delta = rewards[t] + gamma * values[t + 1] * continuation - values[t]
        acc = delta + gamma * lam * continuation * acc
        advantage[t] = acc
    return advantage + values[:-1]


def _pack_numbers(value):
    # torch 1.13's weights_only loader cannot decode pickle BINFLOAT (opcode 71).
    # Preserve float precision using explicitly tagged float64 tensors instead.
    if isinstance(value, float):
        return {"__super5_float64__": torch.tensor(value, dtype=torch.float64)}
    if isinstance(value, dict):
        return {k: _pack_numbers(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_pack_numbers(v) for v in value)
    return value


def _unpack_numbers(value):
    if isinstance(value, dict):
        if set(value) == {"__super5_float64__"}:
            return float(value["__super5_float64__"].item())
        return {k: _unpack_numbers(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_unpack_numbers(v) for v in value)
    return value


def load_checkpoint(path, device="cpu"):
    # Tensor / primitive-only checkpoint: no custom class unpickling is needed.
    data = _unpack_numbers(torch.load(str(path), map_location=device, weights_only=True))
    if data.get("format_version") != FORMAT_VERSION or data.get("purpose") not in ("nonweapon_navigation", "combat"):
        raise ValueError("Unsupported checkpoint format or purpose")
    cfg = Config(**data["config"]).validate()
    return data, cfg


def atomic_save(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "wb") as stream:
            torch.save(_pack_numbers(data), stream)
            stream.flush()
            os.fsync(stream.fileno())
        for attempt in range(8):
            try:
                os.replace(temporary, str(path))
                break
            except PermissionError as exc:
                # Windows scanners/indexers may briefly hold the destination.
                # Retry the same atomic operation; never delete the old checkpoint.
                if os.name != "nt" or getattr(exc, "winerror", None) not in (5, 32, 33) or attempt == 7:
                    raise
                time.sleep(min(0.05 * 2 ** attempt, 1.0))
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


class Agent4Tactics:
    """Agent4 tactics embedded locally; the original opponent stays unchanged."""
    def __init__(self):
        self.size_x = self.size_y = 1000
        self.detector_num = self.fighter_num = 0
        self.long_range, self.short_range = 120, 50
        self.fire_interval = 2
        self.salvo_cap = 2
        self.shot_wait = 8
        self.formation_width = 100
        self.lead_steps = 4
        self.reserve_release_step = 600
        self._reset()

    def _reset(self):
        self.last_step = None
        self.tracks = {}
        self.last_shot = {}
        self.pending = []
        self.side = 1

    def get_obs_ind(self):
        return 'raw'

    def set_map_info(self, size_x, size_y, detector_num, fighter_num):
        if min(size_x, size_y) <= 0 or min(detector_num, fighter_num) < 0:
            raise ValueError('Invalid map dimensions or unit counts')
        self.size_x, self.size_y = size_x, size_y
        self.detector_num, self.fighter_num = int(detector_num), int(fighter_num)
        self.enemy_unit_count = self.detector_num + self.fighter_num
        self._reset()

    @staticmethod
    def _distance(unit, target):
        return math.hypot(target['pos_x'] - unit['pos_x'], target['pos_y'] - unit['pos_y'])

    def _course(self, unit, x, y):
        x = min(self.size_x - 20, max(20, x))
        y = min(self.size_y - 20, max(20, y))
        dx, dy = x - unit['pos_x'], y - unit['pos_y']
        return int(math.degrees(math.atan2(dy, dx))) % 360 if dx or dy else int(unit['course']) % 360

    def _update_tracks(self, obs, step):
        joint = obs.get('joint_obs_dict', {})
        alive = joint.get('alive_status_enemy_list')
        seen = {}
        for target in joint.get('passive_detection_enemy_list', []):
            seen[int(target['id'])] = target
        for unit in obs['detector_obs_list'] + obs['fighter_obs_list']:
            if unit['alive']:
                for target in unit.get('r_visible_list', []):
                    seen[int(target['id'])] = target
        for eid, target in seen.items():
            old = self.tracks.get(eid)
            vx = vy = 0.
            if old and step > old['step']:
                dt = step - old['step']
                vx = (target['pos_x'] - old['pos_x']) / dt
                vy = (target['pos_y'] - old['pos_y']) / dt
                # Bound extrapolation after intermittent observation.
                speed = math.hypot(vx, vy)
                if speed > 4:
                    vx, vy = vx * 4 / speed, vy * 4 / speed
            self.tracks[eid] = dict(target, vx=vx, vy=vy, step=step)
        self.tracks = {eid: t for eid, t in self.tracks.items()
                       if step - t['step'] <= 25 and
                       (alive is None or eid > len(alive) or alive[eid - 1])}
        self.pending = [p for p in self.pending if p[1] > step and
                        (alive is None or p[0] > len(alive) or alive[p[0] - 1])]

    def _move(self, unit, idx, step, assigned):
        targets = list(self.tracks.values())
        if targets:
            target = min(targets, key=lambda t: self._distance(unit, t) +
                         18 * assigned[int(t['id'])] + 2 * (step - t['step']))
            assigned[int(target['id'])] += 1
            dist = self._distance(unit, target)
            if unit.get('l_missile_left', 0) + unit.get('s_missile_left', 0) <= 0:
                # Empty fighters survive and keep sharing detections.
                if dist < 170:
                    dx, dy = unit['pos_x'] - target['pos_x'], unit['pos_y'] - target['pos_y']
                    length = max(1., math.hypot(dx, dy))
                    return self._course(unit, unit['pos_x'] + dx / length * 150,
                                        unit['pos_y'] + dy / length * 150)
            age = step - target['step']
            lead = min(12, age + self.lead_steps)
            return self._course(unit, target['pos_x'] + target['vx'] * lead,
                                target['pos_y'] + target['vy'] * lead)
        # Concentrate early to fight the opponent's straight, dispersed charge.
        # Later patrol separate lanes instead of flying indefinitely into a wall.
        slot = (idx / max(1, self.fighter_num - 1) - .5)
        if step < 180:
            x = self.size_x * (.65 if self.side == 1 else .35)
            y = self.size_y * .5 + slot * self.formation_width
        else:
            phase = (step // 100 + idx // 3) % 4
            x = self.size_x * (.2 if phase in (0, 3) else .8)
            y = self.size_y * (.25 if phase in (0, 1) else .75) + slot * 100
        return self._course(unit, x, y)

    @staticmethod
    def _jam(unit):
        frequencies = [int(s['r_fp']) for s in unit.get('j_recv_list', [])
                       if 1 <= int(s.get('r_fp', 0)) <= 10]
        if frequencies:
            fp, count = Counter(frequencies).most_common(1)[0]
            if count * 2 > len(frequencies):
                return fp
        return 11

    def _attacks(self, units, step):
        result = np.zeros(len(units), dtype=np.int32)
        load = Counter(eid for eid, expiry in self.pending)
        options = []
        offset = self.enemy_unit_count
        for idx, unit in enumerate(units):
            if not unit['alive'] or step - self.last_shot.get(idx, -1000) < self.fire_interval:
                continue
            for target in unit.get('r_visible_list', []):
                eid = int(target['id'])
                if not 1 <= eid <= offset:
                    continue
                dist = self._distance(unit, target)
                short, long = unit.get('s_missile_left', 0), unit.get('l_missile_left', 0)
                # Keep a small short-range reserve, released in danger/endgame.
                if dist <= self.short_range and short > 0 and (
                        short > 1 or dist <= 35 or step >= self.reserve_release_step):
                    options.append((dist - 100, idx, eid, eid + offset))
                elif dist <= self.long_range and long > 0:
                    options.append((dist, idx, eid, eid))
        # Best firing opportunities first; cap recent salvos across the team.
        for score, idx, eid, command in sorted(options):
            if result[idx] or load[eid] >= self.salvo_cap:
                continue
            result[idx] = command
            load[eid] += 1
            self.last_shot[idx] = step
            self.pending.append((eid, step + self.shot_wait))
        return result

    def get_action(self, obs_dict, step_cnt):
        units = obs_dict.get('fighter_obs_list', [])
        detectors = obs_dict.get('detector_obs_list', [])
        if len(units) != self.fighter_num or len(detectors) != self.detector_num:
            raise ValueError('Observation unit counts do not match set_map_info')
        if self.last_step is None or step_cnt <= self.last_step:
            self._reset()
            living = [u for u in units + detectors if u['alive']]
            if living:
                self.side = 1 if np.mean([u['pos_x'] for u in living]) < self.size_x / 2 else 2
        self.last_step = step_cnt
        self._update_tracks(obs_dict, step_cnt)
        fighter_actions = np.zeros((self.fighter_num, 4), dtype=np.int32)
        detector_actions = np.zeros((self.detector_num, 2), dtype=np.int32)
        attacks = self._attacks(units, step_cnt)
        assigned = Counter()
        for idx, unit in enumerate(units):
            if unit['alive']:
                # Radar action is a frequency, NOT an angular sector. Hop every step
                # so agent2 cannot reliably jam the next step's frequency.
                frequency = 1 + (step_cnt * 7 + idx * 3) % 10
                fighter_actions[idx] = [self._move(unit, idx, step_cnt, assigned),
                                        frequency, self._jam(unit), attacks[idx]]
        for idx, unit in enumerate(detectors):
            if unit['alive']:
                x = self.size_x * (.3 if self.side == 1 else .7)
                y = self.size_y * (.3 if (step_cnt // 100 + idx) % 2 else .7)
                detector_actions[idx] = [self._course(unit, x, y), 1 + (step_cnt + idx * 3) % 10]
        return detector_actions, fighter_actions


def tactical_actions(tactics, obs, step, residual_indices=None):
    """Rules own combat channels; policy may adjust armed search headings by <=12 degrees.

    Residual indices are the PPO actions (not absolute headings). Tracking an
    enemy or exhausting ammunition disables residuals; detectors retain rules.
    """
    d, f = tactics.get_action(obs, step)
    if residual_indices is not None and not tactics.tracks:
        offsets = np.array([0] + list(range(-12, 0)) + list(range(1, 12)), dtype=np.int32)
        for i, unit in enumerate(obs['fighter_obs_list']):
            if unit['alive'] and unit.get('l_missile_left', 0) + unit.get('s_missile_left', 0) > 0:
                f[i, 0] = (f[i, 0] + offsets[residual_indices[tactics.detector_num + i]]) % 360
    assert_safe_actions(d, f, enemy_count=tactics.enemy_unit_count)
    return d, f


# Existing callers, including fight.py, now get the tactical floor.
class Agent:
    """Agent4 tactical floor by default; learned residual control is opt-in."""
    def __init__(self, checkpoint=None, device="cpu", deterministic=True, config=None,
                 combat_control="agent4"):
        if combat_control not in ('agent4', 'residual', 'legacy'):
            raise ValueError('Invalid combat_control')
        self.combat_control = combat_control
        self.tactics = Agent4Tactics()
        self.device = select_device(device)
        self.deterministic = bool(deterministic)
        self.cfg = (config or Config()).validate()
        self.state = None
        self.loaded_checkpoint = None
        chosen = checkpoint
        if chosen is None and combat_control != 'agent4':
            chosen = os.environ.get("SUPER5_CHECKPOINT")
        if chosen is None and combat_control != 'agent4' and (DEFAULT_RUN / "best.pt").is_file():
            chosen = DEFAULT_RUN / "best.pt"
        if chosen is not None:
            data, self.cfg = load_checkpoint(chosen, self.device)
            with preserve_random_state():
                self.actor = Actor(self.cfg).to(self.device)
            self.actor.load_state_dict(data["actor"])
            self.loaded_checkpoint = str(Path(chosen).resolve())
        else:
            with preserve_random_state():
                self.actor = Actor(self.cfg).to(self.device)
            if combat_control != 'agent4':
                warnings.warn("No super5 checkpoint: network is untrained.", RuntimeWarning, stacklevel=2)
        if combat_control == 'residual' and (
                chosen is None or self.cfg.combat_control != 'residual' or data.get('updates', 0) < 1):
            raise ValueError('Residual inference requires a trained residual checkpoint')
        self.actor.eval()

    def get_obs_ind(self):
        return "raw"

    def set_map_info(self, size_x, size_y, detector_num, fighter_num, enemy_unit_count=None):
        self.size_x, self.size_y = size_x, size_y
        self.detector_num, self.fighter_num = int(detector_num), int(fighter_num)
        self.state = NavigationState(self.cfg, size_x, size_y, detector_num, fighter_num)
        self.enemy_unit_count = self.state.n if enemy_unit_count is None else enemy_unit_count
        if (not isinstance(self.enemy_unit_count, (int, np.integer)) or
                not 0 <= self.enemy_unit_count <= 64):
            raise ValueError("enemy_unit_count must be an integer in [0, 64]")
        self.tactics.set_map_info(size_x, size_y, detector_num, fighter_num)
        self.tactics.enemy_unit_count = self.enemy_unit_count

    def reset(self):
        self.tactics._reset()
        if self.state is not None:
            self.state.reset(self.state.goals)

    def set_navigation_goals(self, goals_xy):
        if self.state is None:
            raise RuntimeError("Call set_map_info before setting navigation goals")
        self.state.reset(np.asarray(goals_xy, dtype=np.float32) / self.state.scale)

    def get_action(self, obs_dict, step_cnt):
        if self.state is None:
            raise RuntimeError("Call set_map_info before get_action")
        if not isinstance(step_cnt, (int, np.integer)) or step_cnt < 0:
            raise ValueError("step_cnt must be a nonnegative integer")
        features, alive = self.state.observe(obs_dict, step_cnt)
        if self.combat_control == 'agent4':
            commands = self.tactics.get_action(obs_dict, step_cnt)
            assert_safe_actions(*commands, enemy_count=self.enemy_unit_count)
            return commands
        if not alive.any():
            return safe_actions(np.zeros(self.state.n, dtype=np.int64), alive, self.detector_num)
        with torch.no_grad():
            logits = self.actor(torch.as_tensor(features, device=self.device))
            actions = logits.argmax(-1) if self.deterministic else Categorical(logits=logits).sample()
        if self.combat_control == 'residual':
            return tactical_actions(self.tactics, obs_dict, step_cnt, actions.cpu().numpy())
        # Explicit legacy mode preserves absolute network headings.
        return safe_actions(actions.cpu().numpy(), alive, self.detector_num,
                            obs=obs_dict, enemy_count=self.enemy_unit_count)


class NavigationAgent:
    """Heading-only inference. No checkpoint means deterministic waypoint guidance.

    Checkpoints are explicit: neither environment variables nor default run files
    override this choice. Invalid checkpoints fail instead of silently falling back.
    """
    def __init__(self, checkpoint=None, device="cpu", config=None):
        self.device = select_device(device)
        self.cfg = (config or Config()).validate()
        self.state = self.actor = self.loaded_checkpoint = None
        self.policy_source = "waypoint"
        if checkpoint is not None:
            path = Path(checkpoint).expanduser().resolve()
            data, self.cfg = load_checkpoint(path, self.device)
            with preserve_random_state():
                actor = Actor(self.cfg).to(self.device)
            actor.load_state_dict(data["actor"])
            if any(not torch.isfinite(p).all() for p in actor.parameters()):
                raise ValueError("Checkpoint contains non-finite actor weights")
            self.loaded_checkpoint = str(path)
            updates = data.get("updates")
            if not isinstance(updates, int) or isinstance(updates, bool) or updates < 0:
                raise ValueError("Checkpoint updates must be a nonnegative integer")
            if updates:
                self.actor = actor.eval()
                self.policy_source = "checkpoint"

    def set_map_info(self, size_x, size_y, detector_num, fighter_num):
        self.state = NavigationState(self.cfg, size_x, size_y, detector_num, fighter_num)

    def get_obs_ind(self):
        return "raw"

    def reset(self):
        if self.state is not None:
            self.state.reset(self.state.goals)

    def set_navigation_goals(self, goals_xy):
        if self.state is None:
            raise RuntimeError("Call set_map_info before setting navigation goals")
        self.state.reset(np.asarray(goals_xy, dtype=np.float32) / self.state.scale)

    def get_action(self, obs_dict, step_cnt):
        if self.state is None:
            raise RuntimeError("Call set_map_info before get_action")
        if (isinstance(step_cnt, bool) or not isinstance(step_cnt, (int, np.integer))
                or step_cnt < 0):
            raise ValueError("step_cnt must be a nonnegative integer")
        features, alive = self.state.observe(obs_dict, step_cnt)
        if self.actor is None:
            # Convert normalized goal deltas to map coordinates (rectangular maps).
            delta = features[:, 6:8] * self.state.scale
            angle = np.degrees(np.arctan2(delta[:, 1], delta[:, 0])) % 360
            indices = (np.floor(angle / 15 + 0.5).astype(np.int64) % len(HEADINGS))
        else:
            with torch.no_grad():
                logits = self.actor(torch.as_tensor(features, device=self.device))
                if not torch.isfinite(logits).all():
                    raise FloatingPointError("Non-finite navigation logits")
                indices = logits.argmax(-1).cpu().numpy()
        return safe_actions(indices, alive, self.state.detectors)


class MaCANavigationEnv:
    """Native MaCA with zero ammunition map and two heading-only action streams."""
    def __init__(self, cfg, seed, side=0, render=False):
        self.cfg, self.side = cfg, side
        if side not in (0, 1):
            raise ValueError("side must be 0 or 1")
        self.tmp = tempfile.TemporaryDirectory(prefix="super5_navigation_")
        self.render = render
        try:
            source = json.loads(resolve_map(cfg.map_path).read_text(encoding="utf-8-sig"))
            for team in ("side1", "side2"):
                for unit in source[team + "_fighter_list"]:
                    unit["l_missile_num"] = unit["s_missile_num"] = 0
            path = Path(self.tmp.name) / "navigation.map"
            path.write_text(json.dumps(source), encoding="utf-8")
            for entry in (str(ROOT), str(ROOT / "environment")):
                if entry not in sys.path:
                    sys.path.insert(0, entry)
            from interface import Environment
            random.seed(seed)
            np.random.seed(seed)
            self.env = Environment(str(path), "raw", "raw", max_step=cfg.episode_steps,
                                   render=render, random_pos=True, random_seed=seed, log=False)
            self.size = self.env.get_map_size()
            self.counts = self.env.get_unit_num()
            self.detectors, self.fighters = self.counts[side * 2:side * 2 + 2]
            self.steps = 0
            self.observations = self.env.get_obs()
        except Exception:
            self.tmp.cleanup()
            raise

    def observation(self):
        return self.observations[self.side]

    def step(self, actions):
        assert_safe_actions(*actions)
        other = 1 - self.side
        od, of = self.counts[2 * other:2 * other + 2]
        other_obs = self.observations[other]
        units = list(other_obs["detector_obs_list"]) + list(other_obs["fighter_obs_list"])
        # The unused team is an inert navigation background, not an opponent policy.
        indices = np.array([int(u.get("course", 0)) % 360 // 15 for u in units], dtype=np.int64)
        passive = safe_actions(indices, [u["alive"] for u in units], od)
        streams = [actions, passive] if self.side == 0 else [passive, actions]
        # Empty ammunition triggers MaCA's combat termination after one tick.
        # This adapter defines a separate finite navigation task. Clear only that
        # engine stop flag so movement continues; never restore any ammunition.
        self.env.env.done = False
        result = self.env.step(*streams[0], *streams[1])
        if result is False:
            raise RuntimeError("Native MaCA rejected a navigation action")
        self.steps += 1
        self.observations = self.env.get_obs()
        obs = self.observation()
        alive = any(u["alive"] for u in obs["detector_obs_list"] + obs["fighter_obs_list"])
        # Horizon is the end of this finite navigation task, not a rollout cutoff.
        return obs, bool(self.steps >= self.cfg.episode_steps or not alive)

    def close(self):
        if self.render:
            import pygame
            pygame.display.quit()
        self.tmp.cleanup()


class ToyNavigationEnv:
    """Explicit lightweight point-navigation test backend, NOT a MaCA substitute."""
    def __init__(self, cfg, seed, side=0, render=False):
        if render:
            raise ValueError("Toy backend does not render")
        self.cfg = cfg
        self.size = (1000, 1000)
        self.detectors, self.fighters = 0, cfg.toy_units
        rng = np.random.RandomState(seed)
        self.pos = rng.uniform(0.12, 0.88, size=(self.fighters, 2)).astype(np.float32)
        self.course = rng.randint(0, 360, self.fighters)
        self.steps = 0

    def observation(self):
        units = [dict(id=i + 1, alive=True, pos_x=float(p[0] * 1000),
                      pos_y=float(p[1] * 1000), course=int(self.course[i]))
                 for i, p in enumerate(self.pos)]
        return dict(detector_obs_list=[], fighter_obs_list=units, joint_obs_dict={})

    def step(self, actions):
        assert_safe_actions(*actions)
        self.course = actions[1][:, 0].copy()
        radians = np.deg2rad(self.course)
        self.pos += np.stack([np.cos(radians), np.sin(radians)], -1) * 0.01
        self.pos = np.clip(self.pos, 0, 1)
        self.steps += 1
        return self.observation(), self.steps >= self.cfg.episode_steps

    def close(self):
        pass


class MaCACombatEnv:
    """Original armed map, native termination, fixed observable-rule opponent."""
    def __init__(self, cfg, seed, side=0, render=False):
        if side not in (0, 1):
            raise ValueError("side must be 0 or 1")
        for entry in (str(ROOT), str(ROOT / "environment")):
            if entry not in sys.path:
                sys.path.insert(0, entry)
        from interface import Environment
        from agent.My_agent.super_agent4 import Agent as Agent4
        random.seed(seed)
        np.random.seed(seed)
        self.cfg, self.side, self.render = cfg, side, render
        self.env = Environment(str(resolve_map(cfg.map_path)), "raw", "raw",
                               max_step=cfg.episode_steps, random_seed=seed,
                               random_pos=cfg.random_pos, render=render, log=False)
        self.size, self.counts = self.env.get_map_size(), self.env.get_unit_num()
        self.detectors, self.fighters = self.counts[2 * side:2 * side + 2]
        self.enemy_count = sum(self.counts[2 * (1-side):2 * (1-side) + 2])
        self.opponent = Agent4()
        self.opponent.set_map_info(*self.size, *self.counts[2*(1-side):2*(1-side)+2])
        self.observations = self.env.get_obs()
        self.steps = self.shots = self.jams = self.ammo_used = 0
        self.reward_totals = {}

    @staticmethod
    def units(obs):
        return obs['detector_obs_list'] + obs['fighter_obs_list']

    def observation(self):
        return self.observations[self.side]

    def step(self, actions):
        assert_safe_actions(*actions, enemy_count=self.enemy_count)
        other = 1 - self.side
        obs = self.observations[other]
        if self.cfg.mode == "adversarial":
            response = self.opponent.get_action(obs, self.steps + 1)
        else:
            units = self.units(obs)
            response = safe_actions(np.array([int(u['course']) % 360 // 15 for u in units]),
                                    [u['alive'] for u in units], self.counts[2*other],
                                    obs, self.detectors + self.fighters)
        streams = [actions, response] if self.side == 0 else [response, actions]
        self.before = self.observations
        if self.env.step(*streams[0], *streams[1]) is False:
            raise RuntimeError("Native MaCA rejected combat actions")
        self.steps += 1
        self.shots += int(np.count_nonzero(actions[1][:, 3]))
        self.last_jams = int(np.count_nonzero(actions[1][:, 2]))
        self.jams += self.last_jams
        self.observations = self.env.get_obs()
        return self.observation(), bool(self.env.get_done() or self.steps >= self.cfg.episode_steps)

    def reward(self, navigation_reward, terminal):
        own0, own1 = self.units(self.before[self.side]), self.units(self.observation())
        enemy0 = self.units(self.before[1-self.side])
        enemy1 = self.units(self.observations[1-self.side])
        alive = lambda units: sum(bool(u['alive']) for u in units)
        # Only units alive before AND after count ammunition deltas: destruction
        # must never be mistaken for firing all remaining missiles.
        used = sum(max(0, a.get(k, 0)-b.get(k, 0)) for a, b in zip(own0, own1)
                   if a['alive'] and b['alive'] for k in ('l_missile_left', 's_missile_left'))
        self.ammo_used += used
        n, en = max(1, len(own0)), max(1, len(enemy0))
        scores = self.env.get_reward()
        outcome = int(np.sign(scores[self.side*3+2] - scores[(1-self.side)*3+2]))
        cfg = self.cfg
        terms = dict(survival=cfg.survival_weight * (alive(own1)/n/cfg.episode_steps - (alive(own0)-alive(own1))/n),
                     task=cfg.task_weight * navigation_reward,
                     kills=cfg.kill_weight * (alive(enemy0)-alive(enemy1))/en,
                     resources=-cfg.resource_weight * (used + self.last_jams/cfg.episode_steps)/n,
                     outcome=cfg.outcome_weight * outcome if terminal else 0.0)
        for k, v in terms.items():
            self.reward_totals[k] = self.reward_totals.get(k, 0.0) + v
        return sum(terms.values()), terms

    def summary(self):
        scores = self.env.get_reward()
        delta = scores[self.side*3+2] - scores[(1-self.side)*3+2]
        return dict(result='win' if delta > 0 else 'loss' if delta < 0 else 'draw',
                    survivors=sum(u['alive'] for u in self.units(self.observation())),
                    enemy_survivors=sum(u['alive'] for u in self.units(self.observations[1-self.side])),
                    attack_commands=self.shots, interference_commands=self.jams,
                    missiles_consumed_surviving_units=self.ammo_used,
                    native_done=bool(self.env.get_done()), reward_components=self.reward_totals)

    def close(self):
        if self.render:
            import pygame
            pygame.display.quit()


def make_env(cfg, seed, side=0, render=False):
    if cfg.mode != "navigation":
        return MaCACombatEnv(cfg, int(seed) % (2 ** 31), side, render)
    cls = MaCANavigationEnv if cfg.backend == "maca" else ToyNavigationEnv
    return cls(cfg, int(seed) % (2 ** 31), side, render)


def navigation_goals(state, obs, seed, difficulty):
    """Task generator only; never decides an action or uses another team."""
    rng = np.random.RandomState(seed % (2 ** 31))
    # Half the tasks are independently sampled waypoints; half are circle formations.
    if rng.rand() < 0.5:
        target = rng.uniform(0.1, 0.9, (state.n, 2))
    else:
        target = state.default_goals().astype(np.float64)
        target += rng.uniform(-0.1, 0.1, (1, 2))
    pos, _, _ = state.read(obs)
    target = pos + (0.15 + 0.85 * difficulty) * (target - pos)
    return np.clip(target, 0.05, 0.95).astype(np.float32)


@contextmanager
def preserve_random_state():
    py, numpy_state, cpu = random.getstate(), np.random.get_state(), torch.get_rng_state()
    cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
    try:
        yield
    finally:
        random.setstate(py)
        np.random.set_state(numpy_state)
        torch.set_rng_state(cpu)
        if cuda:
            torch.cuda.set_rng_state_all(cuda)


def collect_episode(actor, critic, cfg, device, seed, difficulty=1.0,
                    deterministic=False, render=False, side=0, baseline_agent=None):
    env = make_env(cfg, seed, side, render)
    was_training = actor.training
    actor.eval()
    try:
        state = NavigationState(cfg, *env.size, env.detectors, env.fighters)
        if state.n == 0:
            raise ValueError("Training / evaluation requires a nonempty controlled team")
        obs = env.observation()
        tactics = None
        if cfg.mode != 'navigation' and cfg.combat_control != 'legacy':
            tactics = Agent4Tactics()
            tactics.set_map_info(*env.size, env.detectors, env.fighters)
            tactics.enemy_unit_count = env.enemy_count
        if baseline_agent is not None:
            baseline_agent.set_map_info(*env.size, env.detectors, env.fighters)
        state.reset(navigation_goals(state, obs, seed + 31, difficulty))
        features, alive = state.observe(obs, 0)
        records = {k: [] for k in ("obs", "alive", "actions", "log_prob", "reward",
                                   "value", "baseline", "terminal")}
        total_return, metric_sums = 0.0, {}
        for t in range(cfg.episode_steps):
            xt = torch.as_tensor(features, device=device).unsqueeze(0)
            mt = torch.as_tensor(alive, device=device).unsqueeze(0)
            with torch.no_grad():
                dist = Categorical(logits=actor(xt))
                actions = dist.logits.argmax(-1) if deterministic else dist.sample()
                if critic is not None:
                    _, value, baseline, _ = critic.values(xt, mt, actions, dist.probs)
                    value = value.item()
                    baseline = baseline[0].cpu().numpy()
                else:
                    value, baseline = 0.0, np.zeros(state.n, dtype=np.float32)
            action_np = actions[0].cpu().numpy()
            combat = cfg.mode != "navigation"
            commands = safe_actions(action_np, alive, env.detectors,
                                    obs if combat else None, env.enemy_count if combat else None)
            if tactics is not None:
                commands = tactical_actions(tactics, obs, t + 1,
                                            action_np if cfg.combat_control == 'residual' else None)
            if baseline_agent is not None:
                commands = baseline_agent.get_action(obs, t + 1)
            following, terminal = env.step(commands)
            reward, metrics = state.reward(obs, following)
            if combat:
                reward, terms = env.reward(reward, terminal)
                metrics.update(terms)
            for key, entry in (("obs", features), ("alive", alive), ("actions", action_np),
                               ("log_prob", dist.log_prob(actions)[0].cpu().numpy()),
                               ("reward", reward), ("value", value),
                               ("baseline", baseline), ("terminal", terminal)):
                records[key].append(entry)
            total_return += reward
            for k, v in metrics.items():
                metric_sums[k] = metric_sums.get(k, 0.0) + v
            obs = following
            features, alive = state.observe(obs, t + 1)
            if terminal:
                break
        records["terminal"][-1] = True
        returns = lambda_returns(records["reward"], records["value"] + [0.0],
                                 records["terminal"], cfg.gamma, cfg.gae_lambda)
        batch = {k: np.asarray(v) for k, v in records.items()}
        batch["returns"] = returns.astype(np.float32)
        batch["advantages"] = returns[:, None] - batch["baseline"]
        count = len(records["reward"])
        summary = {k: v / count for k, v in metric_sums.items()}
        summary.update(episode_return=float(total_return), steps=count,
                       final_coverage=float(state.visited.mean()), seed=int(seed), side=side,
                       attack_commands=0, interference_commands=0)
        if cfg.mode != "navigation":
            summary.update(env.summary())
            summary['combat_control'] = cfg.combat_control
        return batch, summary
    finally:
        actor.train(was_training)
        env.close()


class Trainer:
    def __init__(self, cfg, device="cpu"):
        self.cfg = cfg.validate()
        self.device = select_device(str(device))
        torch.manual_seed(cfg.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(cfg.seed)
        self.actor, self.critic = Actor(cfg).to(self.device), CreditCritic(cfg).to(self.device)
        self.optimizer = torch.optim.Adam(list(self.actor.parameters()) + list(self.critic.parameters()),
                                          lr=cfg.learning_rate, eps=1e-5)
        self.updates, self.env_steps = 0, 0
        self.best_score = -float("inf")
        self.eval_seeds = [100003, 100019, 100043, 100049]
        self.source_digest = map_digest(cfg)

    def update(self, batch):
        cfg = self.cfg
        data = {k: torch.as_tensor(batch[k], device=self.device) for k in
                ("obs", "alive", "actions", "log_prob", "returns", "advantages")}
        data["obs"] = data["obs"].float()
        data["actions"] = data["actions"].long()
        mask = data["alive"].bool()
        if not mask.any():
            raise ValueError("No living agents in rollout")
        advantages = data["advantages"].float()
        valid = advantages[mask]
        advantages = (advantages - valid.mean()) / valid.std(unbiased=False).clamp_min(1e-6)
        if not torch.isfinite(advantages).all() or not torch.isfinite(data["returns"]).all():
            raise FloatingPointError("Non-finite rollout")
        metrics = []
        early_stop = False
        for epoch in range(cfg.epochs):
            order = torch.randperm(len(mask), device=self.device)
            for indices in order.split(cfg.batch_steps):
                live = mask[indices]
                if not live.any():
                    continue
                dist = Categorical(logits=self.actor(data["obs"][indices]))
                new_log = dist.log_prob(data["actions"][indices])
                log_ratio = new_log - data["log_prob"][indices]
                ratio = log_ratio.exp()
                unclipped = ratio * advantages[indices]
                clipped = ratio.clamp(1 - cfg.clip, 1 + cfg.clip) * advantages[indices]
                policy_loss = -torch.minimum(unclipped, clipped)[live].mean()
                entropy = dist.entropy()[live].mean()
                q, v, _, _ = self.critic.values(data["obs"][indices], live,
                                               data["actions"][indices], dist.probs.detach())
                target = data["returns"][indices]
                rows = live.any(1)
                value_loss = 0.5 * (F.mse_loss(q[rows], target[rows]) +
                                    F.mse_loss(v[rows], target[rows]))
                loss = policy_loss + cfg.value_coef * value_loss - cfg.entropy * entropy
                if not torch.isfinite(loss):
                    raise FloatingPointError("Non-finite optimization loss")
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                params = list(self.actor.parameters()) + list(self.critic.parameters())
                gradient = nn.utils.clip_grad_norm_(params, cfg.grad_clip)
                if not torch.isfinite(gradient):
                    raise FloatingPointError("Non-finite gradient; parameters were not updated")
                self.optimizer.step()
                kl = ((ratio - 1) - log_ratio)[live].mean().detach().item()
                metrics.append((policy_loss.item(), value_loss.item(), entropy.item(), kl))
                if kl > 0.03:
                    early_stop = True
                    break
            if early_stop:
                break
        self.updates += 1
        self.env_steps += len(mask)
        return dict(zip(("policy_loss", "value_loss", "entropy", "approx_kl"),
                        np.mean(metrics, axis=0).tolist()), early_stop=early_stop)

    def checkpoint(self):
        return dict(format_version=FORMAT_VERSION, purpose="nonweapon_navigation" if self.cfg.mode == "navigation" else "combat",
                    config=asdict(self.cfg), actor=self.actor.state_dict(), critic=self.critic.state_dict(),
                    optimizer=self.optimizer.state_dict(), updates=self.updates, env_steps=self.env_steps,
                    best_score=self.best_score, eval_seeds=self.eval_seeds, map_digest=self.source_digest,
                    torch_rng=torch.get_rng_state(),
                    cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])

    def restore(self, path):
        data, cfg = load_checkpoint(path, self.device)
        if asdict(cfg) != asdict(self.cfg) or data["map_digest"] != self.source_digest:
            raise ValueError("Checkpoint configuration or map content mismatch")
        self.actor.load_state_dict(data["actor"])
        self.critic.load_state_dict(data["critic"])
        self.optimizer.load_state_dict(data["optimizer"])
        self.updates, self.env_steps = int(data["updates"]), int(data["env_steps"])
        self.best_score, self.eval_seeds = float(data["best_score"]), list(data["eval_seeds"])
        torch.set_rng_state(data["torch_rng"].cpu())
        if data.get("cuda_rng") and self.device.type == "cuda":
            torch.cuda.set_rng_state_all([x.cpu() for x in data["cuda_rng"]])


def evaluate(actor, cfg, device, seeds, render=False):
    records = []
    with preserve_random_state():
        for seed in seeds:
            for side in ((0, 1) if cfg.backend == "maca" else (0,)):
                _, row = collect_episode(actor, None, cfg, device, seed,
                                         deterministic=True, render=render, side=side)
                records.append(row)
    returns = np.array([r["episode_return"] for r in records])
    return dict(mean_return=float(returns.mean()), std_return=float(returns.std()),
                episodes=len(records), games=records)


def emit_json(data, path=None):
    content = json.dumps(data, ensure_ascii=False, allow_nan=False)
    print(content, flush=True)
    if path is not None:
        with Path(path).open("a", encoding="utf-8") as stream:
            stream.write(content + "\n")


def train_command(args):
    if args.resume:
        _, cfg = load_checkpoint(args.resume)
    else:
        cfg = Config(backend=args.backend, map_path=str(resolve_map(args.map)), seed=args.seed,
                     episode_steps=args.episode_steps, hidden=args.hidden, toy_units=args.toy_units,
                     curriculum_updates=args.curriculum_updates).validate()
        if args.command == "combat-train":
            cfg.mode, cfg.random_pos = args.mode, args.random_pos
            cfg.combat_control = args.combat_control
            for key in ("survival_weight", "task_weight", "kill_weight", "resource_weight", "outcome_weight"):
                setattr(cfg, key, getattr(args, key))
            cfg.validate()
    if args.command == "combat-train" and cfg.mode == "navigation":
        raise ValueError("combat-train resume requires a combat checkpoint")
    if args.command == "train" and cfg.mode != "navigation":
        raise ValueError("Use combat-train to resume a combat checkpoint")
    out = Path(args.out).resolve() if args.out else (
        Path(args.resume).resolve().parent if args.resume else
        HERE / 'super5_runs' / 'combat' if args.command == 'combat-train' else DEFAULT_RUN)
    latest, best, log = out / "latest.pt", out / "best.pt", out / "metrics.jsonl"
    if not args.resume and any(p.exists() for p in (latest, best, log)):
        raise FileExistsError("Run directory contains results; use --resume or choose a new --out")
    if args.resume and latest.exists() and Path(args.resume).resolve() != latest.resolve():
        raise ValueError("Resume from latest.pt, or select a fresh --out for branching")
    out.mkdir(parents=True, exist_ok=True)
    trainer = Trainer(cfg, args.device)
    if args.resume:
        trainer.restore(args.resume)
    emit_json(dict(event="configuration", config=asdict(cfg), device=str(trainer.device),
                   out=str(out), resumed=bool(args.resume)), log)
    if not args.resume or not best.exists():
        report = evaluate(trainer.actor, cfg, trainer.device, trainer.eval_seeds)
        trainer.best_score = report["mean_return"]
        atomic_save(best, trainer.checkpoint())
        atomic_save(latest, trainer.checkpoint())
        emit_json(dict(event="initial_evaluation", update=trainer.updates, **report), log)
    start = time.monotonic()
    for local_update in range(args.updates):
        index = trainer.updates
        difficulty = min(1.0, (index + 1) / max(1, cfg.curriculum_updates))
        if cfg.curriculum_updates == 0:
            difficulty = 1.0
        seed = (cfg.seed + 104729 * index) % (2 ** 31)
        batch, summary = collect_episode(trainer.actor, trainer.critic, cfg, trainer.device,
                                         seed, difficulty, side=index % 2)
        metrics = trainer.update(batch)
        emit_json(dict(event="training", update=trainer.updates, env_steps=trainer.env_steps,
                       difficulty=difficulty, elapsed_seconds=round(time.monotonic() - start, 2),
                       **summary, **metrics), log)
        if trainer.updates % args.eval_every == 0 or local_update == args.updates - 1:
            report = evaluate(trainer.actor, cfg, trainer.device, trainer.eval_seeds)
            improved = report["mean_return"] > trainer.best_score
            if improved:
                trainer.best_score = report["mean_return"]
                atomic_save(best, trainer.checkpoint())
            emit_json(dict(event="evaluation", update=trainer.updates, improved=improved,
                           best_return=trainer.best_score, **report), log)
        atomic_save(latest, trainer.checkpoint())
    emit_json(dict(event="complete", updates=trainer.updates, env_steps=trainer.env_steps,
                   latest=str(latest), best=str(best), best_return=trainer.best_score))


def evaluation_command(args):
    data, cfg = load_checkpoint(args.checkpoint)
    if cfg.mode != 'navigation':
        raise ValueError('Use combat-evaluate for combat checkpoints')
    if args.map is not None:
        cfg.map_path = str(resolve_map(args.map))
    if args.episode_steps is not None:
        cfg.episode_steps = args.episode_steps
    cfg.validate()
    device = select_device(args.device)
    actor = Actor(cfg).to(device)
    actor.load_state_dict(data["actor"])
    report = evaluate(actor, cfg, device, args.seeds, args.render)
    emit_json(dict(event="evaluation", checkpoint=str(Path(args.checkpoint).resolve()),
                   checkpoint_updates=data["updates"], backend=cfg.backend, **report))


def combat_evaluation_command(args):
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    if any(not 0 <= seed < 2**31 for seed in args.seeds):
        raise ValueError("seeds must be in [0, 2**31)")
    data, cfg = load_checkpoint(args.checkpoint) if args.checkpoint else (None, Config())
    cfg.backend, cfg.mode, cfg.random_pos = "maca", args.mode, args.random_pos
    if args.combat_control == 'residual' and (data is None or cfg.combat_control != 'residual'):
        raise ValueError('Residual evaluation requires a residual checkpoint')
    cfg.combat_control = args.combat_control
    cfg.map_path, cfg.episode_steps = args.map, args.episode_steps
    cfg.validate()
    device = select_device(args.device)
    with preserve_random_state():
        torch.manual_seed(cfg.seed)
        actor = Actor(cfg).to(device)
        if data:
            actor.load_state_dict(data['actor'])
        from agent.My_agent.super_agent4 import Agent as Agent4
        rows = []
        for seed in args.seeds:
            for side in (0, 1):
                for name in ('Agent5', 'Agent4'):
                    _, row = collect_episode(actor, None, cfg, device, seed, deterministic=True,
                                             side=side, baseline_agent=Agent4() if name == 'Agent4' else None)
                    rows.append(dict(candidate=name, **row))
    aggregates = {}
    for name in ('Agent5', 'Agent4'):
        games = [r for r in rows if r['candidate'] == name]
        aggregates[name] = dict(games=len(games), **{k: sum(r['result'] == k for r in games)
                                                   for k in ('win', 'loss', 'draw')})
        aggregates[name]['mean_return'] = float(np.mean([r['episode_return'] for r in games]))
    report = dict(protocol='paired-same-opponent-swapped-sides-v1', config=asdict(cfg),
                  seeds=args.seeds, map_sha256=map_digest(cfg),
                  checkpoint_sha256=hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest() if data else None,
                  checkpoint_updates=data['updates'] if data else 0,
                  policy_source='embedded-agent4' if cfg.combat_control == 'agent4' else
                                'checkpoint' if data else 'seeded-untrained-network',
                  source_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in
                                 (Path(__file__), HERE / 'super_agent4.py')},
                  runtime=dict(python=sys.version, numpy=np.__version__, torch=torch.__version__,
                               device=str(device), threads=torch.get_num_threads()),
                  opponent='Agent4' if cfg.mode == 'adversarial' else 'fixed-heading-self-defense',
                  summary=aggregates, games=rows)
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    emit_json(dict(report=str(path.resolve()), summary=aggregates))


def navigation_evaluation_command(args):
    """Compare explicit weights or waypoint baseline in the same unarmed task."""
    agent = NavigationAgent(args.checkpoint, args.device, Config(backend=args.backend))
    cfg = agent.cfg
    if cfg.mode != 'navigation':
        raise ValueError('Use combat-evaluate for combat checkpoints')
    if args.map is not None:
        cfg.map_path = str(resolve_map(args.map))
    if args.episode_steps is not None:
        cfg.episode_steps = args.episode_steps
    cfg.validate()
    rows = []
    with preserve_random_state():
        for seed in args.seeds:
            for side in ((0, 1) if cfg.backend == "maca" else (0,)):
                env = make_env(cfg, seed, side)
                try:
                    agent.set_map_info(*env.size, env.detectors, env.fighters)
                    state, obs = agent.state, env.observation()
                    if state.n == 0:
                        raise ValueError("Navigation evaluation requires at least one unit")
                    state.reset(navigation_goals(state, obs, seed + 31, 1.0))
                    start, _, alive = state.read(obs)
                    initial = float(np.linalg.norm(state.goals - start, axis=1)[alive].mean())
                    total, boundary, samples = 0.0, 0, 0
                    for step in range(cfg.episode_steps):
                        actions = agent.get_action(obs, step)
                        following, done = env.step(actions)
                        reward, _ = state.reward(obs, following)
                        total += reward
                        pos, _, alive = state.read(following)
                        boundary += int((np.minimum(pos, 1 - pos).min(1)[alive] < 0.03).sum())
                        samples += int(alive.sum())
                        obs = following
                        if done:
                            break
                    state.observe(obs, step + 1)
                    distance = np.linalg.norm(state.goals - pos, axis=1)[alive]
                    rows.append(dict(seed=seed, side=side, steps=step + 1,
                                     episode_return=total, initial_distance=initial,
                                     final_distance=float(distance.mean()) if len(distance) else None,
                                     final_goal_fraction=float((distance < 0.05).mean()) if len(distance) else 0.0,
                                     final_coverage=float(state.visited.mean()),
                                     boundary_fraction=boundary / max(1, samples)))
                finally:
                    env.close()
    emit_json(dict(event="navigation_evaluation", policy_source=agent.policy_source,
                   checkpoint=agent.loaded_checkpoint, backend=cfg.backend,
                   episodes=len(rows), mean_return=float(np.mean([r["episode_return"] for r in rows])),
                   distance_units="normalized map coordinates", games=rows))


def self_test(native=False, navigation_only=False):
    """Mathematical checks, real parameter updates, resume and interface tests."""
    class Tests(unittest.TestCase):
        def setUp(self):
            self.cfg = Config(backend="toy", hidden=32, heads=4, episode_steps=12,
                              toy_units=3, epochs=2, batch_steps=6)
            torch.manual_seed(13)

        @unittest.skipUnless(native and not navigation_only, 'native combat opt-in')
        def test_embedded_agent4_floor_and_residual_gates(self):
            cfg = Config(mode='adversarial', episode_steps=64)
            env = MaCACombatEnv(cfg, 11)
            from agent.My_agent.super_agent4 import Agent as Agent4
            reference = Agent4()
            candidate = Agent(config=cfg)
            reference.set_map_info(*env.size, env.detectors, env.fighters)
            candidate.set_map_info(*env.size, env.detectors, env.fighters)
            try:
                for step in range(1, 65):
                    obs = env.observation()
                    a, b = candidate.get_action(obs, step), reference.get_action(obs, step)
                    for x, y in zip(a, b):
                        np.testing.assert_array_equal(x, y)
                    _, done = env.step(a)
                    if done:
                        break
                candidate.reset()
                reference._reset()
                for x, y in zip(candidate.get_action(obs, 1), reference.get_action(obs, 1)):
                    np.testing.assert_array_equal(x, y)
                for threatened in (False, True):
                    import copy
                    sample = copy.deepcopy(obs)
                    for u in sample['fighter_obs_list']:
                        u['r_visible_list'] = ([dict(id=1, pos_x=u['pos_x']+20, pos_y=u['pos_y'])]
                                               if threatened else [])
                    sample['joint_obs_dict'] = {}
                    rules, residual = Agent4Tactics(), Agent4Tactics()
                    for obj in (rules, residual):
                        obj.set_map_info(*env.size, env.detectors, env.fighters)
                    a = tactical_actions(rules, sample, 1)
                    b = tactical_actions(residual, sample, 1, np.ones(env.detectors+env.fighters, dtype=int))
                    np.testing.assert_array_equal(a[0], b[0])
                    np.testing.assert_array_equal(a[1][:, 1:], b[1][:, 1:])
                    delta = (b[1][:, 0] - a[1][:, 0] + 180) % 360 - 180
                    self.assertTrue((np.abs(delta) <= 12).all())
                    if threatened:
                        np.testing.assert_array_equal(a[1], b[1])
            finally:
                env.close()

        @unittest.skipUnless(native and not navigation_only, 'native combat opt-in')
        def test_combat_update_resume_and_repeat(self):
            for mode in ('self-defense', 'adversarial'):
                cfg = Config(mode=mode, combat_control='residual', hidden=32, episode_steps=8, epochs=1)
                trainer = Trainer(cfg)
                before = [p.detach().clone() for p in trainer.actor.parameters()]
                batch, row = collect_episode(trainer.actor, trainer.critic, cfg, trainer.device, 11)
                self.assertAlmostEqual(row['episode_return'], sum(row['reward_components'].values()))
                trainer.update(batch)
                self.assertTrue(any(not torch.equal(a, b) for a, b in zip(before, trainer.actor.parameters())))
                with tempfile.TemporaryDirectory() as folder:
                    path = Path(folder) / 'combat.pt'
                    atomic_save(path, trainer.checkpoint())
                    restored = Trainer(cfg)
                    restored.restore(path)
                    self.assertEqual(restored.updates, 1)
                    a = evaluate(trainer.actor, cfg, trainer.device, [23])
                    b = evaluate(restored.actor, cfg, restored.device, [23])
                    self.assertEqual(a, b)

        def test_combat_reward_components(self):
            env = MaCACombatEnv.__new__(MaCACombatEnv)
            env.cfg, env.side = Config(mode='adversarial', episode_steps=10), 0
            def obs(alive, ammo):
                return dict(detector_obs_list=[], fighter_obs_list=[dict(alive=alive, l_missile_left=ammo)])
            class Scores:
                def get_reward(self):
                    return (0, 0, 1, 0, 0, 0)
            env.env = Scores()
            env.before = [obs(True, 2), obs(True, 2)]
            env.observations = [obs(True, 1), obs(False, 0)]
            env.ammo_used, env.last_jams, env.reward_totals = 0, 1, {}
            reward, terms = env.reward(2.0, True)
            self.assertGreater(terms['survival'], 0)
            self.assertEqual(terms['kills'], 1)
            self.assertEqual(terms['task'], 0.2)
            self.assertLess(terms['resources'], 0)
            self.assertEqual(terms['outcome'], 1)
            self.assertAlmostEqual(reward, sum(terms.values()))
            env.observations[0] = obs(False, 0)
            _, terms = env.reward(0, False)
            self.assertLess(terms['survival'], 0)
            self.assertEqual(env.ammo_used, 1)  # lost inventory is not expenditure

        def test_action_boundary(self):
            for d, f in ((0, 0), (0, 10), (2, 10), (2, 0)):
                n = d + f
                da, fa = safe_actions(np.arange(n) % 24, np.ones(n, bool), d)
                self.assertEqual(da.shape, (d, 2))
                self.assertEqual(fa.shape, (f, 4))
                assert_safe_actions(da, fa)
            with self.assertRaises(ValueError):
                assert_safe_actions(np.zeros((0, 2), np.int32), np.ones((2, 4), np.int32))

        def test_navigation_fallback_and_reset(self):
            cfg = Config(backend="toy", toy_units=3, episode_steps=100)
            env = ToyNavigationEnv(cfg, 2)
            agent = NavigationAgent(config=cfg)
            agent.set_map_info(*env.size, 0, 3)
            first = env.observation()
            initial = np.linalg.norm(agent.state.goals - env.pos, axis=1).mean()
            first_actions = agent.get_action(first, 0)
            for step in range(100):
                actions = agent.get_action(env.observation(), step)
                assert_safe_actions(*actions)
                env.step(actions)
            self.assertLess(np.linalg.norm(agent.state.goals - env.pos, axis=1).mean(), initial)
            self.assertTrue((np.linalg.norm(agent.state.goals - env.pos, axis=1) < 0.05).all())
            for expected, actual in zip(first_actions, agent.get_action(first, 0)):
                np.testing.assert_array_equal(expected, actual)
            self.assertIsNone(agent.actor)

        def test_navigation_rectangular_map_and_empty_units(self):
            agent = NavigationAgent(config=self.cfg)
            agent.set_map_info(2000, 1000, 0, 1)
            agent.set_navigation_goals([[1500, 750]])
            obs = dict(detector_obs_list=[], fighter_obs_list=[
                dict(alive=True, pos_x=500, pos_y=250, course=0)])
            self.assertEqual(agent.get_action(obs, 0)[1][0, 0], 30)
            obs["fighter_obs_list"][0]["alive"] = False
            self.assertFalse(agent.get_action(obs, 1)[1].any())
            agent.set_map_info(1000, 1000, 0, 0)
            assert_safe_actions(*agent.get_action(dict(detector_obs_list=[], fighter_obs_list=[]), 0))
            with self.assertRaises(ValueError):
                agent.get_action(obs, -1)
            with self.assertRaises(ValueError):
                agent.set_map_info(1000, 1000, 0, 1.5)

        def test_navigation_output_ignores_non_navigation_fields(self):
            obs = ToyNavigationEnv(self.cfg, 8).observation()
            agent = NavigationAgent(config=self.cfg)
            agent.set_map_info(1000, 1000, 0, 3)
            expected = agent.get_action(obs, 0)
            obs["joint_obs_dict"] = {"passive_detection_enemy_list": [{"id": 1}]}
            for unit in obs["fighter_obs_list"]:
                unit.update(l_missile_left=99, s_missile_left=99,
                            r_visible_list=[{"id": 1}], j_recv_list=[{"r_fp": 8}])
            actual = agent.get_action(obs, 0)
            assert_safe_actions(*actual)
            for a, b in zip(expected, actual):
                np.testing.assert_array_equal(a, b)

        def test_navigation_checkpoint_and_resume(self):
            trainer = Trainer(self.cfg)
            with tempfile.TemporaryDirectory(prefix="super5_nav_test_") as tmp:
                path = Path(tmp) / "test.pt"
                atomic_save(path, trainer.checkpoint())
                self.assertEqual(NavigationAgent(path).policy_source, "waypoint")
                batch, _ = collect_episode(trainer.actor, trainer.critic, self.cfg, trainer.device, 9)
                trainer.update(batch)
                atomic_save(path, trainer.checkpoint())
                agent = NavigationAgent(path)
                self.assertEqual(agent.policy_source, "checkpoint")
                agent.set_map_info(1000, 1000, 0, 3)
                obs = ToyNavigationEnv(self.cfg, 8).observation()
                features, alive = NavigationState(self.cfg, 1000, 1000, 0, 3).observe(obs, 0)
                expected = safe_actions(trainer.actor(torch.as_tensor(features)).argmax(-1).numpy(), alive, 0)
                for a, b in zip(expected, agent.get_action(obs, 0)):
                    np.testing.assert_array_equal(a, b)
                trainer.update(batch)
                resumed = Trainer(self.cfg)
                resumed.restore(path)
                resumed.update(batch)
                for key, value in trainer.actor.state_dict().items():
                    torch.testing.assert_close(value, resumed.actor.state_dict()[key], rtol=0, atol=0)
                with self.assertRaises(FileNotFoundError):
                    NavigationAgent(Path(tmp) / "missing.pt")
                data = trainer.checkpoint()
                next(iter(data["actor"].values())).fill_(float("nan"))
                atomic_save(path, data)
                with self.assertRaises(ValueError):
                    NavigationAgent(path)

        # [LEGACY COMBAT TESTS] Not part of --navigation-only verification.
        def test_self_defense_ranges_and_ammunition(self):
            unit = dict(alive=True, pos_x=100, pos_y=100,
                        s_missile_left=1, l_missile_left=1,
                        r_visible_list=[dict(id=2, pos_x=150, pos_y=100)])
            for distance, short, long, expected in (
                    (50, 1, 1, 14), (51, 1, 1, 2), (120, 0, 1, 2),
                    (121, 1, 1, 0), (50, 0, 0, 0), (51, 1, 0, 0)):
                unit.update(s_missile_left=short, l_missile_left=long)
                unit["r_visible_list"][0]["pos_x"] = 100 + distance
                self.assertEqual(SelfDefenseModule.commands([unit], 12)[0], expected)
            unit.update(alive=False, s_missile_left=1)
            self.assertEqual(SelfDefenseModule.commands([unit], 12)[0], 0)

        def test_self_defense_requires_valid_local_observation(self):
            unit = dict(alive=True, pos_x=100, pos_y=100, s_missile_left=1)
            for targets in (None, [], [dict(id=0, pos_x=110, pos_y=100)],
                            [dict(id=13, pos_x=110, pos_y=100)],
                            [dict(id=1, pos_x=float("nan"), pos_y=100)],
                            [dict(id=1, pos_x=110)],
                            [dict(id=1, alive=False, pos_x=110, pos_y=100)]):
                unit["r_visible_list"] = targets
                self.assertEqual(SelfDefenseModule.commands([unit], 12)[0], 0)
            unit["r_visible_list"] = [dict(id=2, pos_x=130, pos_y=100),
                                      dict(id=1, pos_x=110, pos_y=100)]
            self.assertEqual(SelfDefenseModule.commands([unit], 12)[0], 13)

        def test_self_defense_action_channels(self):
            detector = dict(alive=True)
            fighter = dict(alive=True, pos_x=100, pos_y=100, s_missile_left=1,
                           r_visible_list=[dict(id=3, pos_x=110, pos_y=100)],
                           j_recv_list=[dict(r_fp=4), dict(r_fp=4), dict(r_fp=8)])
            obs = dict(detector_obs_list=[detector], fighter_obs_list=[fighter, dict(alive=False)])
            d, f = safe_actions(np.array([1, 2, 3]), [True, True, False], 1, obs, 5)
            np.testing.assert_array_equal(d, [[15, 1]])
            np.testing.assert_array_equal(f, [[30, 2, 4, 8], [0, 0, 0, 0]])
            assert_safe_actions(d, f, 5)
            for signals, expected in (([], 0), ([dict(r_fp=0), dict(r_fp=99)], 0),
                                      ([dict(r_fp=2), dict(r_fp=7)], 11)):
                fighter["j_recv_list"] = signals
                self.assertEqual(SelfDefenseModule.jam_frequency(fighter), expected)
            for column, invalid in ((0, 360), (1, 11), (2, 12), (3, 11), (3, -1)):
                bad = f.copy()
                bad[0, column] = invalid
                with self.assertRaises(ValueError):
                    assert_safe_actions(d, bad, 5)
            for detectors, fighters in ((0, 0), (0, 2), (2, 0)):
                obs = dict(detector_obs_list=[dict(alive=False)] * detectors,
                           fighter_obs_list=[dict(alive=False)] * fighters)
                n = detectors + fighters
                d, f = safe_actions(np.zeros(n, dtype=np.int64), np.zeros(n, bool),
                                    detectors, obs, 5)
                self.assertEqual(d.shape, (detectors, 2))
                self.assertEqual(f.shape, (fighters, 4))
                self.assertFalse(d.any() or f.any())

        def test_ignore_other_team_and_weapons(self):
            env = ToyNavigationEnv(self.cfg, 2)
            a, b = env.observation(), env.observation()
            b["joint_obs_dict"] = {"passive_detection_enemy_list": [dict(id=999, pos_x=1)]}
            for u in b["fighter_obs_list"]:
                u.update(l_missile_left=999, s_missile_left=999,
                         r_visible_list=[dict(id=1, pos_x=20)], j_recv_list=[dict(r_fp=8)])
            sa = NavigationState(self.cfg, 1000, 1000, 0, 3)
            sb = NavigationState(self.cfg, 1000, 1000, 0, 3)
            np.testing.assert_array_equal(sa.observe(a, 0)[0], sb.observe(b, 0)[0])

        def test_episode_reset_and_invalid_observation(self):
            env = ToyNavigationEnv(self.cfg, 3)
            state = NavigationState(self.cfg, 1000, 1000, 0, 3)
            first, _ = state.observe(env.observation(), 0)
            state.observe(env.observation(), 4)
            np.testing.assert_array_equal(first, state.observe(env.observation(), 0)[0])
            obs = env.observation()
            obs["fighter_obs_list"][0]["pos_x"] = float("nan")
            with self.assertRaises(ValueError):
                state.observe(obs, 1)

        def test_terminal_returns(self):
            got = lambda_returns([1., 2.], [5., 6., 100.], [False, True], 0.9, 1.)
            np.testing.assert_allclose(got, [2.8, 2.], atol=1e-6)
            got = lambda_returns([1.], [2., 7.], [False], 0.9, 1.)
            np.testing.assert_allclose(got, [7.3], atol=1e-6)

        def test_counterfactual_exact_and_action_independent(self):
            critic = CreditCritic(self.cfg)
            obs = torch.randn(1, 3, self.cfg.features)
            alive = torch.tensor([[True, True, False]])
            probs = torch.softmax(torch.randn(1, 3, 24), -1)
            actions = torch.tensor([[3, 7, 0]])
            _, _, baseline, corr = critic.values(obs, alive, actions, probs)
            self.assertTrue(corr[0, 0, 0] and corr[0, 1, 1])
            self.assertFalse(corr[:, :, 2].any())
            for i in (0, 1):
                changed = actions.clone()
                changed[0, i] = 17
                new_baseline = critic.values(obs, alive, changed, probs)[2]
                torch.testing.assert_close(baseline[0, i], new_baseline[0, i], atol=1e-6, rtol=1e-5)
            bias, c, _ = critic.encode(obs, alive)
            explicit = torch.zeros(())
            for a in range(24):
                for b in range(24):
                    explicit += probs[0, 0, a] * probs[0, 1, b] * (bias[0] + c[0, 0, a] + c[0, 1, b])
            value = critic.values(obs, alive, actions, probs)[1][0]
            torch.testing.assert_close(value, explicit, atol=1e-6, rtol=1e-5)

        def test_all_dead_and_single_unit(self):
            critic = CreditCritic(self.cfg)
            for n in (1, 3):
                obs = torch.zeros(2, n, self.cfg.features)
                mask = torch.zeros(2, n, dtype=torch.bool)
                actions = torch.zeros(2, n, dtype=torch.long)
                probs = torch.ones(2, n, 24) / 24
                q, v, b, _ = critic.values(obs, mask, actions, probs)
                self.assertTrue(torch.isfinite(b).all())
                self.assertTrue((q == 0).all() and (v == 0).all())

        # [LEGACY MIXED TEST] Includes combat outputs; excluded by --navigation-only.
        def test_learning_save_load_and_resume(self):
            trainer = Trainer(self.cfg)
            batch, _ = collect_episode(trainer.actor, trainer.critic, self.cfg, trainer.device, 9)
            old = {k: v.clone() for k, v in trainer.actor.state_dict().items()}
            metrics = trainer.update(batch)
            self.assertTrue(all(np.isfinite(metrics[k]) for k in ("policy_loss", "value_loss", "entropy")))
            self.assertTrue(any(not torch.equal(v, old[k]) for k, v in trainer.actor.state_dict().items()))
            with tempfile.TemporaryDirectory(prefix="super5_test_") as tmp:
                path = Path(tmp) / "test.pt"
                atomic_save(path, trainer.checkpoint())
                agent = Agent(path)
                agent.set_map_info(1000, 1000, 0, 3)
                obs = ToyNavigationEnv(self.cfg, 8).observation()
                before = {k: v.clone() for k, v in agent.actor.state_dict().items()}
                assert_safe_actions(*agent.get_action(obs, 1), enemy_count=3)
                obs["fighter_obs_list"][0].update(
                    s_missile_left=1, j_recv_list=[dict(r_fp=6)],
                    r_visible_list=[dict(id=2,
                                         pos_x=obs["fighter_obs_list"][0]["pos_x"] + 10,
                                         pos_y=obs["fighter_obs_list"][0]["pos_y"])])
                _, actions = agent.get_action(obs, 2)
                self.assertEqual(actions[0, 3], 5)
                self.assertEqual(actions[0, 2], 6)
                self.assertTrue((actions[:, 1] > 0).all())
                for k, v in agent.actor.state_dict().items():
                    torch.testing.assert_close(v, before[k], rtol=0, atol=0)
                trainer.update(batch)
                expected = {k: v.clone() for k, v in trainer.actor.state_dict().items()}
                resumed = Trainer(self.cfg)
                resumed.restore(path)
                resumed.update(batch)
                for k, v in resumed.actor.state_dict().items():
                    torch.testing.assert_close(v, expected[k], rtol=0, atol=0)

        def test_evaluation_preserves_randomness(self):
            actor = Actor(self.cfg)
            before = torch.get_rng_state().clone()
            first = evaluate(actor, self.cfg, torch.device("cpu"), [3])
            self.assertTrue(torch.equal(before, torch.get_rng_state()))
            second = evaluate(actor, self.cfg, torch.device("cpu"), [3])
            self.assertEqual(first, second)

        # [LEGACY COMBAT TEST] Excluded by --navigation-only.
        @unittest.skipUnless(native, "enable with self-test --native")
        def test_native_self_defense_actions_and_ammo_consumption(self):
            for entry in (str(ROOT), str(ROOT / "environment")):
                if entry not in sys.path:
                    sys.path.insert(0, entry)
            from interface import Environment
            for path in ("maps/1000_1000_fighter10v10.map", "maps/1000_1000_2_10_vs_2_10.map"):
                env = Environment(str(resolve_map(path)), "raw", "raw", max_step=300,
                                  render=False, random_pos=False, random_seed=7, log=False)
                counts = env.get_unit_num()
                size = np.asarray(env.get_map_size())
                obs = env.get_obs()
                initial_ammo = sum(u["l_missile_left"] + u["s_missile_left"]
                                   for team in obs for u in team["fighter_obs_list"])
                fired = False
                for step in range(300):
                    streams = []
                    for side in (0, 1):
                        units = obs[side]["detector_obs_list"] + obs[side]["fighter_obs_list"]
                        # Fixed center-directed headings exercise the game interface
                        # independently of navigation policy quality.
                        indices = np.zeros(len(units), dtype=np.int64)
                        for i, unit in enumerate(units):
                            if unit["alive"]:
                                delta = size / 2 - [unit["pos_x"], unit["pos_y"]]
                                indices[i] = int(round(math.degrees(math.atan2(delta[1], delta[0])) / 15)) % 24
                        other = 1 - side
                        actions = safe_actions(indices, [u["alive"] for u in units], counts[side * 2],
                                               obs[side], sum(counts[other * 2:other * 2 + 2]))
                        fired = fired or bool(actions[1][:, 3].any())
                        streams.append(actions)
                    self.assertIsNot(env.step(*streams[0], *streams[1]), False)
                    obs = env.get_obs()
                    ammo = sum(u.get("l_missile_left", 0) + u.get("s_missile_left", 0)
                               for team in obs for u in team["fighter_obs_list"])
                    if fired and ammo < initial_ammo:
                        break
                    if env.get_done():
                        break
                self.assertTrue(fired, "No self-defense command reached the native engine")
                self.assertLess(ammo, initial_ammo, "Native engine did not consume ammunition")

        @unittest.skipUnless(native, "enable with self-test --native")
        def test_native_maps_both_sides_and_ammo(self):
            for path in ("maps/1000_1000_fighter10v10.map", "maps/1000_1000_2_10_vs_2_10.map"):
                for side in (0, 1):
                    cfg = Config(episode_steps=4, map_path=path, hidden=32)
                    env = MaCANavigationEnv(cfg, 7, side)
                    try:
                        for obs in env.observations:
                            for u in obs["fighter_obs_list"]:
                                self.assertEqual(u["l_missile_left"] + u["s_missile_left"], 0)
                        state = NavigationState(cfg, *env.size, env.detectors, env.fighters)
                        obs = env.observation()
                        positions = []
                        for step in range(4):
                            x, alive = state.observe(obs, step)
                            obs, done = env.step(safe_actions(np.zeros(state.n, dtype=np.int64), alive, env.detectors))
                            self.assertEqual(done, step == 3)
                            positions.append(obs["fighter_obs_list"][0]["pos_x"])
                        self.assertTrue(done)
                        self.assertGreater(len(set(positions)), 1, "Native navigation froze after ammo termination")
                    finally:
                        env.close()

    suite = unittest.defaultTestLoader.loadTestsFromTestCase(Tests)
    if navigation_only:
        # Existing mixed/combat tests remain available through the original CLI.
        suite = unittest.TestSuite(t for t in suite if "self_defense" not in t._testMethodName
                                   and t._testMethodName != "test_learning_save_load_and_resume")
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    return 0 if result.wasSuccessful() else 1


def positive_int(text):
    value = int(text)
    if value <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return value


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command")
    train = sub.add_parser("train", help="Train heading-only cooperative navigation")
    train.add_argument("--backend", choices=("maca", "toy"), default="maca")
    train.add_argument("--map", default="maps/1000_1000_fighter10v10.map")
    train.add_argument("--out", help="Run directory; default super5_runs/default")
    train.add_argument("--resume", help="Restore ALL training settings and optimizer; updates are additional")
    train.add_argument("--updates", type=positive_int, default=300)
    train.add_argument("--episode-steps", type=positive_int, default=512)
    train.add_argument("--eval-every", type=positive_int, default=20)
    train.add_argument("--seed", type=int, default=17)
    train.add_argument("--hidden", type=positive_int, default=64)
    train.add_argument("--toy-units", type=positive_int, default=6)
    train.add_argument("--curriculum-updates", type=int, default=200)
    evaluate_parser = sub.add_parser("evaluate", help="Evaluate navigation only, using sanitized native maps")
    evaluate_parser.add_argument("--checkpoint", required=True)
    evaluate_parser.add_argument("--map", help="Optional transfer-evaluation map")
    evaluate_parser.add_argument("--episode-steps", type=positive_int)
    evaluate_parser.add_argument("--seeds", nargs="+", type=int, default=[200003, 200009, 200017])
    evaluate_parser.add_argument("--render", action="store_true")
    navigation = sub.add_parser("navigation-evaluate", help="Unarmed waypoint baseline or explicit checkpoint")
    navigation.add_argument("--checkpoint", help="Omit to evaluate the untrained waypoint baseline")
    navigation.add_argument("--backend", choices=("maca", "toy"), default="maca",
                            help="Used without a checkpoint; otherwise uses checkpoint backend")
    navigation.add_argument("--map")
    navigation.add_argument("--episode-steps", type=positive_int)
    navigation.add_argument("--seeds", nargs="+", type=int, default=[200003, 200009, 200017])
    test = sub.add_parser("self-test", help="Run invariant, math, gradient and checkpoint tests")
    test.add_argument("--native", action="store_true", help="Also test original MaCA on both bundled maps")
    test.add_argument("--navigation-only", action="store_true", help="Exclude legacy combat and mixed tests")
    for command in (train, evaluate_parser, navigation, test):
        command.add_argument("--threads", type=positive_int, default=1)
        command.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cpu")
    combat_train = sub.add_parser("combat-train", parents=[train], add_help=False,
                                  help="Armed native combat PPO (learned headings, rule-based weapons)")
    combat_train.add_argument('--combat-control', choices=('residual', 'legacy'), default='residual',
                              help='Learn bounded tactical search corrections or legacy absolute headings')
    combat_eval = sub.add_parser("combat-evaluate", help="Paired Agent4 comparison on original armed maps")
    combat_eval.add_argument('--checkpoint')
    combat_eval.add_argument('--combat-control', choices=('agent4', 'residual', 'legacy'), default='agent4')
    combat_eval.add_argument('--map', default='maps/1000_1000_fighter10v10.map')
    combat_eval.add_argument('--episode-steps', type=positive_int, default=1000)
    combat_eval.add_argument('--seeds', nargs='+', type=int, default=[11, 23, 37])
    combat_eval.add_argument('--out', required=True)
    combat_eval.add_argument('--device', choices=('cpu', 'cuda', 'auto'), default='cpu')
    combat_eval.add_argument('--threads', type=positive_int, default=1)
    for command in (combat_train, combat_eval):
        command.add_argument('--mode', choices=('self-defense', 'adversarial'), default='adversarial')
        command.add_argument('--random-pos', action='store_true')
    for name, default in (('survival', 1.0), ('task', 0.1), ('kill', 1.0), ('resource', 0.02), ('outcome', 1.0)):
        combat_train.add_argument('--' + name + '-weight', type=float, default=default)
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return 0
    torch.set_num_threads(args.threads)
    if args.command == "self-test":
        return self_test(args.native, args.navigation_only)
    try:
        if args.command in ("train", "combat-train"):
            train_command(args)
        elif args.command == "combat-evaluate":
            combat_evaluation_command(args)
        elif args.command == "navigation-evaluate":
            navigation_evaluation_command(args)
        else:
            evaluation_command(args)
    except KeyboardInterrupt:
        print("Stopped. Resume latest.pt from the last completed episode/update.", file=sys.stderr)
        return 130
    except (ValueError, OSError, ImportError, RuntimeError, FloatingPointError, pickle.UnpicklingError) as exc:
        print("super5: {}: {}".format(type(exc).__name__, exc), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
