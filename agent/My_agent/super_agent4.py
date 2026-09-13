# -*- coding: utf-8 -*-
"""MaCA rule agent tuned against Larry's agent2; no training weights required.

Uses only the supplied raw observation. Parameters are empirical heuristics,
not assumptions about hidden hit probabilities or enemy ammunition.
"""
import math
from collections import Counter
import numpy as np


class Agent:
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
            if unit['l_missile_left'] + unit['s_missile_left'] <= 0:
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
        offset = self.detector_num + self.fighter_num
        for idx, unit in enumerate(units):
            if not unit['alive'] or step - self.last_shot.get(idx, -1000) < self.fire_interval:
                continue
            for target in unit.get('r_visible_list', []):
                eid = int(target['id'])
                if not 1 <= eid <= offset:
                    continue
                dist = self._distance(unit, target)
                short, long = unit['s_missile_left'], unit['l_missile_left']
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
