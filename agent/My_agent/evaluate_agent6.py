# -*- coding: utf-8 -*-
"""Deterministic paired native matches and development-only Agent6 sweep."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from agent.My_agent.super_agent6 import Agent
from agent.My_agent.super_agent_5_pro import Config, MaCACombatEnv, preserve_random_state
from agent.My_agent.super_agent_5_pro import np, torch


def run(parameters, seeds, layouts=(False, True), steps=1000, baseline=False,
        map_path='maps/1000_1000_fighter10v10.map'):
    from agent.My_agent.super_agent4 import Agent as Agent4
    rows = []
    with preserve_random_state():
        for random_pos in layouts:
            for seed in seeds:
                for side in (0, 1):
                    cfg = Config(mode='adversarial', episode_steps=steps, random_pos=random_pos, map_path=map_path)
                    env = MaCACombatEnv(cfg, seed, side)
                    try:
                        agent = Agent4() if baseline else Agent(**parameters)
                        agent.set_map_info(*env.size, env.detectors, env.fighters)
                        if not baseline:
                            agent.enemy_unit_count = env.enemy_count
                        for step in range(1, steps+1):
                            _, done = env.step(agent.get_action(env.observation(), step))
                            if done:
                                break
                        row = env.summary()
                        row.update(seed=seed, side=side, random_pos=random_pos, steps=step)
                        rows.append(row)
                    finally:
                        env.close()
    wins = sum(r['result']=='win' for r in rows)
    losses = sum(r['result']=='loss' for r in rows)
    return dict(parameters=parameters, wins=wins, losses=losses, draws=len(rows)-wins-losses,
                games=len(rows), win_rate=wins/len(rows),
                survival_margin=sum(r['survivors']-r['enemy_survivors'] for r in rows)/len(rows), rows=rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seeds', type=int, nargs='+', default=[11,23,37])
    parser.add_argument('--layout', choices=('fixed','random','both'), default='both')
    parser.add_argument('--steps', type=int, default=1000)
    parser.add_argument('--parameters', default='{}')
    parser.add_argument('--map', default='maps/1000_1000_fighter10v10.map')
    parser.add_argument('--sweep', action='store_true')
    parser.add_argument('--baseline', action='store_true')
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    if args.steps < 1 or any(not 0 <= s < 2**31 for s in args.seeds):
        parser.error('Positive steps and seeds in [0, 2**31) required')
    layouts = (False,True) if args.layout=='both' else (args.layout=='random',)
    candidates = [json.loads(args.parameters)]
    if args.sweep:
        candidates = [{}]
        candidates += [dict(fire_interval=1, salvo_cap=cap, shot_wait=wait, reserve_release_step=0)
                       for cap in (1,2,3,4) for wait in (1,3,5)]
        candidates += [dict(formation_width=width) for width in (0,40,200,400)]
        candidates += [dict(jam_advance=advance) for advance in (7,4)]
        candidates += [dict(standoff=distance) for distance in (60,90,110)]
        # Keep the original development sweep reproducible after promoting +7.
        candidates = [dict(jam_advance=0, **c) if 'jam_advance' not in c else c for c in candidates]
    results = []
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    for candidate in candidates:
        result = run(candidate, args.seeds, layouts, args.steps, args.baseline, args.map)
        results.append(result)
        print(json.dumps({k:v for k,v in result.items() if k!='rows'}), flush=True)
        report = dict(seeds=args.seeds, layouts=list(layouts), steps=args.steps, baseline=args.baseline,
                      runtime=dict(python=sys.version, numpy=np.__version__, torch=torch.__version__),
                      protocol='same-seeds-both-sides-vs-unchanged-agent4',
                      map=args.map, map_sha256=hashlib.sha256((ROOT/args.map).read_bytes()).hexdigest(),
                      source_sha256={p:hashlib.sha256((ROOT/'agent/My_agent'/p).read_bytes()).hexdigest()
                                     for p in ('super_agent4.py','super_agent_5_pro.py','super_agent6.py','evaluate_agent6.py')},
                      results=results)
        out.write_text(json.dumps(report, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
