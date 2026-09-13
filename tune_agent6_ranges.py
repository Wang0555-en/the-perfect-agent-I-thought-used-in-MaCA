"""Development sweep only; fixed seeds are disjoint from final holdout."""
from pathlib import Path
import sys
import json
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from agent.My_agent.evaluate_agent6 import run

if __name__ == '__main__':
    results = []
    for distance in (60, 90, 150, 180, 250, 400):
        p = dict(jam_advance=0, fire_interval=1, salvo_cap=2, shot_wait=3,
                 reserve_release_step=0, long_range=distance)
        r = run(p, [11,23])
        results.append(r)
        Path('tmp/agent6_ranges.json').write_text(json.dumps(results, indent=2), encoding='utf-8')
        print(json.dumps({k:v for k,v in r.items() if k!='rows'}), flush=True)
