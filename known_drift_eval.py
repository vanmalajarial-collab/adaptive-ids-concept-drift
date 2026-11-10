"""Known-drift delay and alarm counts for controlled/synthetic streams only.
Do NOT label ordinary CICIDS2017 day boundaries as ground-truth drifts.
"""
import argparse
import json
from pathlib import Path
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('--points', type=int, nargs='+', required=True,
                    help='Known global row indices at which a controlled drift begins')
args = parser.parse_args()
df = pd.read_csv('results/hybrid_batches.csv')
alarms = df.loc[df['adaptation_attempted'].astype(bool), 'stream_stop'].astype(int).tolist()
points = sorted(args.points)
false_alarm_positions = [a for a in alarms if a < points[0]]
out = {'known_change_points': points, 'hybrid_alarm_row_positions': alarms,
       'alarms_before_first_known_change': false_alarm_positions,
       'per_change_detection_delay_rows': [], 'undetected_known_changes': 0,
       'note': 'Valid only with externally known true concept-drift change points.'}
for i, p in enumerate(points):
    next_p = points[i+1] if i+1 < len(points) else float('inf')
    found = next((a for a in alarms if p <= a < next_p), None)
    out['per_change_detection_delay_rows'].append(None if found is None else found-p)
    if found is None:
        out['undetected_known_changes'] += 1
Path('results/known_drift_metrics.json').write_text(json.dumps(out, indent=2))
print(json.dumps(out, indent=2))
