"""01-04: Read timestamped CSE-CIC-IDS2018 TrafficForML CSVs.

A stride keeps every N-th record in each raw CSV in original capture order.
This is deterministic temporal thinning, NOT a random shuffle.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd

SKIP_COLS = {'Flow ID', 'Source IP', 'Destination IP', 'Timestamp', 'Label',
             'Unnamed: 0', 'Source', 'Destination'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw', default='data/raw', help='Folder containing timestamped CICIDS CSVs')
    parser.add_argument('--stride', type=int, default=10,
                        help='Use every Nth row per file (1=full, 10=starter)')
    args = parser.parse_args()
    if args.stride < 1:
        parser.error('--stride must be >= 1')
    files = sorted(Path(args.raw).rglob('*.csv'))
    if not files:
        parser.error('No CSVs found. Download and unzip CSE-CIC-IDS2018 TrafficForML into data/raw/')
    pieces = []
    sources = []
    print(f'Found {len(files)} CSV file(s).')
    for file in files:
        parts = []
        rows_seen = 0
        for chunk in pd.read_csv(file, chunksize=20000, low_memory=False, encoding='latin1'):
            chunk.columns = chunk.columns.str.strip()
            if not {'Timestamp', 'Label'}.issubset(chunk.columns):
                raise ValueError(f'{file.name}: missing Timestamp/Label. Use GeneratedLabelledFlows, NOT MachineLearningCSV.')
            # Absolute row offsets make sampling deterministic across chunk sizes.
            keep = (np.arange(len(chunk)) + rows_seen) % args.stride == 0
            rows_seen += len(chunk)
            parts.append(chunk.iloc[keep].copy())
        if parts:
            piece = pd.concat(parts, ignore_index=True)
            piece['__source_file'] = file.name
            pieces.append(piece)
            sources.append({'file': str(file), 'bytes': file.stat().st_size,
                            'raw_rows': rows_seen, 'kept_rows': len(piece)})
            print(f'  {file.name}: {rows_seen:,} -> {len(piece):,}')
    data = pd.concat(pieces, ignore_index=True)
    # Source files are day/month/year in the original GeneratedLabelledFlows release.
    stamp = pd.to_datetime(data['Timestamp'].astype(str).str.strip(),
                           format='mixed', dayfirst=True, errors='coerce')
    valid = stamp.notna() & data['Label'].notna()
    data = data.loc[valid].copy()
    data['__timestamp'] = stamp.loc[valid]
    data['Label'] = data['Label'].astype(str).str.strip()
    data = data.loc[data['Label'] != ''].copy()
    data = data.sort_values('__timestamp', kind='stable').reset_index(drop=True)
    if data.empty:
        raise ValueError('All rows had missing timestamps or labels. Check timestamp format.')
    print('Time range:', data['__timestamp'].iloc[0], 'to', data['__timestamp'].iloc[-1])
    print('Original class counts:', data['Label'].value_counts().head(16).to_dict())
    # Drop duplicated columns such as Fwd Header Length.1.
    drop_cols = SKIP_COLS | {'__timestamp', '__source_file'}
    candidates = []
    for c in data.columns:
        if c in drop_cols or c.startswith('Unnamed:'):
            continue
        if c.endswith('.1') and c[:-2] in data.columns:
            continue
        candidates.append(c)
    # Exclude nonnumeric or mostly nonnumeric input; missing numeric values become NaN.
    cols = []
    features = {}
    # Determine feature eligibility using ONLY an early portion of the stream.
    # Avoid selecting columns based on future evaluation records.
    early_rows = max(1, int(len(data) * 0.20))
    for col in candidates:
        vec = pd.to_numeric(data[col], errors='coerce')
        vec = vec.replace([np.inf, -np.inf], np.nan)
        if vec.iloc[:early_rows].notna().mean() >= 0.5:
            cols.append(col)
            features[col] = vec.astype('float32')
    if not cols:
        raise ValueError('No numeric feature columns found.')
    xdf = pd.DataFrame(features)
    y = (data['Label'].str.upper() != 'BENIGN').astype('int8')
    # Exact feature + time + label duplicates only; do not remove similar flows.
    check = xdf.copy()
    check['__timestamp'] = data['__timestamp'].to_numpy()
    check['__label'] = y.to_numpy()
    unique_mask = ~check.duplicated(keep='first')
    x = xdf.loc[unique_mask].to_numpy(dtype=np.float32)
    yy = y.loc[unique_mask].to_numpy(dtype=np.int8)
    times = data.loc[unique_mask, '__timestamp'].to_numpy(dtype='datetime64[ns]')
    out = Path('data/processed')
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / 'raw_data.npz', X=x, y=yy,
                        times=times.astype('datetime64[ns]').astype('int64'),
                        columns=np.array(cols, dtype=str))
    info = {'source': 'CSE-CIC-IDS2018 TrafficForML', 'files': sources,
            'stride': args.stride, 'rows': len(yy), 'features': len(cols),
            'benign': int((yy == 0).sum()), 'attack': int((yy == 1).sum()),
            'earliest': str(times.min()), 'latest': str(times.max()),
            'timestamp_note': 'dayfirst=True; inspect date range (expect 2018-02-14 through 2018-02-15).'}
    (out / 'data_report.json').write_text(json.dumps(info, indent=2), encoding='utf8')
    print(f'Saved {len(yy):,} flows, {len(cols)} features -> {out / "raw_data.npz"}')
    if len(np.unique(yy)) < 2:
        print('WARNING: Only one class found. Include multiple days / attack files.')


if __name__ == '__main__':
    main()
