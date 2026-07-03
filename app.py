import os
import json
from datetime import datetime
from flask import Flask, render_template, request, jsonify
import fitparse
import numpy as np

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024
app.config['UPLOAD_FOLDER'] = '/tmp/uploads' if os.environ.get('VERCEL') == '1' else os.path.join(os.path.dirname(__file__), 'uploads')
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

def parse_fit(path):
    ff = fitparse.FitFile(path)
    session = {}
    for msg in ff.get_messages('session'):
        for field in msg:
            val = field.value
            if isinstance(val, datetime):
                val = val.isoformat()
            session[field.name] = val
    records = []
    for msg in ff.get_messages('record'):
        d = {}
        for field in msg:
            val = field.value
            if isinstance(val, datetime):
                val = val.isoformat()
            d[field.name] = val
        records.append(d)
    return session, records

def build_time_series(records):
    if not records:
        return {}
    start_str = records[0].get('timestamp')
    if not start_str:
        return {}
    start = datetime.fromisoformat(start_str) if isinstance(start_str, str) else start_str
    series = {
        'time_min': [], 'heart_rate': [], 'speed_kmh': [],
        'cadence': [], 'distance_km': [], 'altitude': [],
    }
    for r in records:
        elapsed = (datetime.fromisoformat(r['timestamp']) - start).total_seconds()
        series['time_min'].append(round(elapsed / 60, 2))
        series['heart_rate'].append(r.get('heart_rate'))
        spd = r.get('enhanced_speed', r.get('speed'))
        series['speed_kmh'].append(round(spd * 3.6, 2) if spd else None)
        series['cadence'].append(r.get('cadence'))
        dist = r.get('distance')
        series['distance_km'].append(round(dist / 1000, 3) if dist else None)
        series['altitude'].append(r.get('enhanced_altitude', r.get('altitude')))
    return series

def compute_stats(series):
    t = series.get('time_min', [])
    hr = [v for ti, v in zip(t, series.get('heart_rate', [])) if v is not None and ti <= 30]
    spd = [v for ti, v in zip(t, series.get('speed_kmh', [])) if v is not None and ti <= 30]
    cad = [v for ti, v in zip(t, series.get('cadence', [])) if v is not None and ti <= 30]
    dist = series.get('distance_km', [])
    dist_last = [v for ti, v in zip(t, dist) if v is not None and ti <= 30]
    stats = {}
    if hr:
        stats['hr_avg'] = int(round(np.mean(hr)))
        stats['hr_max'] = max(hr)
        stats['hr_min'] = min(hr)
        stats['hr_std'] = round(float(np.std(hr)), 1)
    if spd:
        stats['speed_avg_kmh'] = round(np.mean(spd), 1)
        stats['speed_max_kmh'] = round(max(spd), 1)
    if cad:
        stats['cadence_avg'] = int(round(np.mean(cad)))
        stats['cadence_max'] = max(cad)
    if dist_last:
        stats['distance_km'] = round(dist_last[-1], 2)
    return stats

def compute_steady_state(series):
    hr = series.get('heart_rate', [])
    t = series.get('time_min', [])
    blocks = {1: [], 2: [], 3: [], 4: []}
    for ti, h in zip(t, hr):
        if ti < 10 or ti > 30 or h is None:
            continue
        block = min(int((ti - 10) // 5) + 1, 4)
        blocks[block].append(h)
    ss = {}
    all_hr = [h for h in hr if h is not None]
    hr_10_30 = [h for ti, h in zip(t, hr) if 10 <= ti <= 30 and h is not None]
    if hr_10_30:
        ss['steady_hr_avg'] = int(round(np.mean(hr_10_30)))
        ss['steady_hr_min'] = min(hr_10_30)
        ss['steady_hr_max'] = max(hr_10_30)
        ss['steady_hr_std'] = round(float(np.std(hr_10_30)), 1)
    block_avgs = []
    for b in [1, 2, 3, 4]:
        if blocks[b]:
            block_avgs.append(int(round(np.mean(blocks[b]))))
        else:
            block_avgs.append(None)
    ss['block_avgs'] = block_avgs
    if block_avgs[0] is not None and block_avgs[3] is not None:
        ss['hr_drift'] = block_avgs[3] - block_avgs[0]
    return ss

def compute_hr_zones(series):
    hr = [v for t, v in zip(series.get('time_min', []), series.get('heart_rate', []))
          if v is not None and t <= 30]
    if not hr:
        return {}
    total = len(hr)
    zones = [
        (0, 110, 'zone1'), (110, 130, 'zone2'), (130, 150, 'zone3'),
        (150, 165, 'zone4'), (165, 250, 'zone5'),
    ]
    result = {}
    for lo, hi, name in zones:
        count = sum(1 for h in hr if lo <= h < hi)
        result[name] = round(count / total * 100, 1)
    return result

def compute_hr_distribution(series, bins=20):
    hr = [v for t, v in zip(series.get('time_min', []), series.get('heart_rate', []))
          if v is not None and t <= 30]
    if not hr:
        return {'bins_center': [], 'counts': []}
    mn, mx = min(hr), max(hr)
    pad = (mx - mn) * 0.1 or 5
    edges = np.linspace(mn - pad, mx + pad, bins + 1)
    counts, _ = np.histogram(hr, bins=edges)
    centers = [(edges[i] + edges[i+1]) / 2 for i in range(len(counts))]
    return {
        'bins_center': [round(c, 1) for c in centers],
        'counts': [round(float(c), 4) for c in (counts / np.sum(counts))],
    }

def process_fit_file(filepath, filename):
    session_data, records = parse_fit(filepath)
    series = build_time_series(records)
    stats = compute_stats(series)
    steady_state = compute_steady_state(series)
    hr_zones = compute_hr_zones(series)
    hr_dist = compute_hr_distribution(series)
    date_str = ''
    label = ''
    if 'start_time' in session_data:
        st = session_data['start_time']
        if isinstance(st, str):
            label = st[:10]
        else:
            label = str(st)[:10]
    elif 'Run' in filename:
        parts = filename.split('Run')
        if len(parts) > 1:
            ds = parts[1].replace('.fit', '')
            label = f"{ds[:4]}-{ds[4:6]}-{ds[6:8]}" if len(ds) >= 8 else filename
    else:
        label = filename
    return {
        'filename': filename,
        'label': label,
        'start_time': session_data.get('start_time'),
        'total_distance_m': session_data.get('total_distance'),
        'total_elapsed_time_s': session_data.get('total_elapsed_time'),
        'total_calories': session_data.get('total_calories'),
        'avg_temperature': session_data.get('avg_temperature'),
        'session_avg_hr': session_data.get('avg_heart_rate'),
        'session_max_hr': session_data.get('max_heart_rate'),
        'session_avg_cadence': session_data.get('avg_running_cadence'),
        'session_avg_speed_ms': session_data.get('avg_speed'),
        'series': series,
        'stats': stats,
        'steady_state': steady_state,
        'hr_zones': hr_zones,
        'hr_distribution': hr_dist,
    }

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/upload', methods=['POST'])
def upload_files():
    if 'files' not in request.files:
        return jsonify({'error': 'No files provided'}), 400
    files = request.files.getlist('files')
    if not files or all(f.filename == '' for f in files):
        return jsonify({'error': 'No files selected'}), 400
    results = []
    errors = []
    for f in files:
        if not f.filename.lower().endswith('.fit'):
            errors.append(f"{f.filename}: not a .fit file")
            continue
        save_path = os.path.join(app.config['UPLOAD_FOLDER'], f.filename)
        f.save(save_path)
        try:
            parsed = process_fit_file(save_path, f.filename)
            results.append(parsed)
        except Exception as e:
            errors.append(f"{f.filename}: {str(e)}")
        finally:
            try:
                os.remove(save_path)
            except OSError:
                pass
    return jsonify({'ok': True, 'sessions': results, 'errors': errors})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_DEBUG', '0') == '1'
    app.run(host='0.0.0.0', port=port, debug=debug)
