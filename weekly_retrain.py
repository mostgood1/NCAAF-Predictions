"""Weekly retrain + prediction regeneration orchestrator.

Steps:
1. (Soon) Ingest latest completed games (assumes upstream ingestion already ran).
2. Run retune_models.py with a new version prefix (timestamp or increment) if enough data.
3. If models produced, regenerate enhanced predictions overlaying model outputs.
4. Write a manifest JSON describing artifacts for the app/deployment.

Usage examples:
  python weekly_retrain.py              # auto timestamped prefix
  python weekly_retrain.py --prefix rf_wk03
  python weekly_retrain.py --no-replace-predicted

The app currently prefers model_* values for display; replacing predicted_* makes
historical diffing harder, so default is NOT to replace (toggle with flag).
"""
from __future__ import annotations
import os, json, subprocess, sys, time
from pathlib import Path
from datetime import datetime, timezone
import re
import pandas as pd

from src.data.weather_enrichment import enrich_dataframe

BASE_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
RETUNE = BASE_DIR / 'src' / 'modeling' / 'retune_models.py'
GEN = BASE_DIR / 'generate_enhanced_predictions.py'
MODELS_DIR = BASE_DIR / 'models'
DATA_DIR = BASE_DIR / 'data'
MANIFEST = MODELS_DIR / 'model_manifest.json'

def _run(cmd: list[str]):
    proc = subprocess.run(cmd, capture_output=True, text=True)
    return proc.returncode, proc.stdout.strip(), proc.stderr.strip()

def _suggest_prefix():
    # Increment last rf_v* if present else timestamp
    existing = [p.name for p in MODELS_DIR.glob('rf_v*_metrics.json')]
    versions = []
    for name in existing:
        m = re.match(r'rf_v(\d+)_metrics.json', name)
        if m:
            versions.append(int(m.group(1)))
    if versions:
        nxt = max(versions)+1
        return f'rf_v{nxt}'
    return 'rf_' + datetime.utcnow().strftime('%Y%m%d')

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--prefix', help='Explicit model version prefix (e.g., rf_v3).')
    ap.add_argument('--min-final-games', type=int, default=80)
    ap.add_argument('--replace-predicted', action='store_true', help='Overwrite predicted_* columns with model_* values.')
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()

    prefix = args.prefix or _suggest_prefix()
    ts = datetime.utcnow().replace(tzinfo=timezone.utc).isoformat()
    results = {'started_at_utc': ts, 'model_prefix': prefix}

    if not RETUNE.exists():
        print(json.dumps({'status':'error','reason':'retune_script_missing','path': str(RETUNE)}))
        return
    if not GEN.exists():
        print(json.dumps({'status':'error','reason':'generate_script_missing','path': str(GEN)}))
        return

    if args.dry_run:
        print(json.dumps({'status':'dry_run','model_prefix':prefix}))
        return

    # 1. Retrain
    code, out, err = _run([sys.executable, str(RETUNE), '--out-prefix', prefix, '--min-final-games', str(args.min_final_games)])
    results['retune_rc'] = code
    results['retune_stdout'] = out
    if err:
        results['retune_stderr'] = err
    try:
        retune_json = json.loads(out)
        results['retune_status'] = retune_json.get('status')
    except Exception:
        results['retune_status'] = 'unparsed'

    # If retrain skipped, reuse previous prefix (if it existed) and still regenerate predictions with whatever models exist
    # 2a. Pre-enrich base enhanced file with weather before regeneration to maximize feature completeness
    try:
        enh_file = DATA_DIR / 'college_football_schedule_2025_predicted_totals_enhanced.csv'
        if enh_file.exists():
            _df_base = pd.read_csv(enh_file)
            pre_rows = len(_df_base)
            _df_base = enrich_dataframe(_df_base)
            _df_base.to_csv(enh_file, index=False)
            results['pre_enrichment_rows'] = pre_rows
            results['post_enrichment_rows'] = len(_df_base)
            miss_weather = int((_df_base['weather_temp'].isna() | _df_base['weather_wind'].isna()).sum()) if {'weather_temp','weather_wind'}.issubset(_df_base.columns) else None
            results['post_enrichment_missing_weather'] = miss_weather
    except Exception as e:
        results['pre_enrichment_error'] = str(e)

    # 2b. Regenerate predictions
    gen_cmd = [sys.executable, str(GEN), '--model-prefix', prefix]
    if args.replace_predicted:
        gen_cmd.append('--replace-predicted')
    code2, out2, err2 = _run(gen_cmd)
    results['generate_rc'] = code2
    results['generate_stdout'] = out2
    if err2:
        results['generate_stderr'] = err2
    try:
        gen_json = json.loads(out2)
        results['predictions_file'] = gen_json.get('output_file')
    except Exception:
        pass

    # 3. Persist manifest
    manifest = {
        'model_prefix': prefix,
        'generated_at_utc': datetime.utcnow().replace(tzinfo=timezone.utc).isoformat(),
        'replace_predicted': args.replace_predicted,
        'retune': {
            'rc': code,
            'status': results.get('retune_status'),
        },
        'regeneration': {
            'rc': code2,
            'output_file': results.get('predictions_file')
        }
    }
    try:
        MANIFEST.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
        results['manifest'] = MANIFEST.name
    except Exception as e:
        results['manifest_error'] = str(e)

    print(json.dumps({'status':'ok', **results}))

if __name__ == '__main__':
    main()
