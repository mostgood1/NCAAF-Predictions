import os, sys
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
	sys.path.insert(0, BASE_DIR)
import app  # noqa: E402
import pandas as pd  # noqa: E402

app._overlay_lines_2025_if_present()
path = os.path.join(BASE_DIR,'data','college_football_betting_lines_2025.csv')
df = pd.read_csv(path) if os.path.exists(path) else pd.DataFrame()
print('week3_line_rows', len(df[(df.get('year')==2025) & (df.get('week')==3)]))
