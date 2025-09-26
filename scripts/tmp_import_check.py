import importlib.util as iu
from pathlib import Path
p = Path(r"c:/Users/mostg/OneDrive/Coding/NCAAFCompare/app.py")
spec = iu.spec_from_file_location('app', str(p))
mod = iu.module_from_spec(spec)
spec.loader.exec_module(mod)
app = getattr(mod, 'app', None)
if app is None:
    raise SystemExit('No Flask app found')
print('import ok; routes=', len(list(app.url_map.iter_rules())))
