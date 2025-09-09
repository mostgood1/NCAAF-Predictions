import sys, os
ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path: sys.path.insert(0, ROOT)
import app
c=app.app.test_client()
r=c.get('/')
print('status', r.status_code, 'bytes', len(r.data))
print(r.data[:200].decode('utf-8','ignore'))
