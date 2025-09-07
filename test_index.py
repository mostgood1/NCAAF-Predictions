from app import app
with app.test_client() as c:
    r=c.get('/?week=2&full=1')
    print('index status', r.status_code, 'len', len(r.data))
    txt=r.data.decode(errors='ignore')
    markers=['Louisville','James Madison','FINAL','28','14']
    for m in markers:
        print(m, m in txt)
