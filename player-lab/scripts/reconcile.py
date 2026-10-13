exec(open('scripts/ingest.py').read().split('with concurrent.futures.ThreadPoolExecutor')[0])
with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
 for year in range(2017,2027):
  path=D/f'season-{year}.json';raw=json.loads(path.read_text())
  if raw.get('verifiedCounts'):print('Already verified',year,flush=True);continue
  keys={e:{(r['playerId'],r['team']):r for r in raw[e]} for e in ['playertotals','playeradvancedstats']}
  union=set(keys['playertotals'])|set(keys['playeradvancedstats'])
  expected=len(raw['playertotals'])
  if len(union)<expected:
   for e in keys:
    base=f'https://api.server.nbaapi.com/api/{e}?season={year}&isPlayoff=false&pageSize=50&sortBy=playerId&ascending=true'
    first=get(base);pages=[first]+list(pool.map(get,[base+f'&page={p}' for p in range(2,first['pagination']['pages']+1)]))
    keys[e]={(r['playerId'],r['team']):r for x in pages for r in x['data']}
   union=set(keys['playertotals'])|set(keys['playeradvancedstats'])
  jobs=[]
  for e,rows in keys.items():
   missing=union-set(rows)
   jobs.extend((e,pid) for pid in sorted({p for p,t in missing}))
  def repair(job):
   e,pid=job;x=get(f'https://api.server.nbaapi.com/api/{e}?season={year}&isPlayoff=false&pageSize=50&playerId={pid}');return e,x['data']
  for e,rows in pool.map(repair,jobs):
   for r in rows:keys[e][(r['playerId'],r['team'])]=r
  for e,rows in keys.items():
   meta=get(f'https://api.server.nbaapi.com/api/{e}?season={year}&isPlayoff=false&pageSize=1')['pagination']['total']
   if len(rows)!=meta:raise ValueError(f'{year} {e} {len(rows)} != {meta}')
   raw[e]=list(rows.values())
  raw['verifiedCounts']={e:len(rows) for e,rows in keys.items()};raw['fetchedAt']=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime());path.write_text(json.dumps(raw,separators=(',',':')));print('Verified',year,raw['verifiedCounts'],flush=True)
