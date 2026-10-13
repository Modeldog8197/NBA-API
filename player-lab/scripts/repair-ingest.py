exec(open('scripts/ingest.py').read().split('with concurrent.futures.ThreadPoolExecutor')[0])
with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
 for year in range(2017,2027):
  path=D/f'season-{year}.json';raw=json.loads(path.read_text())
  for endpoint in ['playertotals','playeradvancedstats']:
   base=f'https://api.server.nbaapi.com/api/{endpoint}?season={year}&isPlayoff=false&pageSize=50&sortBy=playerId&ascending=true'
   first=get(base+'&page=1');count=first['pagination']['pages'];total=first['pagination']['total']
   pages=[first]+list(pool.map(get,[base+f'&page={p}' for p in range(2,count+1)]))
   unique={(r['playerId'],r['team']):r for page in pages for r in page['data']}
   boundary=set()
   for a,b in zip(pages,pages[1:]):
    if a['data'] and b['data'] and a['data'][-1]['playerId']==b['data'][0]['playerId']:boundary.add(a['data'][-1]['playerId'])
   for x in pool.map(get,[base+'&playerId='+pid for pid in boundary]):
    for r in x['data']:unique[(r['playerId'],r['team'])]=r
   if len(unique)!=total:raise ValueError(f'{year} {endpoint}: {len(unique)} != {total}')
   raw[endpoint]=list(unique.values())
  raw['fetchedAt']=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime());raw['verifiedCounts']={k:len(raw[k]) for k in ['playertotals','playeradvancedstats']}
  path.write_text(json.dumps(raw,separators=(',',':')));print('Verified',year,raw['verifiedCounts'],flush=True)
