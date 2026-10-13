import pathlib,urllib.request,tarfile,io,csv,json,collections,concurrent.futures,hashlib,time
ROOT=pathlib.Path(__file__).resolve().parents[1];RAW=ROOT/'shot-data/raw';RAW.mkdir(parents=True,exist_ok=True)
SHA='e829d4678be1e075f99e5d41a1c5f97089be446b'
def download(start):
 p=RAW/f'shotdetail_{start}.tar.xz'
 if not p.exists():
  for attempt in range(3):
   try:
    with urllib.request.urlopen(f'https://raw.githubusercontent.com/shufinskiy/nba_data/{SHA}/datasets/shotdetail_{start}.tar.xz',timeout=60) as r:p.write_bytes(r.read())
    break
   except Exception:
    if attempt==2:raise
 with tarfile.open(p) as tar:
  member=next(m for m in tar.getmembers() if m.isfile() and m.name.endswith('.csv'))
  with io.TextIOWrapper(tar.extractfile(member)) as source:
   counts=collections.defaultdict(lambda:[0,0,0,0]);dates=[]
   for r in csv.DictReader(source):
    gid=r['GAME_ID'].zfill(10)
    if not gid.startswith('002') or r['SHOT_ATTEMPTED_FLAG']!='1':continue
    k=(int(r['PLAYER_ID']),r['PLAYER_NAME']);counts[k][0]+=1;counts[k][1]+=int(r['SHOT_MADE_FLAG']);counts[k][2]+=r['SHOT_TYPE'].startswith('3');counts[k][3]+=int(r['SHOT_MADE_FLAG'])*r['SHOT_TYPE'].startswith('3');dates.append(r['GAME_DATE'])
 result={'year':start+1,'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'source_commit':SHA,'fetchedAt':p.stat().st_mtime,'firstDate':min(dates),'lastDate':max(dates),'shots':len(dates),'players':[{'nbaId':pid,'name':name,'fga':v[0],'fgm':v[1],'fg3a':v[2],'fg3m':v[3]} for (pid,name),v in counts.items()]}
 (ROOT/f'shot-data/counts-{start+1}.json').write_text(json.dumps(result));print('Fetched',start+1,result['shots'],len(counts),result['lastDate'],flush=True)
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:list(ex.map(download,range(2016,2026)))
