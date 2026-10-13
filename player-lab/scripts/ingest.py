import urllib.request, json, time, concurrent.futures, pathlib, unicodedata
ROOT=pathlib.Path(__file__).resolve().parents[1]
D=ROOT/'data';D.mkdir(exist_ok=True)
def get(url):
 for attempt in range(3):
  try:
   with urllib.request.urlopen(url,timeout=25) as response:return json.load(response)
  except Exception:
   if attempt==2:raise
   time.sleep(attempt+1)
def season(year):
 target=D/f'season-{year}.json'
 if target.exists():return
 allrows={}
 for endpoint in ['playertotals','playeradvancedstats']:
  rows=[];page=1
  while True:
   x=get(f'https://api.server.nbaapi.com/api/{endpoint}?season={year}&isPlayoff=false&pageSize=100&page={page}')
   rows+=x['data']
   if page>=x['pagination']['pages']:break
   page+=1
  allrows[endpoint]=rows
 allrows['fetchedAt']=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())
 target.write_text(json.dumps(allrows,separators=(',',':')))
 print(year,len(allrows['playertotals']),len(allrows['playeradvancedstats']),flush=True)
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:list(ex.map(season,range(2017,2027)))
