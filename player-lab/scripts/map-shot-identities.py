import pathlib,json,collections,unicodedata,re
ROOT=pathlib.Path(__file__).resolve().parents[1]
a=json.loads((ROOT/'data/archive.json').read_text())
def norm(s):return re.sub(r'[^a-z0-9]','',unicodedata.normalize('NFD',s.translate(str.maketrans({'ı':'i','ё':'e','е':'e'}))).encode('ascii','ignore').decode().lower())
def short(s):return re.sub(r'(jr|sr|iii|ii|iv)$','',norm(s))
names=collections.defaultdict(set);ids=collections.defaultdict(set)
for p in (ROOT/'shot-data').glob('counts-*.json'):
 for x in json.loads(p.read_text())['players']:
  names[norm(x['name'])].add(x['nbaId']);ids[x['nbaId']].add(x['name'])
zero_ids={'Ade Murkey':1630278,'Ben Moore':1628500,'Jaime Echenique':1630693,'Javin DeLaurier':1629602,'Jesse Edwards':1642399,'Malcolm Cazalon':1630608,'Michael Foster Jr.':1630701,'Scottie Lewis':1630575,'Trey McKinney-Jones':203590,'Tyler Hall':1629788}
mapping={};unmapped=[]
for p in a['players']:
 aliases={'Adama-Alpha Bal':'Adama Bal','Ron Holland':'Ronald Holland II','Tre Scott':'Trevon Scott','Matthew Hurt':'Matt Hurt','Mitch Creek':'Mitchell Creek','Vince Edwards':'Vincent Edwards','Vince Hunter':'Vincent Hunter'}
 candidates=names.get(norm(aliases.get(p['name'],p['name'])),set())
 if not candidates:
  candidates={pid for name,pids in names.items() if short(name)==short(p['name']) for pid in pids}
 if p['name'] in zero_ids:
  assert all(s['totals']['fga']==0 for s in p['seasons'] if s['total']);mapping[p['id']]=zero_ids[p['name']]
 elif len(candidates)==1:mapping[p['id']]=next(iter(candidates))
 else:unmapped.append(p['name'])
reverse=collections.defaultdict(list)
for k,v in mapping.items():reverse[v].append(k)
print('Mapped',len(mapping),'Unmapped',len(unmapped),unmapped,flush=True)
print('Ambiguous NBA IDs',[(k,v) for k,v in reverse.items() if len(v)>1])
errors=[];complete=0;missing=0;noattempts=0
for year in range(2017,2027):
 source=json.loads((ROOT/f'shot-data/counts-{year}.json').read_text());counts={}
 for row in source['players']:
  if row['nbaId'] not in counts:counts[row['nbaId']]={k:0 for k in ['fga','fgm','fg3a','fg3m']}
  for k in counts[row['nbaId']]:counts[row['nbaId']][k]+=row[k]
 for p in a['players']:
  r=next((s for s in p['seasons'] if s['year']==year and s['total']),None)
  if not r:continue
  c=counts.get(mapping.get(p['id']))
  if not r['totals']['fga']:noattempts+=1;continue
  if not c:missing+=1;errors.append((p['id'],p['name'],year,'missing',r['totals']['fga']));continue
  diff={k:[c[k],r['totals'][k]] for k in c if c[k]!=r['totals'][k]}
  if diff:errors.append((p['id'],p['name'],year,diff))
  else:complete+=1
print('EXACT',complete,'NO FGA',noattempts,'MISSING',missing,'ERRORS',len(errors));print(errors[:150])
(ROOT/'shot-data/identity-map.json').write_text(json.dumps(mapping));(ROOT/'shot-data/reconciliation.json').write_text(json.dumps({'complete':complete,'noAttemptSeasons':noattempts,'errors':errors,'unmapped':unmapped},indent=2))
