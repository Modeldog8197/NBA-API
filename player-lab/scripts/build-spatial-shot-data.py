import pathlib,json,tarfile,io,csv,collections,math,gzip,hashlib,time
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
ROOT=pathlib.Path(__file__).resolve().parents[1];OUT=ROOT/'shot-data/processed';OUT.mkdir(exist_ok=True)
a=json.loads((ROOT/'data/archive.json').read_text());mapping=json.loads((ROOT/'shot-data/identity-map.json').read_text());byid={mapping[p['id']]:p for p in a['players']};version='spatial-eb-v2'
ZONE_NAMES=['Restricted Area','Paint (Non-RA)','Mid-Range','Left Corner 3','Right Corner 3','Above the Break 3','Outside half court','Unlocated']
JOIN=math.sqrt(237.5**2-220**2)
def zone(x,y):
 if not(-250<=x<=250 and -52.5<=y<=417.5):return 6
 d=math.hypot(x,y)
 if (abs(x)-220 if y<=JOIN else d-237.5)>1e-8:return 3 if y<=JOIN and x<0 else 4 if y<=JOIN else 5
 if d<=40 and y>=0:return 0
 if abs(x)<=80 and y<=137.5:return 1
 return 2
def features(rows):
 xy=np.array([[s[2],s[3]] for s in rows],dtype=float);x=xy[:,0];y=xy[:,1];d=np.hypot(x,y)/300
 zz=np.array([s[5] for s in rows]);one=np.eye(6)[zz]
 return np.column_stack([d,d*d,np.abs(x)/250,y/417.5,np.abs(np.arctan2(x,y))/math.pi,one])
def sigmoid(x):return 1/(1+np.exp(-np.clip(x,-30,30)))
def logit(p):p=np.clip(p,.01,.99);return np.log(p/(1-p))
def metrics(y,p):
 if not len(y):return None
 p=np.clip(p,1e-6,1-1e-6);return {'roc_auc':float(roc_auc_score(y,p)) if len(set(y))>1 else None,'log_loss':float(np.mean(-y*np.log(p)-(1-y)*np.log(1-p))),'brier_score':float(np.mean((y-p)**2)),'n_shots':len(y)}
def zcounts(rows):
 z=np.zeros((6,2),dtype=int)
 for r in rows:z[r[5],0]+=r[4];z[r[5],1]+=1
 return z
registry=[];catalog=[];coverage=[];season_results=[]
for year in range(2017,2027):
 start=year-1;p=ROOT/f'shot-data/raw/shotdetail_{start}.tar.xz';digest=hashlib.sha256(p.read_bytes()).hexdigest();raw=collections.defaultdict(list);seen={};duplicates=0;conflicts=0
 with tarfile.open(p) as t:
  member=next(m for m in t.getmembers() if m.isfile() and m.name.endswith('.csv'))
  for r in csv.DictReader(io.TextIOWrapper(t.extractfile(member))):
   gid=r['GAME_ID'].zfill(10)
   if not gid.startswith('002') or r['SHOT_ATTEMPTED_FLAG']!='1':continue
   key=(gid,int(r['GAME_EVENT_ID']));pid=int(r['PLAYER_ID']);x=float(r['LOC_X']);y=float(r['LOC_Y']);made=int(r['SHOT_MADE_FLAG']);three=r['SHOT_TYPE'].startswith('3');date=r['GAME_DATE']
   item=(pid,x,y,made,three,date)
   if key in seen:
    if seen[key]!=item:conflicts+=1
    else:duplicates+=1
    continue
   seen[key]=item;raw[pid].append([date,gid,x,y,made,zone(x,y),int(three)])
 if conflicts:raise ValueError('Conflicting events '+str(year))
 leaguegames=sorted({(r[0],r[1]) for shots in raw.values() for r in shots});cut1=int(len(leaguegames)*.6);cut2=int(len(leaguegames)*.8);league_train={g[1] for g in leaguegames[:cut1]};league_valid={g[1] for g in leaguegames[cut1:cut2]};league_test={g[1] for g in leaguegames[cut2:]};assert not (league_train&league_valid or league_train&league_test or league_valid&league_test)
 splits={};alltrain=[]
 for pid,shots in raw.items():
  good=sorted([s for s in shots if s[5]<6 and (s[5]>=3)==bool(s[6])],key=lambda s:(s[0],s[1]))
  games=sorted({(s[0],s[1]) for s in good});n=len(games);trainids={g[1] for g in games if g[1] in league_train};validids={g[1] for g in games if g[1] in league_valid};ngt=len(trainids);ngv=len(validids)
  train=[s for s in good if s[1] in trainids];valid=[s for s in good if s[1] in validids];test=[s for s in good if s[1] not in trainids|validids]
  splits[pid]=(good,train,valid,test,len(games),len(trainids),len(validids),len(games)-ngt-ngv)
  alltrain.extend(train)
 X=features(alltrain);Y=np.array([s[4] for s in alltrain]);model=LogisticRegression(C=1.,max_iter=300,solver='lbfgs').fit(X,Y);coef=model.coef_[0];intercept=float(model.intercept_[0]);league=zcounts(alltrain);prior=(league[:,0]+1)/(league[:,1]+2)
 data={};totalshots=0
 for pid,player in byid.items():
  stats=next((s for s in player['seasons'] if s['year']==year and s['total']),None)
  if not stats:continue
  shots=raw.get(pid,[]);totalshots+=len(shots);cells=collections.defaultdict(lambda:[0,0]);zs=[[0,0] for i in range(8)];shotcounts=[0,0,0,0]
  for s in shots:
   x,y,m,z=s[2],s[3],s[4],s[5];zs[z][0]+=m;zs[z][1]+=1;shotcounts[0]+=1;shotcounts[1]+=m;shotcounts[2]+=s[6];shotcounts[3]+=m*s[6]
   if z<6:
    col=min(int((x+250)//25),19);line=min(int((y+52.5)//25),18);c=cells[(col,line)];c[0]+=m;c[1]+=1
  expected=[stats['totals'][k] for k in ['fga','fgm','fg3a','fg3m']];status='complete' if expected==shotcounts else 'provider_discrepancy';coverage.append({'player_id':pid,'player_name':player['name'],'season':f'{start}-{str(year)[2:]}','status':status,'expected':expected,'recorded':shotcounts})
  entry={'bref':player['id'],'counts':shotcounts,'expected':expected,'coverage':status,'cells':[[i,j,*v] for (i,j),v in sorted(cells.items())],'zones':zs,'ft':[stats['totals']['ftm'],stats['totals']['fta']], 'model':None}
  split=splits.get(pid)
  if split and len(split[0]):
   good,train,valid,test,ng,ntr,nv,nte=split;tc=zcounts(train);strength=50;offsets=logit((tc[:,0]+strength*prior)/(tc[:,1]+strength))-logit(prior)
   if valid and len(valid)>=20:
    VX=features(valid);VY=np.array([r[4] for r in valid]);vz=np.array([r[5] for r in valid]);best=None
    for alpha in [20,50,100,200]:
     off=logit((tc[:,0]+alpha*prior)/(tc[:,1]+alpha))-logit(prior);scores=metrics(VY,sigmoid(VX@coef+intercept+off[vz]));
     if best is None or scores['log_loss']<best[0]:best=(scores['log_loss'],alpha,off)
    _,strength,offsets=best
   baseline=(sum(r[4] for r in train)+20*float(Y.mean()))/(len(train)+20);testmetrics=metrics(np.array([r[4] for r in test]),sigmoid(features(test)@coef+intercept+offsets[np.array([r[5] for r in test])])) if test else None
   bm=metrics(np.array([r[4] for r in test]),np.full(len(test),baseline)) if test else None
   validmetrics=metrics(np.array([r[4] for r in valid]),sigmoid(features(valid)@coef+intercept+offsets[np.array([r[5] for r in valid])])) if valid else None
   modelhash=hashlib.sha256(json.dumps({'version':version,'source':digest,'coef':coef.tolist(),'intercept':intercept,'offsets':offsets.tolist()},sort_keys=True).encode()).hexdigest()[:10];mid=f's{pid}-{start}-{str(year)[2:]}-v2-{modelhash}'
   entry['model']={'id':mid,'offsets':[round(float(v),12) for v in offsets],'alpha':strength,'zone_train':tc.tolist(),'n_shots':len(good),'n_games':ng,'split':[[len(train),ntr],[len(valid),nv],[len(test),nte]],'test':testmetrics,'baseline':bm,'validation':validmetrics,'baseline_p':baseline,'limited':len(test)<30 or nte<5 or len(train)<30,'pooled_only':len(train)<30}
   registry.append({'model_id':mid,'player_id':pid,'player_name':player['name'],'season':f'{start}-{str(year)[2:]}','n_shots':len(good),'created_at':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'selected_model':'spatial_eb','source':'NBA shot archive · pooled spatial model','is_spatial':True,'small_sample':entry['model']['limited'],'coverage':status})
  data[str(pid)]=entry
 pack={'year':year,'source_commit':'e829d4678be1e075f99e5d41a1c5f97089be446b','source_sha256':digest,'fetched_at':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime(p.stat().st_mtime)),'coef':[round(float(v),12) for v in coef],'intercept':round(intercept,12),'prior':prior.tolist(),'split_game_counts':[len(league_train),len(league_valid),len(league_test)],'split_cutoff_dates':[leaguegames[cut1-1][0],leaguegames[cut2-1][0]],'players':data}
 encoded=json.dumps(pack,separators=(',',':'),allow_nan=False).encode();(OUT/f'{year}.json.gz').write_bytes(gzip.compress(encoded,compresslevel=9,mtime=0));season_results.append({'year':year,'shots':totalshots,'players':len(data),'exact':sum(v['coverage']=='complete' for v in data.values()),'bytes':len(encoded),'compressed':(OUT/f'{year}.json.gz').stat().st_size,'duplicates':duplicates})
 print('BUILT',season_results[-1],flush=True)
for p in a['players']:
 pid=mapping[p['id']];catalog.append({'id':pid,'name':p['name'],'archive_id':p['id'],'seasons':sorted([f"{s['year']-1}-{str(s['year'])[2:]}" for s in p['seasons'] if s['total']],reverse=True)})
(OUT/'registry.json').write_text(json.dumps(registry,separators=(',',':')));(OUT/'catalog.json').write_text(json.dumps(catalog,separators=(',',':')));(OUT/'coverage.json').write_text(json.dumps({'seasons':season_results,'playerSeasons':coverage,'models':len(registry),'players':len(catalog),'version':version},separators=(',',':')))
print('DONE',len(catalog),'players',len(coverage),'player seasons',len(registry),'spatial estimates',sum(s['shots'] for s in season_results),'shots',flush=True)
