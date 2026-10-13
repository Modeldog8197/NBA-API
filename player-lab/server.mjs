import {createShotHandler} from './shot-runtime.mjs';
const pending = new Map();
const memoryCache = new Map();
const norm = value => String(value).normalize('NFD').replace(/[\u0300-\u036f]/g,'').toLowerCase().replace(/[^a-z0-9]/g,'');
export const ratio = (a,b) => Number.isFinite(a)&&Number.isFinite(b)&&b>0?a/b:null;
export function normalizeLogs(raw, type='regular') {
 if(!Array.isArray(raw.seasonTypes)||!Array.isArray(raw.names)) throw new Error('Game-log schema changed.');
 const rows=new Map();
 for(const season of raw.seasonTypes){
  const title=season.displayName?.toLowerCase()||'';
  const kind=title.includes('play in')||title.includes('play-in')?'playin':title.includes('postseason')||title.includes('playoff')?'playoffs':title.includes('regular season')?'regular':'other';
  if(kind!==type)continue;
  for(const category of season.categories||[])for(const item of category.events||[]){
   const e=raw.events?.[item.eventId];if(!e||!Array.isArray(item.stats))continue;const note=(e.eventNote||'').toLowerCase();if(e.team?.isAllStar||e.opponent?.isAllStar||/all[- ]star|rising stars/.test(note)||/in-season tournament.*championship|nba cup.*(championship|final\b)/.test(note))continue;
   const s=Object.fromEntries(raw.names.map((n,i)=>[n,item.stats[i]]));
   const n=k=>{const v=s[k];return v===undefined||v===''||v==='--'?null:Number(v)};
   const pair=k=>(s[k]||'').split('-').map(Number);
   const [fgm,fga]=pair('fieldGoalsMade-fieldGoalsAttempted');const [fg3m,fg3a]=pair('threePointFieldGoalsMade-threePointFieldGoalsAttempted');const [ftm,fta]=pair('freeThrowsMade-freeThrowsAttempted');
   const pts=n('points');
   rows.set(e.id,{id:e.id,date:e.gameDate,opponent:e.opponent?.abbreviation||'—',team:e.team?.abbreviation||'—',home:e.atVs!=='@',result:e.gameResult||'',score:e.score||'',min:n('minutes'),pts,reb:n('totalRebounds'),ast:n('assists'),stl:n('steals'),blk:n('blocks'),tov:n('turnovers'),fgm,fga,fg3m,fg3a,ftm,fta,ts:ratio(pts,2*(fga+.44*fta))});
  }
 }
 return [...rows.values()].sort((a,b)=>a.date.localeCompare(b.date));
}
export function checkLogCoverage(player,season,type,rows){if(type!=='regular')return null;const total=player.seasons.find(r=>r.year===season&&r.total);if(!total)return null;const mismatches=['pts','reb','ast','fgm','fga','fg3m','fg3a','ftm','fta'].filter(k=>rows.reduce((sum,r)=>sum+(r[k]??0),0)!==total.totals[k]);return {expected:total.gp,returned:rows.length,status:rows.length<total.gp?'partial':rows.length!==total.gp||mismatches.length?'inconsistent':'complete',mismatches};}
export function normalizeScoreboard(raw){
 if(!Array.isArray(raw.events))throw new Error('Scoreboard schema changed.');
 return raw.events.map(e=>{const c=e.competitions?.[0];if(!c)return null;return {id:e.id,date:e.date,name:e.shortName,state:e.status?.type?.state,detail:e.status?.type?.shortDetail||e.status?.type?.description,teams:(c.competitors||[]).map(t=>({id:t.id,name:t.team?.abbreviation,fullName:t.team?.displayName,score:t.score,home:t.homeAway==='home',winner:t.winner}))}}).filter(Boolean);
}
async function upstreamJSON(url, attempts=2, timeout=9000){
 let last;for(let i=0;i<attempts;i++)try{const r=await fetch(url,{headers:{Accept:'application/json'},signal:AbortSignal.timeout(timeout)});if(!r.ok)throw new Error('Provider returned '+r.status);return await r.json()}catch(e){last=e;if(i+1<attempts)await new Promise(r=>setTimeout(r,300*(i+1)))}throw last;
}
function json(body,status=200){return new Response(JSON.stringify(body),{status,headers:{'Content-Type':'application/json; charset=utf-8','Cache-Control':'private, no-store','X-Content-Type-Options':'nosniff'}})}
export function createHandler(archive,assets,savedLogs,shotData){
 const shotHandler=createShotHandler(shotData);
 const byId=new Map(archive.players.map(p=>[p.id,p]));
 async function cached(request,ctx,key,ttl,source,loader,fallback){
  let cache=null;try{cache=globalThis.caches?.default||null}catch{}const ck=new Request(new URL('/__sports_cache/v2/'+key,request.url));let old=memoryCache.get(key)||null;try{const stored=cache?await cache.match(ck):null;if(stored)old=await stored.json()}catch{cache=null}
  if(old&&Date.now()-Date.parse(old.fetchedAt)<ttl*1000)return {...old,stale:false};
  if(pending.has(key))return pending.get(key);
  const task=(async()=>{try{const data=await loader();const value={data,fetchedAt:new Date().toISOString(),source,stale:false};memoryCache.set(key,value);if(memoryCache.size>500)memoryCache.delete(memoryCache.keys().next().value);if(cache)try{const put=cache.put(ck,new Response(JSON.stringify(value),{headers:{'Content-Type':'application/json','Cache-Control':'public, max-age=2592000'}})).catch(()=>{});ctx?.waitUntil?ctx.waitUntil(put):await put}catch{}return value}catch(e){const value=old||fallback;if(value)return {...value,stale:true,warning:'The provider is unavailable. Showing the last saved response.'};throw e}finally{pending.delete(key)}})();pending.set(key,task);return task;
 }
 async function espnId(player,request,ctx){
  if(player.espn)return player.espn;
  const resolved=await cached(request,ctx,'identity-'+player.id,86400*30,'ESPN identity search',async()=>{
   const raw=await upstreamJSON('https://site.web.api.espn.com/apis/search/v2?query='+encodeURIComponent(player.name)+'&limit=10&type=player&sport=basketball&league=nba');
   const hits=(raw.results||[]).filter(r=>r.type==='player').flatMap(r=>r.contents||[]).filter(p=>p.defaultLeagueSlug==='nba'&&norm(p.displayName)===norm(player.name));
   const ids=[...new Set(hits.map(h=>h.uid?.match(/~a:(\d+)$/)?.[1]).filter(Boolean))];if(ids.length!==1)throw new Error('No unambiguous game-log identity was found for this player.');return ids[0];
  });return resolved.data;
 }
 return {async fetch(request,env,ctx){
  const u=new URL(request.url),path=u.pathname;
  try{
   if(assets[path]){if(request.method!=='GET'&&request.method!=='HEAD')return json({error:'Method not allowed'},405);const a=assets[path];return new Response(request.method==='HEAD'?null:a.body,{headers:{'Content-Type':a.type,'Cache-Control':'private, max-age=300','X-Content-Type-Options':'nosniff','Referrer-Policy':'same-origin','Content-Security-Policy':"default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; font-src https://fonts.gstatic.com; img-src 'self' data: https://a.espncdn.com; connect-src 'self'; object-src 'none'; base-uri 'self'; frame-ancestors 'self' https://chatgpt.com"}})}
   if(path.startsWith('/legacy-api/')){
    const expanded=await shotHandler(request.clone());if(expanded)return expanded;
    const p=path.slice('/legacy-api'.length);
    if(p==='/session')return json({preparation_enabled:false,requires_key:true,session_token:null});
    const allowedGet=/^\/(health|seasons|models(?:\/[a-zA-Z0-9-]+)?|players\/search|players\/\d+\/seasons|player\/(freethrow|shot-chart)|court)$/.test(p);
    const allowedPost=p==='/predict'||p==='/predict/batch';
    if(!(request.method==='GET'&&allowedGet||request.method==='POST'&&allowedPost))return json({detail:'This private site supports saved model inference. Training remains available in the original project.'},403);
    if(u.search.length>500)return json({detail:'Query too long'},400);
    const body=request.method==='POST'?await request.text():undefined;if(body&&body.length>60000)return json({detail:'Request too large'},413);
    const response=await fetch('https://nba-api-modeldog8197.onrender.com'+p+u.search,{method:request.method,headers:{Accept:'application/json',...(body?{'Content-Type':'application/json'}:{})},body,signal:AbortSignal.timeout(85000)});
    const text=await response.text();const bodyText=p==='/health'&&response.ok?JSON.stringify({...JSON.parse(text),training_enabled:false}):text;return new Response(bodyText,{status:response.status,headers:{'Content-Type':'application/json','Cache-Control':'private, no-store'}});
   }
   if(request.method!=='GET')return json({error:'Method not allowed'},405);
   if(path==='/api/archive')return json(archive);
   if(path==='/api/live')return json(await cached(request,ctx,'live',25,'ESPN scoreboard',async()=>normalizeScoreboard(await upstreamJSON('https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard'))));
   const log=path.match(/^\/api\/players\/([a-z0-9]+)\/logs$/);
   if(log){const player=byId.get(log[1]);const season=Number(u.searchParams.get('season'));const type=u.searchParams.get('type')||'regular';if(!player||!archive.seasons.includes(season)||!['regular','playoffs','playin'].includes(type))return json({error:'Invalid player, season or game type.'},400);
    const key=player.id+'-'+season+'-'+type;
    const result=await cached(request,ctx,'logs-'+key,season===Math.max(...archive.seasons)?3600:86400,'ESPN game logs',async()=>{const id=await espnId(player,request,ctx);const raw=await upstreamJSON('https://site.api.espn.com/apis/common/v3/sports/basketball/nba/athletes/'+id+'/gamelog?season='+season);return normalizeLogs(raw,type)},savedLogs[key]);return json({...result,coverage:checkLogCoverage(player,season,type,result.data)});
   }
   const game=path.match(/^\/api\/games\/(\d{6,12})$/);
   if(game)return json(await cached(request,ctx,'game-'+game[1],25,'ESPN game summary',async()=>{const r=await upstreamJSON('https://site.api.espn.com/apis/site/v2/sports/basketball/nba/summary?event='+game[1]);if(!r.header||!r.boxscore)throw new Error('Box score unavailable');return {header:r.header,players:r.boxscore.players||[]}}));
   return json({error:'Not found'},404);
  }catch(e){console.error('NBA data route failure',path,String(e?.message||e).slice(0,180));return json({error:path.startsWith('/legacy-api')?'The shot-analysis service is waking or unavailable. Retry shortly.':e.message==='No unambiguous game-log identity was found for this player.'?e.message:'The data provider is temporarily unavailable. Please retry shortly.',detail:path.startsWith('/legacy-api')?'The shot-analysis service is waking or unavailable. Retry shortly.':'No statistics have been estimated or substituted.'},503)}
 }};
}
