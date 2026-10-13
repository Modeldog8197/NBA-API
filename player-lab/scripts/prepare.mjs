import fs from 'node:fs';
import {normalizeLogs} from '../server.mjs';
const source='data/';
const normal=v=>v.normalize('NFD').replace(/[\u0300-\u036f]/g,'').toLowerCase().replace(/[^a-z0-9]/g,'');
const roster=JSON.parse(fs.readFileSync(source+'espn-roster-source.json')).items;
const names=new Map(roster.map(p=>[normal(p.displayName),p.id]));
const aliases={'nicclaxton':'nicolasclaxton','cameronthomas':'camthomas','cameronjohnson':'camjohnson','kenyonmartinjr':'kjmartin','pjwashington':'pjwashington','garypaytonii':'garypaytonii','robertwilliams':'robertwilliamsiii','lonnie walker iv':'lonniewalkeriv'};
const players=new Map();let recordCount=0;let oldest='9999';
const div=(a,b)=>b>0?a/b:null;
for(let year=2017;year<=2026;year++){
 const raw=JSON.parse(fs.readFileSync('data/season-'+year+'.json'));if(!raw.verifiedCounts||raw.verifiedCounts.playertotals!==raw.playertotals.length||raw.verifiedCounts.playeradvancedstats!==raw.playeradvancedstats.length)throw new Error('Unverified provider counts '+year);if(!raw.playertotals.length)throw new Error('Missing season '+year);oldest=raw.fetchedAt<oldest?raw.fetchedAt:oldest;
 const advanced=new Map(raw.playeradvancedstats.map(r=>[r.playerId+'|'+r.team,r]));
 for(const r of raw.playertotals){if(r.isPlayoff||r.season!==year)throw new Error('Wrong season or type');if(!players.has(r.playerId)){const key=normal(r.playerName);players.set(r.playerId,{id:r.playerId,name:r.playerName,espn:names.get(key)||names.get(aliases[key])||null,seasons:[]})}const a=advanced.get(r.playerId+'|'+r.team)||{};const team=/^(TOT|\dTM)$/.test(r.team)?'TOT':r.team;
 const row={year,team,sourceTeam:r.team,total:team==='TOT',position:r.position,gp:r.games,min:div(r.minutesPg,r.games),pts:div(r.points,r.games),reb:div(r.totalRb,r.games),ast:div(r.assists,r.games),stl:div(r.steals,r.games),blk:div(r.blocks,r.games),tov:div(r.turnovers,r.games),fg:div(r.fieldGoals,r.fieldAttempts),three:div(r.threeFg,r.threeAttempts),ft:div(r.ft,r.ftAttempts),ts:div(r.points,2*(r.fieldAttempts+.44*r.ftAttempts)),efg:div(r.fieldGoals+.5*r.threeFg,r.fieldAttempts),atr:div(r.assists,r.turnovers),usg:a.usagePercent??null,bpm:a.box??null,ws:a.winShares??null,per:a.per??null,vorp:a.vorp??null,ftr:div(r.ftAttempts,r.fieldAttempts),threeRate:div(r.threeAttempts,r.fieldAttempts),pts36:div(r.points*36,r.minutesPg),ast36:div(r.assists*36,r.minutesPg),totals:{pts:r.points,reb:r.totalRb,ast:r.assists,fgm:r.fieldGoals,fga:r.fieldAttempts,fg3m:r.threeFg,fg3a:r.threeAttempts,ftm:r.ft,fta:r.ftAttempts,min:r.minutesPg,tov:r.turnovers}};
 if(row.gp<0||row.totals.fgm>row.totals.fga||row.totals.ftm>row.totals.fta)throw new Error('Inconsistent counts '+r.playerId);
 players.get(r.playerId).seasons.push(row);
 }
}
for(const p of players.values())for(let year=2017;year<=2026;year++){const rows=p.seasons.filter(r=>r.year===year);if(rows.length===1)rows[0].total=true;const totals=rows.filter(r=>r.total).sort((a,b)=>b.gp-a.gp||b.totals.min-a.totals.min);if(totals.length>1)p.seasons=p.seasons.filter(r=>r.year!==year||!r.total||r===totals[0]);if(rows.length&&!rows.some(r=>r.total))throw new Error('Missing combined row '+p.id+' '+year);if(rows.length)recordCount++}
const archive={players:[...players.values()].sort((a,b)=>a.name.localeCompare(b.name)),seasons:Array.from({length:10},(_,i)=>2017+i),recordCount,fetchedAt:oldest,source:'NBA_Go public API',scope:'Regular season 2016–17 through 2025–26'};
fs.writeFileSync('data/archive.json',JSON.stringify(archive));fs.writeFileSync('data/espn-roster.json',JSON.stringify(roster.map(p=>({id:p.id,name:p.displayName}))));
const saved=fs.existsSync('data/saved-logs.json')?JSON.parse(fs.readFileSync('data/saved-logs.json')):{};
saved['curryst01-2024-regular']={data:normalizeLogs(JSON.parse(fs.readFileSync(source+'raw-log-curryst01-2024-regular.json'))),fetchedAt:fs.statSync(source+'raw-log-curryst01-2024-regular.json').mtime.toISOString(),source:'ESPN game logs',stale:false};for(const f of fs.readdirSync('data').filter(f=>f.startsWith('raw-log-'))){const key=f.slice(8,-5);const type=key.split('-').at(-1);saved[key]={data:normalizeLogs(JSON.parse(fs.readFileSync('data/'+f)),type),fetchedAt:fs.statSync('data/'+f).mtime.toISOString(),source:'ESPN game logs',stale:false};}fs.writeFileSync('data/saved-logs.json',JSON.stringify(saved));
console.log(JSON.stringify({players:archive.players.length,recordCount,mapped:archive.players.filter(p=>p.espn).length,seasons:archive.seasons}));
