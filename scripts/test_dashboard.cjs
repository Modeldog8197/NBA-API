// Offline regression checks for actual dashboard rendering and async model isolation.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
class Element {
  constructor(tag = 'div') { this.tagName = tag; this.children = []; this.style = {}; this.textContent = ''; this.value = ''; this.hidden = false; this.attributes = {}; this.classList = {add(){},toggle(){}}; }
  append(...items) { this.children.push(...items); }
  replaceChildren(...items) { this.children = items; this.textContent = ''; }
  setAttribute(k, v) { this.attributes[k] = v; }
  removeAttribute(k) { delete this.attributes[k]; }
}
const nodes = new Map();
const document = {
  getElementById(id) { if (!nodes.has(id)) nodes.set(id, new Element()); return nodes.get(id); },
  createElement: tag => new Element(tag), createElementNS: (_, tag) => new Element(tag),
  createDocumentFragment: () => new Element(), querySelectorAll: () => [],
};
const context = vm.createContext({document, Intl, AbortController, setTimeout, clearTimeout, console, TypeError});
const code = fs.readFileSync('static/app.js', 'utf8').replace(/initialize\(\);\s*$/, '');
vm.runInContext(code, context);
const run = source => vm.runInContext(source, context);
const textOf = el => [el.textContent, ...el.children.map(textOf)].join(' ');
run(`renderExplanation({method:'TreeSHAP',base_value:-.2,raw_margin:.3,reconstruction_error:0,
  contributions:[{feature:'LOC_X',label:'X',value:.6,feature_value:230},{feature:'LOC_Y',label:'Y',value:-.1,feature_value:0}]})`);
assert.match(textOf(nodes.get('explanation-content')), /\+0\.60/);
assert.match(textOf(nodes.get('explanation-content')), /-0\.10/);
assert.match(textOf(nodes.get('explanation-content')), /reconstruction checked/);
assert.doesNotMatch(JSON.stringify(nodes.get('explanation-content')), /NaN/);
run(`renderMetadata({model_id:'version-a',n_shots:100,season:'2023-24',source:'nba',selected_model:'candidate_xgboost',
  split:{train:{n_shots:60},final_test:{n_shots:20,n_games:2}},evaluation:{final_test:{
  constant_baseline:{roc_auc:.5,log_loss:.7,brier_score:.25},original_xgboost:{roc_auc:.55,log_loss:.8,brier_score:.26},
  candidate_xgboost:{roc_auc:.6,log_loss:.65,brier_score:.23,calibration_bins:[{predicted:.4,observed:.45,count:20}]}}}})`);
assert.match(textOf(nodes.get('evaluation-content')), /0\.650/);
assert.match(textOf(nodes.get('evaluation-content')), /Original XGBoost recipe/);
assert.match(textOf(nodes.get('evaluation-content')), /60 shots fit the model/);
assert.match(textOf(nodes.get('evaluation-content')), /CALIBRATION/);

(async () => {
  let requestedUrl;
  context.fetch = async url => { requestedUrl = url; return {ok:true,json:async()=>({status:'ok'})}; };
  run(`deployment.apiBaseUrl='https://api.example.org/nba/';`);
  await run(`api('/health')`);
  assert.equal(requestedUrl, 'https://api.example.org/nba/health', 'Pages requests use the configured backend, including its path prefix');
  run(`deployment.apiBaseUrl='';`);
  await run(`api('/health')`);
  assert.equal(requestedUrl, '/health', 'Local requests remain on the FastAPI origin');
  let finish;
  context.fetch = () => new Promise(resolve => { finish = resolve; });
  run(`state.model={model_id:'old-model',player_name:'Old player',season:'2023-24'}; state.location={x:0,y:100};`);
  const pending = run('predict()');
  run(`invalidateSelection();state.model={model_id:'new-model',player_name:'New player',season:'2024-25'};`);
  finish({ok:true,json:async()=>({model_id:'old-model',make_probability:.99,shot_value:2,expected_points:1.98})});
  await pending;
  assert.equal(nodes.get('probability-value').textContent, '—', 'Old response must not appear after model change');
  assert.equal(nodes.get('explanation-content').children[0].className, 'empty-explanation');

  run(`state.player={id:42,name:'Earlier player'};state.season='2023-24';`);
  const waitingForModel = run(`acceptPreparedModel({player_id:42,season:'2023-24',model_id:'old-ready'},state.generation,{id:42},'2023-24')`);
  run(`invalidateSelection();state.player={id:84,name:'Current player'};state.season='2024-25';state.model={model_id:'current-model'};`);
  finish({ok:true,json:async()=>({player_id:42,season:'2023-24',model_id:'old-ready'})});
  await waitingForModel;
  assert.equal(run('state.model.model_id'), 'current-model', 'A late prepared model must not change selection');

  run(`state.model=null;state.session={preparation_enabled:true,requires_key:true,session_token:null};`);
  context.fetch = () => { throw new Error('Hosted preparation must not run without an explicit access key'); };
  await run('ensureSelectedModel()');
  assert.match(nodes.get('preparation-status').textContent, /access key/);
  assert.equal(nodes.get('prepare-model-button').hidden, false);

  await run(`followPreparation({status:'unavailable',message:'Not enough historical shots.'},state.generation,state.player,state.season)`);
  assert.match(nodes.get('preparation-status').textContent, /Not enough historical shots/);
  assert.equal(nodes.get('preparation-progress').hidden, true);

  let finishSeasons;
  context.fetch = () => new Promise(resolve => { finishSeasons=resolve; });
  const waitingForSeasons = run(`selectPlayer({id:42,name:'Earlier player'})`);
  run(`invalidateSelection();state.player={id:84,name:'Current player'};state.season='2024-25';`);
  finishSeasons({ok:true,json:async()=>({player_id:42,seasons:['2023-24']})});
  await waitingForSeasons;
  assert.equal(run('state.season'), '2024-25', 'A late season lookup must not change the new player season');
  assert.equal(run('state.player.id'), 84);
  const chart = {
    player_id:84,season:'2024-25',overall:{made:3,attempts:10,fg_pct:.3},plotted:{attempts:10},
    geometry:{corner_x:220,arc_join_y:Math.sqrt(237.5**2-220**2),arc_radius:237.5,restricted_radius:40,paint_half_width:80,paint_end_y:137.5},
    zones:[{name:'Restricted Area',made:2,attempts:3,fg_pct:2/3},{name:'Left Corner 3',made:1,attempts:7,fg_pct:1/7},
      {name:'Right Corner 3',made:0,attempts:0,fg_pct:null}],
    cells:[{loc_x:-250,loc_y:-52.5,width:25,height:25,made:0,attempts:2,fg_pct:0,intensity:1,attempts_per_sq_ft:.32}],
    max_density:.32,excluded:{invalid_rows:0,duplicate_rows:0,conflicting_rows:0},provenance:{stale:false},
  };
  context.chartFixture = chart;
  run(`state.model=null;state.shotChart=chartFixture;state.location={x:0,y:20};renderShootingStats();drawHeatmap(chartFixture);`);
  assert.equal(nodes.get('overall-fg').textContent, '30.0%');
  assert.equal(nodes.get('zone-fg').textContent, '66.7%');
  run(`moveShot(-230,0)`);
  assert.equal(nodes.get('zone-fg').textContent, '14.3%', 'Zone percentage changes with selected location even without a model');
  assert.equal(nodes.get('overall-fg').textContent, '30.0%', 'Location does not incorrectly filter the overall season denominator');
  run(`moveShot(230,0)`);
  assert.equal(nodes.get('zone-fg').textContent, '—', 'Empty zones are not rendered as 0%');
  assert.match(nodes.get('zone-fg-counts').textContent, /No recorded attempts/);
  assert.equal(nodes.get('heatmap-cells').children[0].children.length, 1, 'Only occupied density cells are colored');
  assert.equal(nodes.get('heatmap-cells').children[0].children[0].attributes.fill, 'rgb(153 0 13)');
  assert.equal(nodes.get('heatmap-cells').children[0].children[0].attributes.x, '50');

  for (const [x,y,zone] of [[220,0,'Mid-Range'],[0,40,'Restricted Area'],[0,41,'Paint (Non-RA)'],
    [80,137.5,'Paint (Non-RA)'],[0,237.5,'Mid-Range'],[0,237.5001,'Above the Break 3']]) {
    run(`state.location={x:${x},y:${y}}`);
    assert.equal(run('selectedShotZone(chartFixture.geometry)'), zone);
  }

  let finishChart;
  context.fetch = () => new Promise(resolve => { finishChart = resolve; });
  const oldChart = run('loadShotChart()');
  assert.equal(nodes.get('overall-fg').textContent, '—', 'A reload clears previous counts immediately');
  run(`invalidateSelection();state.season='2023-24';`);
  finishChart({ok:true,json:async()=>chart});
  await oldChart;
  assert.equal(run('state.shotChart'), null, 'Late data for an old season must not overwrite the current selection');

  const changed = {...chart,season:'2023-24',overall:{made:9,attempts:10,fg_pct:.9}};
  context.fetch = async () => ({ok:true,json:async()=>changed});
  await run('loadShotChart()');
  assert.equal(nodes.get('overall-fg').textContent, '90.0%', 'Changing season updates the numerator and denominator');
  context.fetch = async () => ({ok:false,status:503,json:async()=>({detail:'NBA unavailable'})});
  await run('loadShotChart()');
  assert.equal(nodes.get('overall-fg').textContent, '—', 'A failed reload cannot keep stale percentages');
  assert.equal(nodes.get('shot-data-retry').hidden, false);

  const pendingPredictions = [];
  context.fetch = () => new Promise(resolve => pendingPredictions.push(resolve));
  run(`state.model={model_id:'current-model',player_name:'Current player',season:'2023-24'};state.location={x:0,y:20};`);
  const firstPrediction = run('predict()');
  run(`state.location={x:230,y:0}`);
  const nextPrediction = run('predict()');
  pendingPredictions[1]({ok:true,json:async()=>({model_id:'current-model',make_probability:.35,shot_value:3,expected_points:1.05})});
  await nextPrediction;
  pendingPredictions[0]({ok:true,json:async()=>({model_id:'current-model',make_probability:.8,shot_value:2,expected_points:1.6})});
  await firstPrediction;
  assert.equal(nodes.get('probability-value').textContent, '35.0', 'A late old-location prediction cannot overwrite the latest location');
  console.log('Dashboard checks passed: density rendering, observed percentages, empty zones, filter updates, stale data/locations, model isolation, and recovery.');
})().catch(error => { console.error(error); process.exitCode=1; });
