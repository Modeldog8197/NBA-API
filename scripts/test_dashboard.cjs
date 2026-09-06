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
  console.log('Dashboard checks passed: rendering, calibration, stale predictions/models/seasons, hosted access, and unavailable-data recovery.');
})().catch(error => { console.error(error); process.exitCode=1; });
