/* All predictions and shot classifications come from the selected server bundle.
   UI controls are in feet; the API uses tenths of a foot. */
'use strict';

const $ = (id) => document.getElementById(id);
const deployment = typeof window !== 'undefined' && window.NBA_DEPLOYMENT
  ? window.NBA_DEPLOYMENT : { mode: 'local', apiBaseUrl: '' };
const apiUrl = path => `${deployment.apiBaseUrl.replace(/\/+$/, '')}${path}`;
const state = {
  models: [], player: null, model: null, season: '', location: { x: 0, y: 150 },
  generation: 0, predictionSequence: 0, ftSequence: 0, heatmapSequence: 0,
  searchSequence: 0, searchResults: [], searchIndex: -1, trainingEnabled: false,
  trainingJob: null, trainedModel: null, controllers: {}, shotChart: null,
  session: { preparation_enabled: false, requires_key: true, session_token: null },
};
let preparationPoll;
const featureNames = {
  LOC_X: 'Horizontal position', LOC_Y: 'Position toward half court', SHOT_DISTANCE: 'Distance to basket',
  SHOT_ANGLE: 'Angle to basket', SHOT_ANGLE_ABS: 'Absolute shot angle', SHOT_TYPE_ENC: 'Shot value',
  loc_x: 'Horizontal position', loc_y: 'Position toward half court', distance_ft: 'Distance to basket',
  shot_distance: 'Distance to basket', shot_angle: 'Angle to basket', shot_angle_abs: 'Absolute shot angle',
  is_three: 'Three-point location', shot_value: 'Shot value',
};
const format = new Intl.NumberFormat('en-US');
const finite = (v) => v !== null && v !== undefined && Number.isFinite(Number(v));
const fixed = (v, digits = 3) => finite(v) ? Number(v).toFixed(digits) : '—';
const percentage = (v) => finite(v) ? `${(Number(v) * 100).toFixed(1)}%` : '—';
const clamp = (v, min, max) => Math.max(min, Math.min(max, v));

function element(tag, className, content) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (content !== undefined) node.textContent = content;
  return node;
}

function abort(name) {
  state.controllers[name]?.abort();
  delete state.controllers[name];
}

async function api(path, options = {}, channel) {
  if (channel) abort(channel);
  const controller = new AbortController();
  if (channel) state.controllers[channel] = controller;
  const timeout = setTimeout(() => controller.abort('timeout'), path.includes('freethrow') ? 60000 : 45000);
  try {
    const response = await fetch(apiUrl(path), { ...options, signal: controller.signal, headers: { Accept: 'application/json', ...(options.body ? { 'Content-Type': 'application/json' } : {}), ...options.headers } });
    const data = await response.json().catch(() => ({}));
    if (!response.ok) {
      const detail = data.detail;
      let message = typeof detail === 'string' ? detail : Array.isArray(detail) ? detail.map((d) => `${d.loc?.slice(1).join('.') || 'Input'}: ${d.msg}`).join('; ') : detail?.message || data.message;
      if (!message) message = `The request could not be completed (${response.status}).`;
      const error = new Error(message); error.status = response.status; throw error;
    }
    return data;
  } catch (error) {
    if (controller.signal.reason === 'timeout') throw new Error('The server took too long to respond. Please try again.');
    if (error instanceof TypeError) throw new Error('Cannot reach the server. Check that the application is running, then try again.');
    throw error;
  } finally {
    clearTimeout(timeout);
    if (channel && state.controllers[channel] === controller) delete state.controllers[channel];
  }
}

function showMessage(message, success = false) {
  $('global-message').textContent = message;
  $('global-message').classList.toggle('success', success);
  $('global-message').hidden = !message;
}

function predictionStatus(message, error = false) {
  $('prediction-status').textContent = message;
  $('prediction-status').classList.toggle('error', error);
}

function clearPrediction() {
  abort('prediction');
  state.predictionSequence += 1;
  $('prediction-panel').setAttribute('aria-busy', 'false');
  $('probability-value').textContent = '—';
  $('probability-unit').hidden = true;
  $('probability-fill').style.width = '0%';
  $('expected-points').textContent = '—';
  $('shot-value').textContent = '—';
  $('shot-type').textContent = state.model ? 'awaiting analysis' : 'select a model';
  $('shot-zone').textContent = 'Awaiting analysis';
  const empty = element('div', 'empty-explanation');
  empty.append(element('span', '', '≋'), element('p', '', 'Analyze a shot to see which features shape its prediction.'));
  $('explanation-content').replaceChildren(empty);
  $('predict-button').disabled = !state.model;
}

function invalidateSelection() {
  state.generation += 1;
  abort('heatmap'); abort('freethrow'); abort('player-seasons'); abort('preparation');
  clearTimeout(preparationPoll);
  state.ftSequence += 1; state.heatmapSequence += 1;
  $('heatmap-cells').replaceChildren();
  $('heatmap-loading').hidden = true;
  $('heatmap-legend').hidden = true;
  state.shotChart = null;
  renderShootingStats();
  clearPrediction();
}

function refreshTrainingContext() {
  $('training-context').textContent = state.player && state.season
    ? `Create a model for ${state.player.name} · ${state.season} regular season.`
    : 'Choose a player and season above, then start training.';
  $('train-button').disabled = !state.trainingEnabled || !state.player || !state.season || Boolean(state.trainingJob);
  if (!state.trainingEnabled) $('key-help').textContent = 'Training is disabled on this server. The project README explains local training and configuration.';
}

function modelLabel(model) {
  const date = model.created_at ? new Date(model.created_at) : null;
  const dateText = date && !Number.isNaN(date.getTime()) ? date.toLocaleDateString(undefined, { month: 'short', day: 'numeric', year: 'numeric' }) : model.model_id;
  const demo = /synthetic|fixture|demo/i.test(model.source || '') ? ' · DEMO' : '';
  return `${dateText} · ${format.format(model.n_shots ?? 0)} shots${demo}`;
}

function updateModelOptions(preferredId = null) {
  const models = state.models.filter((m) => (!state.player || Number(m.player_id) === Number(state.player.id)) && (!state.season || m.season === state.season));
  $('model-select').replaceChildren();
  const placeholder = element('option', '', models.length ? 'Choose a saved version' : 'No model for this selection');
  placeholder.value = '';
  $('model-select').append(placeholder);
  models.forEach((model) => {
    const option = element('option', '', modelLabel(model));
    option.value = model.model_id;
    $('model-select').append(option);
  });
  $('model-select').disabled = !models.length;
  $('empty-model-guide').hidden = models.length > 0;
  const selected = preferredId ? models.find((m) => m.model_id === preferredId) : models[0];
  $('model-select').value = selected?.model_id || '';
  selectModel(selected || null);
}

async function selectPlayer(player) {
  const previousSeason = state.season;
  state.player = { id: Number(player.id), name: player.name };
  $('player-search').value = player.name;
  $('player-help').textContent = `NBA player ID ${player.id}`;
  hideSearch();
  state.season = '';
  selectModel(null);
  const generation = state.generation;
  $('season').replaceChildren(element('option', '', 'Loading seasons…'));
  $('season').disabled = true;
  $('model-select').replaceChildren(element('option', '', 'Choose a season'));
  $('model-select').disabled = true;
  preparationStatus('Loading available seasons…', false, 0);
  loadFreeThrows();
  refreshTrainingContext();
  let seasons = state.models.filter(m => Number(m.player_id) === Number(player.id)).map(m => m.season);
  try {
    if (Number(player.id) !== 0) {
      const result = await api(`/players/${encodeURIComponent(player.id)}/seasons`, {}, 'player-seasons');
      if (generation !== state.generation || Number(state.player?.id) !== Number(player.id)) return;
      seasons = [...new Set([...result.seasons, ...seasons])];
    }
  } catch (error) {
    if (error.name === 'AbortError' || generation !== state.generation) return;
    if (!seasons.length) {
      preparationStatus(`Available seasons could not be loaded. ${error.message}`, true);
      $('season').replaceChildren(element('option', '', 'Seasons unavailable'));
      $('prepare-model-button').textContent = 'Retry season lookup';
      return;
    }
    showMessage('The NBA season list is unavailable. Saved seasons are still available.');
  }
  if (generation !== state.generation) return;
  seasons.sort().reverse();
  $('season').replaceChildren();
  seasons.forEach(season => { const option = element('option', '', season); option.value = season; $('season').append(option); });
  state.season = seasons.includes(previousSeason) ? previousSeason : seasons[0] || '';
  $('season').value = state.season;
  $('season').disabled = !seasons.length;
  if (!seasons.length) $('season').append(element('option', '', 'No supported seasons'));
  updateModelOptions(); loadFreeThrows(); refreshTrainingContext();
  if (!state.season) preparationStatus('No recorded field-goal seasons from 1997–98 onward are available for this player.');
  else ensureSelectedModel();
}

function preparationStatus(message, retry = false, progress = null) {
  $('preparation-status').textContent = message;
  $('prepare-model-button').hidden = !retry;
  $('prepare-model-button').disabled = !state.player;
  $('prepare-model-button').textContent = 'Prepare player model';
  $('preparation-progress').hidden = progress === null;
  $('preparation-message').textContent = message;
  if (progress !== null) $('preparation-meter').value = progress;
}

async function acceptPreparedModel(result, generation, player, season) {
  if (generation !== state.generation || Number(state.player?.id) !== Number(player.id) || state.season !== season) return;
  if (Number(result.player_id) !== Number(player.id) || result.season !== season || !result.model_id) throw new Error('Model identity does not match the selected player and season.');
  const model = await api(`/models/${encodeURIComponent(result.model_id)}`, {}, 'preparation');
  if (generation !== state.generation) return;
  if (Number(model.player_id) !== Number(player.id) || model.season !== season) throw new Error('The prepared model has an inconsistent identity.');
  state.models = [model, ...state.models.filter(item => item.model_id !== model.model_id)];
  updateModelOptions(model.model_id);
  preparationStatus(`${player.name} · ${season} — model ready.`);
}

async function followPreparation(result, generation, player, season, attempt = 0) {
  if (generation !== state.generation) return;
  if (result.status === 'done' || result.status === 'ready') {
    await acceptPreparedModel(result, generation, player, season); return;
  }
  if (result.status === 'error' || result.status === 'unavailable') {
    preparationStatus(result.message || 'A model could not be prepared for this season.', true);
    predictionStatus(result.message || 'Player model unavailable.', true); return;
  }
  if (!result.job_id || attempt > 360) throw new Error('Model preparation status is unavailable. Try again to reconnect to the job.');
  const progress = { queued: 5, fetching: 20, training: 55, evaluating: 80, publishing: 95 };
  preparationStatus(result.message, false, progress[result.status] || 10);
  preparationPoll = setTimeout(async () => {
    if (generation !== state.generation) return;
    try {
      const next = await api(`/models/preparation/${encodeURIComponent(result.job_id)}`, {}, 'preparation');
      await followPreparation(next, generation, player, season, attempt + 1);
    } catch (error) {
      if (error.name !== 'AbortError' && generation === state.generation) preparationStatus(error.message, true);
    }
  }, 1500);
}

async function ensureSelectedModel(explicit = false) {
  if (!state.player) return;
  if (!state.season) { if (explicit) selectPlayer(state.player); return; }
  if (state.model) { preparationStatus(`${state.player.name} · ${state.season} — model ready.`); return; }
  const generation = state.generation, player = { ...state.player }, season = state.season;
  if (!state.session.preparation_enabled) {
    preparationStatus('No saved model for this season. Player preparation is not enabled on this server.'); return;
  }
  const headers = {};
  if (state.session.session_token) headers['X-Local-Session'] = state.session.session_token;
  else {
    const key = $('training-key').value.trim();
    if (!explicit || !key) {
      preparationStatus('This player needs a model. Enter the server access key in Advanced controls to prepare it.', true);
      if (explicit) { $('training-panel').hidden = false; $('show-training').setAttribute('aria-expanded', 'true'); $('training-key').focus(); }
      return;
    }
    headers.Authorization = `Bearer ${key}`;
  }
  preparationStatus('Preparing the selected player and season…', false, 5);
  try {
    const result = await api('/models/prepare', { method: 'POST', headers, body: JSON.stringify({ player_id: player.id, season }) }, 'preparation');
    await followPreparation(result, generation, player, season);
  } catch (error) {
    if (error.name !== 'AbortError' && generation === state.generation) preparationStatus(error.message, true);
  }
}

function selectModel(model) {
  invalidateSelection();
  state.model = model;
  $('predict-button').disabled = !model;
  $('heatmap-toggle').disabled = !state.player || !state.season || Number(state.player.id) === 0;
  $('prediction-player').textContent = model?.player_name || (state.player ? state.player.name : 'Choose your model.');
  $('prediction-subtitle').textContent = model ? `${model.season} · ${format.format(model.n_shots ?? 0)} recorded shots` : 'Select a player and season to analyze a shot.';
  const demo = /synthetic|fixture|demo/i.test(model?.source || '');
  $('prediction-badge').textContent = model ? demo ? 'SYNTHETIC DEMO' : 'MODEL READY' : 'NO MODEL';
  $('model-help').textContent = model ? 'Predictions stay tied to this saved version.' : 'Train a model for this player and season.';
  const quality = model?.evaluation?.selected_minus_constant_test_log_loss;
  $('model-quality-note').hidden = !finite(quality) || quality <= 0;
  predictionStatus(model ? 'Analyzing the selected location…' : 'Select a player and season. The model will be prepared when needed.');
  renderMetadata(model);
  $('constant-model-note').hidden = model?.selected_model !== 'constant_baseline';
  loadShotChart();
  if (model) {
    predict();
  }
}

function drawMarker() {
  const { x, y } = state.location;
  $('shot-marker').setAttribute('transform', `translate(${x + 300} ${y + 92.5})`);
  $('shot-line').setAttribute('x2', String(x + 300));
  $('shot-line').setAttribute('y2', String(y + 92.5));
  $('shot-x').value = String(Number((x / 10).toFixed(2)));
  $('shot-y').value = String(Number((y / 10).toFixed(2)));
  $('shot-distance').textContent = `${(Math.hypot(x, y) / 10).toFixed(1)} ft`;
  $('court-location').textContent = `X ${(x / 10).toFixed(1)} ft  /  Y ${(y / 10).toFixed(1)} ft`;
  $('court').setAttribute('aria-valuenow', (x / 10).toFixed(1));
  $('court').setAttribute('aria-valuetext', `${(x / 10).toFixed(1)} feet horizontal, ${(y / 10).toFixed(1)} feet from hoop toward half court`);
  document.querySelectorAll('.presets button').forEach((button) => button.classList.toggle('active', Math.abs(Number(button.dataset.x) * 10 - x) < 1 && Math.abs(Number(button.dataset.y) * 10 - y) < 1));
}

let predictDebounce;
function moveShot(x, y, immediate = false) {
  if (!Number.isFinite(x) || !Number.isFinite(y)) return;
  state.location = { x: Math.round(clamp(x, -250, 250) * 100) / 100, y: Math.round(clamp(y, -52.5, 417.5) * 100) / 100 };
  clearTimeout(predictDebounce);
  clearPrediction(); drawMarker();
  renderShootingStats();
  if (state.model) {
    predictionStatus('Updating for this location…');
    if (immediate) predict(); else predictDebounce = setTimeout(predict, 220);
  }
}

async function predict() {
  clearTimeout(predictDebounce);
  if (!state.model) return;
  const model = state.model;
  const generation = state.generation;
  const sequence = ++state.predictionSequence;
  const location = { ...state.location };
  $('prediction-panel').setAttribute('aria-busy', 'true');
  $('predict-button').disabled = true;
  predictionStatus('Analyzing this shot…');
  try {
    const data = await api('/predict', { method: 'POST', body: JSON.stringify({ model_id: model.model_id, loc_x: location.x, loc_y: location.y }) }, 'prediction');
    if (sequence !== state.predictionSequence || generation !== state.generation || data.model_id !== state.model?.model_id) return;
    $('probability-value').textContent = (data.make_probability * 100).toFixed(1);
    $('probability-unit').hidden = false;
    $('probability-fill').style.width = `${clamp(data.make_probability * 100, 0, 100)}%`;
    $('expected-points').textContent = fixed(data.expected_points, 2);
    $('shot-value').textContent = `${data.shot_value} PTS`;
    $('shot-type').textContent = data.shot_value === 3 ? 'three-point attempt' : 'two-point attempt';
    $('shot-distance').textContent = `${fixed(data.distance_ft, 1)} ft`;
    $('shot-zone').textContent = data.shot_zone || 'Unavailable';
    predictionStatus(`${model.selected_model === 'constant_baseline' ? 'Constant baseline estimate' : 'Estimate'} for ${model.player_name} · ${model.season}.`);
    renderExplanation(data.explanation);
  } catch (error) {
    if (error.name === 'AbortError' || sequence !== state.predictionSequence || generation !== state.generation) return;
    predictionStatus(error.message, true);
  } finally {
    if (sequence === state.predictionSequence && generation === state.generation) {
      $('prediction-panel').setAttribute('aria-busy', 'false');
      $('predict-button').disabled = !state.model;
    }
  }
}

function renderExplanation(explanation) {
  const target = $('explanation-content');
  target.replaceChildren();
  if (!explanation || !Array.isArray(explanation.contributions) || !explanation.contributions.length) {
    target.append(element('p', 'muted', 'A feature explanation is unavailable for this prediction.'));
    return;
  }
  const contributions = explanation.contributions.map((item) => ({ ...item, contribution: item.value })).sort((a, b) => Math.abs(b.contribution) - Math.abs(a.contribution));
  const maximum = Math.max(...contributions.map((item) => Math.abs(item.contribution)), .00001);
  contributions.forEach((item) => {
    const row = element('div', 'contribution');
    const name = featureNames[item.feature] || String(item.feature).replaceAll('_', ' ').toLowerCase();
    const label = element('span', 'contribution-name', name);
    label.title = `${item.feature}: ${finite(item.feature_value) ? fixed(item.feature_value, 3) : item.feature_value}`;
    const chart = element('div', 'contribution-chart'); chart.setAttribute('aria-hidden', 'true');
    const bar = element('span', 'contribution-bar');
    const width = Math.abs(item.contribution) / maximum * 50;
    bar.style.width = `${width}%`;
    bar.style.left = item.contribution < 0 ? `${50 - width}%` : '50%';
    if (item.contribution < 0) bar.style.background = 'var(--orange)';
    chart.append(bar);
    const score = element('span', `contribution-value${item.contribution < 0 ? ' negative' : ''}`, `${item.contribution >= 0 ? '+' : ''}${fixed(item.contribution, 2)}`);
    score.setAttribute('aria-label', `${fixed(item.contribution, 3)} log-odds`);
    row.append(label, chart, score); target.append(row);
  });
  const footer = element('div', 'explanation-footer');
  target.append(element('p', 'evaluation-note', `Method: ${explanation.method || 'unavailable'}`));
  footer.append(element('span', '', `Base score ${fixed(explanation.base_value, 2)} log-odds`));
  const consistent = finite(explanation.reconstruction_error) && explanation.reconstruction_error < .001;
  footer.append(element('span', '', consistent ? '✓ Score reconstruction checked' : 'Score reconstruction unavailable'));
  target.append(footer);
  if (explanation.note) target.append(element('p', 'evaluation-note', explanation.note));
}

function renderMetadata(model) {
  $('sample-size').textContent = model ? format.format(model.n_shots ?? 0) : '—';
  $('metadata-season').textContent = model?.season || '—';
  $('metadata-version').textContent = model?.model_id || '—';
  $('data-source').textContent = model ? `${model.source || 'unknown source'}${model.data_source?.stale ? ' · STALE CACHE' : ''}` : 'NO MODEL SELECTED';
  const date = model?.created_at ? new Date(model.created_at) : null;
  $('metadata-created').textContent = date && !Number.isNaN(date.getTime()) ? `Created ${date.toLocaleString()}` : 'Immutable model bundle';
  const limitations = model?.limitations || model?.data_limitations;
  $('model-limitations').textContent = Array.isArray(limitations) ? limitations.join(' ') : typeof limitations === 'string' ? limitations : '';
  const target = $('evaluation-content'); target.replaceChildren();
  if (!model?.evaluation) {
    target.append(element('p', 'muted', model ? 'Evaluation measurements are unavailable for this model. No accuracy claim is made.' : 'Evaluation results appear here when a model is selected.'));
    return;
  }
  const evaluation = model.evaluation;
  target.append(element('p', 'evaluation-note', `${format.format(model.split?.train?.n_shots || 0)} shots fit the model; ${format.format(model.n_shots || 0)} usable shots across all partitions. Final test: ${format.format(model.split?.final_test?.n_shots || 0)} shots in ${format.format(model.split?.final_test?.n_games || 0)} later games.`));
  const test = evaluation.test || evaluation.final_test || {};
  const modelMetrics = test[model.selected_model] || null;
  const baseline = test.constant_baseline;
  if (modelMetrics) {
    const table = element('table', 'evaluation-table');
    const caption = element('caption', 'eyebrow', 'FINAL TEST SET'); caption.style.textAlign = 'left'; caption.style.paddingBottom = '8px';
    const head = element('thead'); const headerRow = element('tr');
    ['Model', 'ROC-AUC ↑', 'Log loss ↓', 'Brier ↓'].forEach((label) => headerRow.append(element('th', '', label)));
    head.append(headerRow); const body = element('tbody');
    [['Constant baseline', baseline, ''], ['Original XGBoost recipe', test.original_xgboost, ''], ['Selected model', modelMetrics, 'model-row']].forEach(([label, metrics, className]) => {
      if (!metrics) return;
      const row = element('tr', className); row.append(element('td', '', label));
      ['roc_auc', 'log_loss', 'brier_score'].forEach((metric) => row.append(element('td', '', fixed(metrics[metric] ?? (metric === 'brier_score' ? metrics.brier : undefined)))));
      body.append(row);
    });
    table.append(caption, head, body); target.append(table);
  } else {
    target.append(element('p', 'muted', 'This bundle contains evaluation metadata. Open its model record for the full metrics.'));
  }
  const split = model.split?.strategy;
  const note = typeof split === 'string' ? split : split?.description || 'Chronological game groups; validation and final test data are held separate.';
  target.append(element('p', 'evaluation-note', `${note} ↑ Higher is better. ↓ Lower is better.`));
  target.append(element('p', 'evaluation-note', `Selected on validation: ${(model.selected_model || '').replaceAll('_', ' ')}. ${evaluation.interpretation || ''}`));
  const bins = modelMetrics?.calibration_bins;
  renderCalibration(target, bins);
  const link = element('a', 'evaluation-note', 'View full model record ↗');
  link.href = apiUrl(`/models/${encodeURIComponent(model.model_id)}`); link.target = '_blank'; link.rel = 'noopener'; link.style.display = 'inline-block';
  target.append(link);
}

function svgElement(tag, attributes) {
  const node = document.createElementNS('http://www.w3.org/2000/svg', tag);
  Object.entries(attributes).forEach(([key, value]) => node.setAttribute(key, String(value)));
  return node;
}

function renderCalibration(target, data) {
  let points = [];
  if (Array.isArray(data)) points = data.map((bin) => [bin.mean_predicted_probability ?? bin.mean_predicted ?? bin.predicted ?? bin.prob_pred, bin.fraction_of_positives ?? bin.observed_rate ?? bin.observed ?? bin.prob_true]);
  else if (data && Array.isArray(data.mean_predicted_probability || data.prob_pred)) points = (data.mean_predicted_probability || data.prob_pred).map((p, i) => [p, (data.fraction_of_positives || data.prob_true || [])[i]]);
  points = points.filter((point) => point.every(finite));
  if (!points.length) return;
  const container = element('div', 'calibration-chart');
  container.append(element('p', 'eyebrow', 'CALIBRATION · FINAL TEST'));
  const svg = svgElement('svg', { viewBox: '0 0 340 200', role: 'img', 'aria-label': 'Calibration plot comparing predicted probability and observed make rate. The dashed diagonal indicates ideal calibration.' });
  svg.append(svgElement('path', { d: 'M40 15V160H315', class: 'axis' }));
  svg.append(svgElement('path', { d: 'M40 160L315 15', class: 'ideal' }));
  svg.append(svgElement('polyline', { points: points.map(([x, y]) => `${40 + x * 275},${160 - y * 145}`).join(' '), class: 'curve' }));
  points.forEach(([x, y]) => svg.append(svgElement('circle', { cx: 40 + x * 275, cy: 160 - y * 145, r: 3 })));
  [[40, 176, '0%'], [296, 176, '100%'], [3, 20, '100%'], [20, 160, '0%'], [104, 195, 'Predicted make probability']].forEach(([x, y, value]) => { const text = svgElement('text', { x, y }); text.textContent = value; svg.append(text); });
  container.append(svg, element('p', 'evaluation-note', 'Vertical axis: observed make rate. Dashed line: ideal calibration.'));
  target.append(container);
}

async function loadFreeThrows() {
  const sequence = ++state.ftSequence;
  const player = state.player;
  const season = state.season;
  $('ft-value').textContent = '—';
  $('ft-context').textContent = player ? `${player.name} · ${season}` : 'Select a player and season.';
  $('ft-detail').textContent = player ? 'Loading historical NBA free-throw statistics…' : 'Season free-throw percentage from NBA data. Separate from the shot model.';
  if (!player || !season) return;
  if (Number(player.id) === 0) { $('ft-detail').textContent = 'Synthetic demo: no NBA player or historical free-throw statistics.'; return; }
  try {
    const result = await api(`/player/freethrow?player_id=${encodeURIComponent(player.id)}&season=${encodeURIComponent(season)}`, {}, 'freethrow');
    if (sequence !== state.ftSequence || Number(state.player?.id) !== Number(player.id) || state.season !== season) return;
    $('ft-value').textContent = percentage(result.ft_pct);
    $('ft-detail').textContent = `${format.format(result.ftm ?? 0)} made / ${format.format(result.fta ?? 0)} attempts · ${season} NBA historical statistics. Separate from the model.${result.provenance?.stale ? ' Using stale cached data.' : ''}`;
  } catch (error) {
    if (error.name === 'AbortError' || sequence !== state.ftSequence) return;
    $('ft-detail').textContent = `Historical statistics unavailable: ${error.message}`;
  }
}

function heatColor(intensity) {
  const stops = [[254, 229, 217], [239, 101, 72], [153, 0, 13]];
  const p = clamp(Number(intensity), 0, 1) * 2;
  const left = Math.min(Math.floor(p), 1); const factor = p - left;
  return `rgb(${stops[left].map((value, i) => Math.round(value + (stops[left + 1][i] - value) * factor)).join(' ')})`;
}

function drawHeatmap(chart) {
  const fragment = document.createDocumentFragment();
  chart.cells.forEach((cell) => {
    if (!cell.attempts || !finite(cell.intensity)) return;
    const rect = svgElement('rect', { x: cell.loc_x + 300, y: cell.loc_y + 92.5,
      width: cell.width, height: cell.height, fill: heatColor(cell.intensity), opacity: .72 });
    const title = svgElement('title', {});
    title.textContent = `${cell.made}/${cell.attempts} made (${percentage(cell.fg_pct)}) · ${fixed(cell.attempts_per_sq_ft, 2)} attempts/ft²`;
    rect.append(title); fragment.append(rect);
  });
  $('heatmap-cells').replaceChildren(fragment);
  $('heatmap-legend').hidden = !chart.cells.length;
  $('heatmap-scale').textContent = `0–${fixed(chart.max_density, 2)} attempts/ft²`;
}

function selectedShotZone(geometry) {
  const {x, y} = state.location;
  const distance = Math.hypot(x, y);
  const isThree = (y <= geometry.arc_join_y ? Math.abs(x) - geometry.corner_x : distance - geometry.arc_radius) > 1e-8;
  if (isThree) return y <= geometry.arc_join_y ? (x < 0 ? 'Left Corner 3' : 'Right Corner 3') : 'Above the Break 3';
  if (distance <= geometry.restricted_radius && y >= 0) return 'Restricted Area';
  if (Math.abs(x) <= geometry.paint_half_width && y <= geometry.paint_end_y) return 'Paint (Non-RA)';
  return 'Mid-Range';
}

function renderShootingStats() {
  const chart = state.shotChart;
  $('overall-fg').textContent = chart ? percentage(chart.overall.fg_pct) : '—';
  $('overall-fg-counts').textContent = chart ? `${format.format(chart.overall.made)} / ${format.format(chart.overall.attempts)} made` : 'Recorded season attempts';
  $('zone-fg').textContent = '—';
  $('zone-fg-counts').textContent = 'Select a player and season';
  $('zone-fg-label').textContent = 'Selected zone FG%';
  if (!chart) return;
  const name = selectedShotZone(chart.geometry);
  const zone = chart.zones.find(item => item.name === name);
  $('zone-fg-label').textContent = `${name} FG%`;
  $('zone-fg').textContent = zone ? percentage(zone.fg_pct) : '—';
  $('zone-fg-counts').textContent = zone?.attempts ? `${format.format(zone.made)} / ${format.format(zone.attempts)} made${zone.attempts < 20 ? ' · small sample' : ''}` : 'No recorded attempts in this zone';
  // Historical zone counts remain usable even if there is insufficient data to train a model.
  if (!state.model) $('shot-zone').textContent = name;
}

function loadHeatmap() {
  if (!$('heatmap-toggle').checked) return;
  if (state.shotChart) drawHeatmap(state.shotChart);
  else if (state.controllers.heatmap) $('heatmap-loading').hidden = false;
  else loadShotChart();
}

async function loadShotChart() {
  abort('heatmap');
  const player = state.player, season = state.season;
  state.shotChart = null; renderShootingStats();
  $('heatmap-cells').replaceChildren(); $('heatmap-legend').hidden = true;
  $('shot-data-retry').hidden = true;
  $('shot-data-status').textContent = 'Select a player and season for recorded shooting statistics.';
  if (!player || !season) return;
  if (Number(player.id) === 0) { $('shot-data-status').textContent = 'Synthetic demo: historical NBA shot data is unavailable.'; return; }
  const generation = state.generation; const sequence = ++state.heatmapSequence;
  $('heatmap-loading').hidden = !$('heatmap-toggle').checked;
  $('shot-data-status').textContent = 'Loading recorded shot locations and shooting percentages…';
  try {
    const result = await api(`/player/shot-chart?player_id=${encodeURIComponent(player.id)}&season=${encodeURIComponent(season)}`, {}, 'heatmap');
    if (generation !== state.generation || sequence !== state.heatmapSequence || Number(state.player?.id) !== Number(player.id) || state.season !== season) return;
    if (Number(result.player_id) !== Number(player.id) || result.season !== season) throw new Error('Shot data does not match the selected player and season.');
    state.shotChart = result; renderShootingStats();
    const excluded = Object.values(result.excluded).reduce((total, count) => total + count, 0);
    $('shot-data-status').textContent = `${player.name} · ${season}: ${format.format(result.plotted.attempts)} of ${format.format(result.overall.attempts)} recorded attempts plotted. Red shows shot density.${result.provenance?.stale ? ' Using stale cached NBA data.' : ''}${excluded ? ` ${excluded} invalid, duplicate, or conflicting records excluded.` : ''}`;
    if ($('heatmap-toggle').checked) drawHeatmap(result);
  } catch (error) {
    if (error.name === 'AbortError' || generation !== state.generation || sequence !== state.heatmapSequence) return;
    $('shot-data-status').textContent = `Recorded shooting data unavailable. ${error.message}`;
    $('shot-data-retry').hidden = false;
  } finally {
    if (generation === state.generation && sequence === state.heatmapSequence) $('heatmap-loading').hidden = true;
  }
}

function hideSearch() {
  $('player-results').hidden = true;
  $('player-search').setAttribute('aria-expanded', 'false');
  $('player-search').removeAttribute('aria-activedescendant');
  state.searchIndex = -1;
}

function renderSearch(results, message = '') {
  state.searchResults = results; state.searchIndex = -1;
  $('player-results').replaceChildren();
  if (!results.length) $('player-results').append(element('li', 'no-results', message || 'No players found. Try another name.'));
  results.forEach((player, index) => {
    const item = element('li'); item.id = `player-result-${index}`; item.setAttribute('role', 'option'); item.setAttribute('aria-selected', 'false');
    item.append(element('span', '', player.name), element('small', '', String(player.id)));
    item.addEventListener('pointerdown', (event) => { event.preventDefault(); selectPlayer(player); });
    $('player-results').append(item);
  });
  $('player-results').hidden = false;
  $('player-search').setAttribute('aria-expanded', 'true');
}

let searchDebounce;
async function searchPlayers(query) {
  const sequence = ++state.searchSequence;
  if (query.length < 2) { hideSearch(); return; }
  try {
    const result = await api(`/players/search?q=${encodeURIComponent(query)}`, {}, 'search');
    if (sequence !== state.searchSequence || $('player-search').value.trim() !== query) return;
    const players = Array.isArray(result) ? result : result.players || [];
    renderSearch(players.map((player) => ({ id: player.id ?? player.player_id, name: player.name ?? player.full_name })));
  } catch (error) {
    if (error.name !== 'AbortError' && sequence === state.searchSequence) renderSearch([], error.message);
  }
}

let trainingPoll;
async function pollTraining() {
  if (!state.trainingJob) return;
  try {
    const result = await api(`/model/retrain/status?job_id=${encodeURIComponent(state.trainingJob)}`, {}, 'training-status');
    const progress = { queued: 8, fetching: 25, training: 55, evaluating: 78, publishing: 93, done: 100, error: 100 };
    $('progress-fill').style.width = `${progress[result.status] ?? 12}%`;
    $('training-message').textContent = result.message || `Training status: ${result.status}`;
    if (result.status === 'done') {
      state.trainingJob = null; state.trainedModel = result.model_id;
      $('training-key').value = '';
      const data = await api('/models'); state.models = data.models || [];
      // Populate new versions without switching away from the current bundle.
      const currentId = state.model?.model_id;
      const current = state.model;
      const compatible = state.models.filter((model) => Number(model.player_id) === Number(state.player?.id) && model.season === state.season);
      $('model-select').replaceChildren();
      if (!current) { const option = element('option', '', 'Choose a saved version'); option.value = ''; $('model-select').append(option); }
      compatible.forEach((model) => { const option = element('option', '', modelLabel(model)); option.value = model.model_id; $('model-select').append(option); });
      $('model-select').value = currentId || '';
      $('model-select').disabled = !compatible.length;
      $('empty-model-guide').hidden = compatible.length > 0;
      $('use-trained-model').hidden = !result.model_id;
      refreshTrainingContext();
      return;
    }
    if (result.status === 'error') {
      state.trainingJob = null; $('training-key').value = '';
      $('training-message').textContent = result.error || result.message || 'Training failed. Previous model versions are still available.';
      $('progress-fill').style.background = 'var(--orange)';
      refreshTrainingContext(); return;
    }
    trainingPoll = setTimeout(pollTraining, 2500);
  } catch (error) {
    $('training-message').textContent = `Progress temporarily unavailable: ${error.message} Checking again…`;
    trainingPoll = setTimeout(pollTraining, 6000);
  }
}

async function startTraining(event) {
  event.preventDefault();
  if (!state.player || !state.season || !state.trainingEnabled || state.trainingJob) return;
  const key = $('training-key').value.trim();
  if (!key) { $('training-key').focus(); showMessage('Enter the server’s training access key to start a training job.'); return; }
  $('train-button').disabled = true;
  $('training-progress').hidden = false;
  $('use-trained-model').hidden = true;
  $('progress-fill').style.width = '5%'; $('progress-fill').style.background = '';
  $('training-message').textContent = 'Requesting a training job…';
  showMessage('');
  try {
    const result = await api('/model/retrain', { method: 'POST', headers: { Authorization: `Bearer ${key}` }, body: JSON.stringify({ player_id: state.player.id, season: state.season }) }, 'training-start');
    state.trainingJob = result.job_id;
    $('training-key').value = '';
    $('training-message').textContent = result.message || 'Training queued.';
    pollTraining();
  } catch (error) {
    $('training-message').textContent = error.message;
    $('progress-fill').style.background = 'var(--orange)';
    refreshTrainingContext();
  }
}

function bindEvents() {
  $('player-search').addEventListener('input', () => {
    clearTimeout(searchDebounce); abort('search'); state.searchSequence += 1;
    const query = $('player-search').value.trim();
    // Typed text is not a player selection; clear results immediately to avoid stale labels.
    if (state.player && query !== state.player.name) {
      state.player = null; invalidateSelection(); state.model = null;
      $('model-select').replaceChildren(element('option', '', 'Select a player first'));
      $('model-select').disabled = true;
      selectModel(null); loadFreeThrows(); refreshTrainingContext();
      preparationStatus('Select a player from the search results.');
    }
    $('player-help').textContent = 'Select a player from the results.';
    if (query.length < 2) hideSearch(); else searchDebounce = setTimeout(() => searchPlayers(query), 240);
  });
  $('player-search').addEventListener('keydown', (event) => {
    if (event.key === 'Escape') { hideSearch(); return; }
    if (!['ArrowDown', 'ArrowUp', 'Enter'].includes(event.key) || $('player-results').hidden) return;
    if (event.key === 'Enter') {
      if (state.searchIndex >= 0 && state.searchResults[state.searchIndex]) { event.preventDefault(); selectPlayer(state.searchResults[state.searchIndex]); }
      return;
    }
    event.preventDefault();
    if (!state.searchResults.length) return;
    state.searchIndex = clamp(state.searchIndex + (event.key === 'ArrowDown' ? 1 : -1), 0, state.searchResults.length - 1);
    [...$('player-results').children].forEach((item, i) => item.setAttribute('aria-selected', String(i === state.searchIndex)));
    const active = $(`player-result-${state.searchIndex}`);
    $('player-search').setAttribute('aria-activedescendant', active.id); active.scrollIntoView({ block: 'nearest' });
  });
  document.addEventListener('pointerdown', (event) => { if (!event.target.closest('.player-field')) hideSearch(); });
  $('season').addEventListener('change', () => { state.season = $('season').value; invalidateSelection(); updateModelOptions(); loadFreeThrows(); refreshTrainingContext(); ensureSelectedModel(); });
  $('prepare-model-button').addEventListener('click', () => ensureSelectedModel(true));
  $('model-select').addEventListener('change', () => selectModel(state.models.find((m) => m.model_id === $('model-select').value) || null));
  $('show-training').addEventListener('click', () => { const hidden = !$('training-panel').hidden; $('training-panel').hidden = hidden; $('show-training').setAttribute('aria-expanded', String(!hidden)); });
  $('training-form').addEventListener('submit', startTraining);
  $('use-trained-model').addEventListener('click', () => {
    const model = state.models.find((item) => item.model_id === state.trainedModel);
    if (!model) { showMessage('The new model was not found. Refresh the page to reload available versions.'); return; }
    state.season = model.season; $('season').value = model.season;
    state.player = { id: model.player_id, name: model.player_name }; $('player-search').value = model.player_name;
    $('player-help').textContent = `NBA player ID ${model.player_id}`;
    updateModelOptions(model.model_id); loadFreeThrows(); refreshTrainingContext();
    $('use-trained-model').hidden = true; showMessage(`Selected the new model for ${model.player_name} · ${model.season}.`, true);
  });
  $('predict-button').addEventListener('click', predict);
  [$('shot-x'), $('shot-y')].forEach((input) => input.addEventListener('change', () => {
    if (!$('shot-x').checkValidity() || !$('shot-y').checkValidity() || $('shot-x').value === '' || $('shot-y').value === '') { input.reportValidity(); predictionStatus('Enter coordinates within the displayed court bounds.', true); return; }
    moveShot(Number($('shot-x').value) * 10, Number($('shot-y').value) * 10, true);
  }));
  document.querySelectorAll('.presets button').forEach((button) => button.addEventListener('click', () => moveShot(Number(button.dataset.x) * 10, Number(button.dataset.y) * 10, true)));
  $('court').addEventListener('pointerdown', (event) => {
    const svg = $('court'); const matrix = svg.getScreenCTM(); if (!matrix) return;
    const point = new DOMPoint(event.clientX, event.clientY).matrixTransform(matrix.inverse());
    if (point.x < 50 || point.x > 550 || point.y < 40 || point.y > 510) return;
    svg.focus({ preventScroll: true }); moveShot(point.x - 300, point.y - 92.5, true);
  });
  $('court').addEventListener('keydown', (event) => {
    const step = event.shiftKey ? 50 : 10;
    const delta = { ArrowLeft: [-step, 0], ArrowRight: [step, 0], ArrowUp: [0, -step], ArrowDown: [0, step] }[event.key];
    if (!delta) return;
    event.preventDefault(); moveShot(state.location.x + delta[0], state.location.y + delta[1]);
  });
  $('heatmap-toggle').addEventListener('change', () => {
    if ($('heatmap-toggle').checked) loadHeatmap();
    else { $('heatmap-cells').replaceChildren(); $('heatmap-loading').hidden = true; $('heatmap-legend').hidden = true; }
  });
  $('shot-data-retry').addEventListener('click', loadShotChart);
  window.addEventListener('pagehide', () => { Object.keys(state.controllers).forEach(abort); clearTimeout(preparationPoll); clearTimeout(trainingPoll); clearTimeout(predictDebounce); clearTimeout(searchDebounce); $('training-key').value = ''; });
}

async function initialize() {
  bindEvents(); drawMarker();
  document.querySelectorAll('[data-api-path]').forEach(link => {
    link.href = apiUrl(link.dataset.apiPath);
    link.hidden = deployment.mode === 'pages' && !deployment.apiBaseUrl;
  });
  if (deployment.mode === 'pages' && !deployment.apiBaseUrl) {
    $('system-status').replaceChildren(element('span', 'status-dot'), document.createTextNode('AWAITING LIVE SERVICE'));
    showMessage('The dashboard is published. Live predictions and player data will be available once its server is connected.');
    $('player-search').disabled = true;
    $('player-help').textContent = 'Live player search is not connected yet.';
    $('season').disabled = true; $('model-select').disabled = true;
    $('season').replaceChildren(element('option', '', 'Not connected'));
    $('model-select').replaceChildren(element('option', '', 'Not connected'));
    preparationStatus('Live player data is not connected yet.');
    refreshTrainingContext();
    return;
  }
  const [healthResult, seasonsResult, modelsResult, sessionResult] = await Promise.allSettled([api('/health'), api('/seasons'), api('/models'), api('/session')]);
  if (sessionResult.status === 'fulfilled') state.session = sessionResult.value;
  if (healthResult.status === 'fulfilled') {
    state.trainingEnabled = Boolean(healthResult.value.training_enabled);
    $('system-status').classList.add('online');
    $('system-status').replaceChildren(element('span', 'status-dot'), document.createTextNode('SYSTEM ONLINE'));
  } else {
    $('system-status').classList.add('offline');
    $('system-status').replaceChildren(element('span', 'status-dot'), document.createTextNode('CONNECTION UNAVAILABLE'));
    showMessage(healthResult.reason.message);
  }
  const seasons = seasonsResult.status === 'fulfilled' ? seasonsResult.value.seasons || [] : [];
  state.models = modelsResult.status === 'fulfilled' ? modelsResult.value.models || [] : [];
  state.models.forEach((model) => { if (!seasons.includes(model.season)) seasons.push(model.season); });
  $('season').replaceChildren();
  seasons.forEach((season) => { const option = element('option', '', season); option.value = season; $('season').append(option); });
  state.season = seasonsResult.status === 'fulfilled' ? seasonsResult.value.default_season || seasons[0] : seasons[0] || '';
  if (!seasons.length) { const option = element('option', '', 'Seasons unavailable'); option.value = ''; $('season').append(option); }
  const initialModel = state.models.find((model) => !/synthetic|fixture|demo/i.test(model.source || '')) || state.models[0];
  if (initialModel) {
    state.player = { id: initialModel.player_id, name: initialModel.player_name };
    state.season = initialModel.season; $('player-search').value = initialModel.player_name;
    $('player-help').textContent = `NBA player ID ${initialModel.player_id}`;
  }
  $('season').value = state.season;
  if (modelsResult.status === 'rejected') showMessage(`Saved models could not be loaded. ${modelsResult.reason.message}`);
  else if (seasonsResult.status === 'rejected') showMessage(`Season choices could not be loaded. ${seasonsResult.reason.message}`);
  updateModelOptions(initialModel?.model_id);
  refreshTrainingContext(); loadFreeThrows();
  if (initialModel) selectPlayer(state.player);
  else preparationStatus('Search for a player to load their available seasons.');
}

initialize();
