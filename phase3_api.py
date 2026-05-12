"""
Basketball Shot Quality Model — Phase 3: FastAPI Prediction Server

Prerequisite: run train_and_save_model.py first to generate model.json,
              encoder.joblib, and model_meta.joblib.

Usage:
    pip install -r requirements.txt
    python train_and_save_model.py          # one-time
    python phase3_api.py                    # starts server on port 8000

Endpoints:
    GET  /                          -- Interactive UI (browser)
    GET  /health                    -- Model metadata
    GET  /players/search?q=...      -- Search active NBA players
    GET  /player/freethrow?...      -- Player free throw stats
    POST /model/retrain             -- Retrain model for a new player
    GET  /model/retrain/status      -- Poll retraining progress
    POST /predict                   -- Single-shot xFG% prediction
    POST /predict/batch             -- Batch predictions (up to 100)
    GET  /docs                      -- Swagger UI
"""

import math
import time
import threading
import traceback
import numpy as np
import pandas as pd
import joblib
import xgboost as xgb
from pathlib import Path

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field, field_validator

from nba_api.stats.static import players as nba_players_static


# ─────────────────────────────────────────────
# Startup — load model artefacts
# ─────────────────────────────────────────────
BASE = Path(__file__).parent

MODEL_PATH   = BASE / "model.json"
ENCODER_PATH = BASE / "encoder.joblib"
META_PATH    = BASE / "model_meta.joblib"

for _p in [MODEL_PATH, ENCODER_PATH, META_PATH]:
    if not _p.exists():
        raise FileNotFoundError(
            f"\n\n  Missing artefact: {_p}\n"
            "  Run  python train_and_save_model.py  first.\n"
        )

model   = xgb.XGBClassifier()
model.load_model(str(MODEL_PATH))
encoder = joblib.load(ENCODER_PATH)
meta    = joblib.load(META_PATH)

SHOT_TYPE_CLASSES = meta["shot_type_classes"]
FEATURE_COLS      = meta["feature_cols"]

print(f"Model loaded  | {meta['player']} ({meta['season']}) | AUC={meta['roc_auc']:.3f}")


# ─────────────────────────────────────────────
# Background retraining state
# ─────────────────────────────────────────────
_retrain_state: dict = {"status": "idle", "message": "", "player": None, "season": None}
_retrain_lock  = threading.Lock()


def _do_retrain(player_name: str, season: str) -> None:
    """Runs in a background thread. Re-trains and hot-swaps the global model."""
    global model, encoder, meta, SHOT_TYPE_CLASSES

    from nba_api.stats.endpoints import ShotChartDetail
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder
    from sklearn.metrics import log_loss, roc_auc_score

    try:
        with _retrain_lock:
            _retrain_state.update(status="fetching", message=f"Fetching shots for {player_name}…")

        info = nba_players_static.find_players_by_full_name(player_name)
        if not info:
            raise ValueError(f"Player '{player_name}' not found.")
        player_id = info[0]["id"]

        time.sleep(1)
        endpoint = ShotChartDetail(
            team_id=0,
            player_id=player_id,
            season_nullable=season,
            season_type_all_star="Regular Season",
            context_measure_simple="FGA",
        )
        df = endpoint.get_data_frames()[0].head(5_000).copy()

        with _retrain_lock:
            _retrain_state.update(status="training", message=f"Training on {len(df):,} shots…")

        required = ["LOC_X", "LOC_Y", "SHOT_DISTANCE", "SHOT_TYPE", "SHOT_MADE_FLAG"]
        df = df.dropna(subset=required)
        for col in ["LOC_X", "LOC_Y", "SHOT_DISTANCE", "SHOT_MADE_FLAG"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.dropna(subset=required).reset_index(drop=True)

        df["SHOT_ANGLE"]     = np.arctan2(df["LOC_X"], df["LOC_Y"])
        df["SHOT_ANGLE_ABS"] = df["SHOT_ANGLE"].abs()

        le = LabelEncoder()
        df["SHOT_TYPE_ENC"] = le.fit_transform(df["SHOT_TYPE"])

        X = df[FEATURE_COLS].values
        y = df["SHOT_MADE_FLAG"].values.astype(int)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        new_model = xgb.XGBClassifier(
            objective="binary:logistic",
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            use_label_encoder=False,
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1,
        )
        new_model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

        y_prob = new_model.predict_proba(X_test)[:, 1]
        ll     = float(log_loss(y_test, y_prob))
        auc    = float(roc_auc_score(y_test, y_prob))

        new_model.save_model(str(MODEL_PATH))
        joblib.dump(le, ENCODER_PATH)
        new_meta = {
            "feature_cols":       FEATURE_COLS,
            "shot_type_classes":  list(le.classes_),
            "player":             player_name,
            "season":             season,
            "log_loss":           ll,
            "roc_auc":            auc,
            "n_shots":            len(df),
        }
        joblib.dump(new_meta, META_PATH)

        model             = new_model
        encoder           = le
        meta              = new_meta
        SHOT_TYPE_CLASSES = new_meta["shot_type_classes"]

        with _retrain_lock:
            _retrain_state.update(
                status="done",
                message=f"Ready — {len(df):,} shots | AUC {auc:.3f}",
                player=player_name,
                season=season,
            )

    except Exception as exc:
        with _retrain_lock:
            _retrain_state.update(status="error", message=str(exc))
        traceback.print_exc()


# ─────────────────────────────────────────────
# Prediction helper
# ─────────────────────────────────────────────

def engineer_and_predict(loc_x, loc_y, shot_distance, shot_type_str):
    _type_map = {
        "2pt":           "2PT Field Goal",
        "3pt":           "3PT Field Goal",
        "2PT Field Goal": "2PT Field Goal",
        "3PT Field Goal": "3PT Field Goal",
    }
    resolved = _type_map.get(shot_type_str.strip(), shot_type_str.strip())
    if resolved not in SHOT_TYPE_CLASSES:
        raise ValueError(f"Invalid shot_type '{shot_type_str}'.")

    angle     = math.atan2(loc_x, loc_y)
    angle_abs = abs(angle)
    enc       = int(encoder.transform([resolved])[0])

    X   = np.array([[loc_x, loc_y, shot_distance, enc, angle, angle_abs]])
    xfg = float(model.predict_proba(X)[0, 1])

    if shot_distance <= 8:
        zone = "Paint / Post"
    elif shot_distance <= 23.75:
        zone = "Mid-Range"
    else:
        zone = "Three-Point"

    return {
        "xfg_pct":         round(xfg, 4),
        "xfg_pct_display": f"{xfg:.1%}",
        "confidence_low":  round(max(0.0, xfg - 0.05), 4),
        "confidence_high": round(min(1.0, xfg + 0.05), 4),
        "shot_angle_rad":  round(angle, 4),
        "shot_angle_deg":  round(math.degrees(angle), 2),
        "shot_zone":       zone,
        "shot_type":       resolved,
    }


# ─────────────────────────────────────────────
# Schemas
# ─────────────────────────────────────────────

class ShotRequest(BaseModel):
    loc_x:         float = Field(..., ge=-300, le=300)
    loc_y:         float = Field(..., ge=-100, le=900)
    shot_distance: float = Field(..., ge=0, le=100)
    shot_type:     str

    @field_validator("shot_type")
    @classmethod
    def validate_shot_type(cls, v):
        allowed = {"2pt", "3pt", "2PT Field Goal", "3PT Field Goal"}
        if v.strip() not in allowed:
            raise ValueError(f"shot_type must be one of: {sorted(allowed)}")
        return v.strip()

    model_config = {"json_schema_extra": {"example": {
        "loc_x": -220, "loc_y": 88, "shot_distance": 24, "shot_type": "3pt"
    }}}


class ShotResponse(BaseModel):
    xfg_pct:         float
    xfg_pct_display: str
    confidence_low:  float
    confidence_high: float
    shot_angle_rad:  float
    shot_angle_deg:  float
    shot_zone:       str
    shot_type:       str


class BatchRequest(BaseModel):
    shots: list[ShotRequest] = Field(..., min_length=1, max_length=100)


class RetrainRequest(BaseModel):
    player_name: str = Field(..., min_length=2)
    season:      str = Field("2023-24", pattern=r"^\d{4}-\d{2}$")


# ─────────────────────────────────────────────
# App
# ─────────────────────────────────────────────

app = FastAPI(
    title="Basketball xFG% API",
    description="Expected Field Goal % predictions powered by XGBoost + NBA shot data.",
    version="3.1.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────

@app.get("/players/search", summary="Search active NBA players")
def search_players(q: str = ""):
    if len(q) < 2:
        return []
    q_low = q.lower()
    all_p = nba_players_static.get_active_players()
    hits  = [p for p in all_p if q_low in p["full_name"].lower()][:10]
    return [{"id": p["id"], "name": p["full_name"]} for p in hits]


@app.get("/player/freethrow", summary="Get a player's most recent season FT%")
def get_freethrow(player_name: str = ""):
    from nba_api.stats.endpoints import PlayerCareerStats

    name = player_name.strip() or meta["player"]
    info = nba_players_static.find_players_by_full_name(name)
    if not info:
        raise HTTPException(status_code=404, detail=f"Player '{name}' not found.")

    player_id = info[0]["id"]
    time.sleep(0.5)

    career = PlayerCareerStats(player_id=player_id)
    df     = career.get_data_frames()[0]

    if df.empty:
        raise HTTPException(status_code=404, detail="No career stats found.")

    latest = df.iloc[-1]
    ft_pct = float(latest.get("FT_PCT") or 0)
    ftm    = int(latest.get("FTM")    or 0)
    fta    = int(latest.get("FTA")    or 0)
    season = str(latest.get("SEASON_ID", "N/A"))

    return {
        "player":         name,
        "season":         season,
        "ft_pct":         ft_pct,
        "ft_pct_display": f"{ft_pct:.1%}",
        "ftm":            ftm,
        "fta":            fta,
    }


@app.post("/model/retrain", summary="Retrain model for a different player")
def start_retrain(req: RetrainRequest, background_tasks: BackgroundTasks):
    with _retrain_lock:
        if _retrain_state["status"] in ("fetching", "training"):
            raise HTTPException(409, "A retrain is already in progress.")
        _retrain_state.update(
            status="fetching",
            message="Starting…",
            player=req.player_name,
            season=req.season,
        )
    background_tasks.add_task(_do_retrain, req.player_name, req.season)
    return {"queued": True, "player": req.player_name, "season": req.season}


@app.get("/model/retrain/status", summary="Poll retraining progress")
def retrain_status():
    with _retrain_lock:
        return dict(_retrain_state)


@app.get("/health", summary="Model health and metadata")
def health():
    return {
        "status":      "ok",
        "player":      meta["player"],
        "season":      meta["season"],
        "n_shots":     meta["n_shots"],
        "log_loss":    meta["log_loss"],
        "roc_auc":     meta["roc_auc"],
        "shot_types":  SHOT_TYPE_CLASSES,
    }


@app.post("/predict", response_model=ShotResponse, summary="Predict xFG% for one shot")
def predict(shot: ShotRequest):
    try:
        return engineer_and_predict(
            shot.loc_x, shot.loc_y, shot.shot_distance, shot.shot_type
        )
    except ValueError as exc:
        raise HTTPException(422, str(exc))


@app.post("/predict/batch", summary="Predict xFG% for multiple shots (max 100)")
def predict_batch(batch: BatchRequest):
    results = []
    for i, s in enumerate(batch.shots):
        try:
            results.append(
                engineer_and_predict(s.loc_x, s.loc_y, s.shot_distance, s.shot_type)
            )
        except ValueError as exc:
            raise HTTPException(422, f"Shot {i}: {exc}")
    return {"predictions": results, "count": len(results)}


# ─────────────────────────────────────────────
# Browser UI
# ─────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def root():
    player_name = meta["player"]
    season      = meta["season"]
    auc         = meta["roc_auc"]
    n_shots     = meta["n_shots"]
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>xFG% Shot Predictor</title>
<style>
:root {{
  --bg:     #0d1117;
  --panel:  #161b22;
  --card:   #1a2030;
  --border: #2a3348;
  --gold:   #f0a500;
  --gold2:  #fbbf24;
  --muted:  #6b7280;
  --text:   #dde1f0;
  --red:    #f87171;
  --green:  #4ade80;
  --radius: 12px;
}}

*, *::before, *::after {{
  box-sizing: border-box;
  margin: 0;
  padding: 0;
}}

body {{
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  background: var(--bg);
  color: var(--text);
  min-height: 100vh;
  padding: 28px 16px 64px;
}}

.page {{
  max-width: 840px;
  margin: 0 auto;
}}

/* ── Header ── */
.header {{
  display: flex;
  align-items: center;
  gap: 14px;
  margin-bottom: 28px;
}}

.header-title {{
  font-size: 1.3rem;
  font-weight: 800;
  letter-spacing: -.02em;
}}

.header-sub {{
  font-size: .78rem;
  color: var(--muted);
  margin-top: 2px;
}}

.model-badge {{
  margin-left: auto;
  background: var(--panel);
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 8px 14px;
  font-size: .78rem;
  line-height: 1.6;
  text-align: right;
}}

.model-badge strong {{
  color: var(--gold);
  display: block;
  font-size: .9rem;
}}

/* ── Layout grid ── */
.grid {{
  display: grid;
  grid-template-columns: 300px 1fr;
  gap: 20px;
  align-items: start;
}}

@media (max-width: 660px) {{
  .grid {{ grid-template-columns: 1fr; }}
}}

/* ── Cards ── */
.card {{
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 18px;
}}

.card-label {{
  font-size: .68rem;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: .1em;
  color: var(--muted);
  margin-bottom: 12px;
}}

/* ── Court ── */
.court-card {{
  padding: 14px;
  cursor: crosshair;
}}

.court-card svg {{
  display: block;
  width: 100%;
  height: auto;
  border-radius: 8px;
}}

.court-hint {{
  font-size: .7rem;
  color: var(--muted);
  text-align: center;
  margin-top: 8px;
}}

/* ── Right column ── */
.right-col {{
  display: flex;
  flex-direction: column;
  gap: 16px;
}}

/* ── Player search ── */
.search-wrap {{
  position: relative;
}}

.search-input {{
  width: 100%;
  padding: 9px 12px 9px 34px;
  background: var(--panel);
  border: 1px solid var(--border);
  border-radius: 8px;
  color: var(--text);
  font-size: .9rem;
  outline: none;
  transition: border-color .2s;
}}

.search-input::placeholder {{ color: var(--muted); }}
.search-input:focus {{ border-color: var(--gold); }}

.search-icon {{
  position: absolute;
  left: 10px;
  top: 50%;
  transform: translateY(-50%);
  font-size: .9rem;
  color: var(--muted);
  pointer-events: none;
}}

.dropdown {{
  position: absolute;
  z-index: 50;
  width: 100%;
  top: calc(100% + 4px);
  background: var(--panel);
  border: 1px solid var(--border);
  border-radius: 8px;
  box-shadow: 0 8px 24px rgba(0,0,0,.5);
  overflow: hidden;
  display: none;
}}

.dropdown.open {{ display: block; }}

.dd-item {{
  padding: 9px 14px;
  cursor: pointer;
  font-size: .88rem;
  border-bottom: 1px solid var(--border);
  transition: background .15s;
}}

.dd-item:last-child {{ border-bottom: none; }}
.dd-item:hover {{ background: var(--card); color: var(--gold); }}

.retrain-row {{
  margin-top: 10px;
  display: none;
  align-items: center;
  gap: 8px;
}}

.retrain-row.show {{ display: flex; }}

.retrain-btn {{
  flex: 1;
  padding: 8px 12px;
  font-size: .83rem;
  font-weight: 700;
  background: var(--gold);
  color: #111318;
  border: none;
  border-radius: 7px;
  cursor: pointer;
  transition: background .2s;
}}

.retrain-btn:hover {{ background: var(--gold2); }}
.retrain-btn:disabled {{ background: #374151; color: var(--muted); cursor: not-allowed; }}

.retrain-status {{
  font-size: .75rem;
  color: var(--muted);
  margin-top: 6px;
  min-height: 16px;
}}

.spinner {{
  display: inline-block;
  width: 11px;
  height: 11px;
  border: 2px solid #374151;
  border-top-color: var(--gold);
  border-radius: 50%;
  animation: spin .6s linear infinite;
  vertical-align: middle;
  margin-right: 5px;
}}

@keyframes spin {{ to {{ transform: rotate(360deg); }} }}

/* ── Shot detail fields ── */
.info-grid {{
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 10px;
  margin-bottom: 12px;
}}

.info-field label {{
  display: block;
  font-size: .68rem;
  text-transform: uppercase;
  letter-spacing: .08em;
  color: var(--muted);
  margin-bottom: 4px;
}}

.info-field input {{
  width: 100%;
  padding: 8px 10px;
  background: var(--panel);
  border: 1px solid var(--border);
  border-radius: 7px;
  color: var(--text);
  font-size: .9rem;
  outline: none;
  transition: border-color .2s;
}}

.info-field input:focus {{ border-color: var(--gold); }}

/* ── Shot type toggle ── */
.type-toggle {{
  display: flex;
  gap: 8px;
  margin-bottom: 14px;
}}

.type-btn {{
  flex: 1;
  padding: 8px;
  font-size: .85rem;
  font-weight: 600;
  background: var(--panel);
  border: 1.5px solid var(--border);
  border-radius: 8px;
  color: var(--muted);
  cursor: pointer;
  transition: all .18s;
  text-align: center;
}}

.type-btn.active {{
  border-color: var(--gold);
  color: var(--gold);
  background: rgba(240,165,0,.1);
}}

.type-btn.hidden {{ display: none; }}

/* ── Predict button ── */
.predict-btn {{
  width: 100%;
  padding: 11px;
  font-size: .95rem;
  font-weight: 800;
  background: var(--gold);
  color: #111318;
  border: none;
  border-radius: 9px;
  cursor: pointer;
  letter-spacing: .01em;
  transition: background .2s, transform .1s;
}}

.predict-btn:hover  {{ background: var(--gold2); }}
.predict-btn:active {{ transform: scale(.98); }}

/* ── Result card ── */
.result-card {{
  transition: opacity .3s, transform .3s;
}}

.result-card.hidden {{
  opacity: 0;
  transform: translateY(8px);
  pointer-events: none;
}}

.gauge-wrap {{
  display: flex;
  justify-content: center;
  align-items: center;
  margin-bottom: 16px;
  position: relative;
}}

.gauge-wrap svg {{ overflow: visible; }}

.gauge-center {{
  position: absolute;
  text-align: center;
  pointer-events: none;
}}

.gauge-num {{
  font-size: 2rem;
  font-weight: 900;
  line-height: 1;
}}

.gauge-sub {{
  font-size: .65rem;
  color: var(--muted);
  text-transform: uppercase;
  letter-spacing: .1em;
  margin-top: 2px;
}}

.result-stats {{
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 8px;
  margin-bottom: 12px;
}}

.rstat {{
  background: var(--panel);
  border-radius: 8px;
  padding: 10px 12px;
}}

.rstat .k {{
  font-size: .65rem;
  text-transform: uppercase;
  letter-spacing: .08em;
  color: var(--muted);
}}

.rstat .v {{
  font-size: .9rem;
  font-weight: 700;
  margin-top: 3px;
}}

.model-pills {{
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
  margin-top: 12px;
}}

.mpill {{
  font-size: .72rem;
  padding: 3px 10px;
  border-radius: 99px;
  background: var(--panel);
  border: 1px solid var(--border);
  color: var(--muted);
}}

.mpill span {{ color: var(--text); font-weight: 600; }}

/* ── FT popup ── */
.ft-popup {{
  background: var(--panel);
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 12px 14px;
  margin-top: 10px;
  display: none;
}}

.ft-popup.show {{ display: block; }}

.ft-popup .ft-pct {{
  font-size: 1.6rem;
  font-weight: 800;
  color: var(--gold);
}}

.ft-popup .ft-sub {{
  font-size: .75rem;
  color: var(--muted);
  margin-top: 2px;
}}

/* ── Error ── */
.err {{
  color: var(--red);
  font-size: .8rem;
  margin-top: 8px;
}}
</style>
</head>
<body>
<div class="page">

  <!-- Header -->
  <header class="header">
    <span style="font-size:1.8rem">🏀</span>
    <div>
      <div class="header-title">xFG% Shot Predictor</div>
      <div class="header-sub">Expected Field Goal Probability — XGBoost</div>
    </div>
    <div class="model-badge" id="model-badge">
      <strong id="badge-player">{player_name}</strong>
      <span id="badge-meta">{season} &nbsp;&middot;&nbsp; {n_shots:,} shots &nbsp;&middot;&nbsp; AUC {auc:.3f}</span>
    </div>
  </header>

  <!-- Main grid -->
  <div class="grid">

    <!-- LEFT — Court -->
    <div class="card court-card" id="court-wrap">
      <div class="card-label">Court &mdash; click to place shot</div>

      <!--
        Coordinate system (SCALE = 0.55 px per tenth-of-foot):
          Basket SVG origin: (150, 22)
          LOC_X: negative = left, positive = right
          LOC_Y: 0 = basket, increases toward half-court
        Key positions:
          FT line y   : 22 + 190*0.55 = 126.5
          3PT corner x: 150 +/- 220*0.55 = 29 / 271
          3PT arc start y: 22 + 89.5*0.55 = 71.2
          3PT arc radius: 237.5*0.55 = 130.6
      -->
      <svg id="court-svg" viewBox="0 0 300 295" xmlns="http://www.w3.org/2000/svg">

        <!-- Court surface -->
        <rect width="300" height="295" fill="#152032" rx="8"/>

        <!-- Boundary -->
        <rect x="6" y="6" width="288" height="283"
              fill="none" stroke="#1e3a5a" stroke-width="1.5" rx="6"/>

        <!-- Paint fill -->
        <rect x="106" y="22" width="88" height="104.5"
              fill="rgba(240,165,0,.05)" stroke="#1e3a5a" stroke-width="1.2"/>

        <!-- FT circle upper (solid) -->
        <path d="M 117,126.5 A 33,33 0 0,1 183,126.5"
              fill="none" stroke="#1e3a5a" stroke-width="1.2"/>

        <!-- FT circle lower (dashed) -->
        <path d="M 117,126.5 A 33,33 0 0,0 183,126.5"
              fill="none" stroke="#1e3a5a" stroke-width="1.2" stroke-dasharray="4 3"/>

        <!-- Restricted arc -->
        <path d="M 128,22 A 22,22 0 0,1 172,22"
              fill="none" stroke="#1e3a5a" stroke-width="1.2"/>

        <!-- THREE-POINT LINE (gold) -->
        <line x1="29" y1="6" x2="29" y2="71.2" stroke="#f0a500" stroke-width="2"/>
        <line x1="271" y1="6" x2="271" y2="71.2" stroke="#f0a500" stroke-width="2"/>
        <path d="M 29,71.2 A 130.6,130.6 0 0,1 271,71.2"
              fill="none" stroke="#f0a500" stroke-width="2"/>

        <!-- Backboard -->
        <rect x="133" y="14" width="34" height="3"
              fill="none" stroke="#f0a500" stroke-width="2"/>

        <!-- Basket ring -->
        <circle cx="150" cy="22" r="5"
                fill="none" stroke="#f0a500" stroke-width="2"/>

        <!-- Zone labels -->
        <text x="150" y="105" text-anchor="middle"
              font-size="7.5" fill="#1e3a5a" font-weight="600" letter-spacing="1">MID-RANGE</text>
        <text x="150" y="185" text-anchor="middle"
              font-size="7.5" fill="#1e3a5a" font-weight="600" letter-spacing="1">THREE-POINT</text>

        <!-- FT% button on the free throw line -->
        <g id="ft-btn" style="cursor:pointer" onclick="fetchFTPct()">
          <rect x="122" y="120" width="56" height="15" rx="7.5"
                fill="#152032" stroke="#f0a500" stroke-width="1.5"/>
          <text x="150" y="131" text-anchor="middle"
                font-size="8" font-weight="700" fill="#f0a500"
                style="pointer-events:none">FT %</text>
        </g>

        <!-- Shot dot -->
        <circle id="shot-dot" cx="150" cy="22" r="6"
                fill="#f0a500" fill-opacity="0"
                stroke="#fff" stroke-width="1.5"
                style="filter:drop-shadow(0 0 5px rgba(240,165,0,.9));transition:cx .15s,cy .15s"/>

        <!-- Ripple ring -->
        <circle id="shot-ripple" cx="150" cy="22" r="6"
                fill="none" stroke="#f0a500" stroke-opacity="0" stroke-width="1"/>
      </svg>

      <div class="court-hint">Basket &#8593; &nbsp;&middot;&nbsp; Gold line = 3-point arc &nbsp;&middot;&nbsp; Click FT% for free throw stats</div>
    </div>

    <!-- RIGHT — Controls -->
    <div class="right-col">

      <!-- Player search -->
      <div class="card">
        <div class="card-label">Player Model</div>
        <div class="search-wrap">
          <span class="search-icon">&#128269;</span>
          <input class="search-input" id="player-search"
                 placeholder="Search player name&hellip;"
                 autocomplete="off" spellcheck="false">
          <div class="dropdown" id="player-dd"></div>
        </div>
        <div class="retrain-row" id="retrain-row">
          <button class="retrain-btn" id="retrain-btn" onclick="startRetrain()">
            Train for <span id="retrain-name">&mdash;</span>
          </button>
        </div>
        <div class="retrain-status" id="retrain-status"></div>
      </div>

      <!-- Shot details -->
      <div class="card">
        <div class="card-label">Shot Details</div>
        <div class="info-grid">
          <div class="info-field">
            <label>LOC_X <small style="color:#374151">(tenths ft)</small></label>
            <input type="number" id="loc_x" value="0" step="1" oninput="syncFromFields()">
          </div>
          <div class="info-field">
            <label>LOC_Y <small style="color:#374151">(tenths ft)</small></label>
            <input type="number" id="loc_y" value="0" step="1" oninput="syncFromFields()">
          </div>
          <div class="info-field" style="grid-column:1/-1">
            <label>Distance (feet)</label>
            <input type="number" id="shot_distance" value="0.0" step="0.5" min="0" max="100"
                   oninput="onDistanceChange()">
          </div>
        </div>

        <div class="card-label">Shot Type</div>
        <div class="type-toggle">
          <button class="type-btn active" id="btn-2pt" onclick="selectType('2pt')">2PT</button>
          <button class="type-btn"        id="btn-3pt" onclick="selectType('3pt')">3PT</button>
        </div>

        <button class="predict-btn" onclick="doPredikt()">Calculate xFG%</button>
        <div class="err" id="err-msg"></div>
      </div>

      <!-- Result -->
      <div class="card result-card hidden" id="result-card">
        <div class="card-label">Result</div>

        <!-- Gauge -->
        <div class="gauge-wrap">
          <svg width="170" height="105" viewBox="0 0 170 105">
            <path d="M 15,95 A 70,70 0 0,1 155,95"
                  fill="none" stroke="#1e2a3a" stroke-width="16" stroke-linecap="round"/>
            <path id="gauge-arc"
                  d="M 15,95 A 70,70 0 0,1 155,95"
                  fill="none" stroke="#f0a500" stroke-width="16" stroke-linecap="round"
                  stroke-dasharray="220" stroke-dashoffset="220"
                  style="transition:stroke-dashoffset .7s ease,stroke .4s"/>
          </svg>
          <div class="gauge-center">
            <div class="gauge-num" id="gauge-num" style="color:var(--gold)">&mdash;</div>
            <div class="gauge-sub">xFG%</div>
          </div>
        </div>

        <div class="result-stats">
          <div class="rstat"><div class="k">Zone</div><div class="v" id="r-zone">&mdash;</div></div>
          <div class="rstat"><div class="k">Shot Angle</div><div class="v" id="r-angle">&mdash;</div></div>
          <div class="rstat"><div class="k">Shot Type</div><div class="v" id="r-type">&mdash;</div></div>
          <div class="rstat"><div class="k">Conf. Range</div><div class="v" id="r-ci">&mdash;</div></div>
        </div>

        <div class="model-pills">
          <div class="mpill">Player <span id="pill-player">{player_name}</span></div>
          <div class="mpill">Season <span id="pill-season">{season}</span></div>
          <div class="mpill">AUC <span id="pill-auc">{auc:.3f}</span></div>
        </div>

        <!-- FT% panel inside result -->
        <div class="ft-popup" id="ft-popup">
          <div class="card-label" style="margin-bottom:6px">Free Throw %</div>
          <div class="ft-pct" id="ft-pct-val">&mdash;</div>
          <div class="ft-sub" id="ft-sub">&mdash;</div>
        </div>
      </div>

      <!-- FT% standalone (shown before first prediction) -->
      <div class="card ft-popup" id="ft-popup-standalone">
        <div class="card-label" style="margin-bottom:6px">Free Throw %</div>
        <div class="ft-pct" id="ft-pct-val2">&mdash;</div>
        <div class="ft-sub" id="ft-sub2">&mdash;</div>
      </div>

    </div>
  </div>
</div>

<script>
/* ── Court coordinate constants ── */
const SCALE     = 0.55;
const BASKET_SX = 150;
const BASKET_SY = 22;
const SVG_VW    = 300;
const SVG_VH    = 295;

function nbaToSvg(lx, ly) {{
  return {{ sx: BASKET_SX + lx * SCALE, sy: BASKET_SY + ly * SCALE }};
}}

function svgToNba(sx, sy) {{
  return {{
    lx: (sx - BASKET_SX) / SCALE,
    ly: (sy - BASKET_SY) / SCALE
  }};
}}

/* ── State ── */
let selectedType   = "2pt";
let selectedPlayer = null;
let retrainPoll    = null;

/* ── Court click ── */
const courtSvg  = document.getElementById("court-svg");
const dot       = document.getElementById("shot-dot");
const ripple    = document.getElementById("shot-ripple");

courtSvg.addEventListener("click", function(e) {{
  if (e.target.closest("#ft-btn")) return;

  const rect   = courtSvg.getBoundingClientRect();
  const scaleX = SVG_VW / rect.width;
  const scaleY = SVG_VH / rect.height;
  const sx     = (e.clientX - rect.left) * scaleX;
  const sy     = (e.clientY - rect.top)  * scaleY;

  const csx = Math.max(8, Math.min(SVG_VW - 8, sx));
  const csy = Math.max(BASKET_SY, Math.min(SVG_VH - 8, sy));

  const nba  = svgToNba(csx, csy);
  const lxR  = Math.round(nba.lx);
  const lyR  = Math.round(Math.max(0, nba.ly));
  const dist = +(Math.sqrt(nba.lx * nba.lx + nba.ly * nba.ly) / 10).toFixed(1);

  document.getElementById("loc_x").value         = lxR;
  document.getElementById("loc_y").value         = lyR;
  document.getElementById("shot_distance").value = dist;

  placeDot(csx, csy);
  autoShotType(dist);
}});

function placeDot(sx, sy) {{
  dot.setAttribute("cx", sx);
  dot.setAttribute("cy", sy);
  dot.setAttribute("fill-opacity", "1");

  ripple.setAttribute("cx", sx);
  ripple.setAttribute("cy", sy);
  ripple.setAttribute("r", "6");
  ripple.setAttribute("stroke-opacity", "0.9");

  var r = 6, op = 0.9;
  var id = setInterval(function() {{
    r  += 1.8;
    op -= 0.07;
    if (op <= 0) {{ clearInterval(id); return; }}
    ripple.setAttribute("r", r);
    ripple.setAttribute("stroke-opacity", op);
  }}, 25);
}}

function syncFromFields() {{
  var lx   = parseFloat(document.getElementById("loc_x").value)  || 0;
  var ly   = parseFloat(document.getElementById("loc_y").value)  || 0;
  var dist = +(Math.sqrt(lx * lx + ly * ly) / 10).toFixed(1);
  document.getElementById("shot_distance").value = dist;

  var svg = nbaToSvg(lx, Math.max(0, ly));
  placeDot(
    Math.max(8, Math.min(SVG_VW - 8, svg.sx)),
    Math.max(BASKET_SY, Math.min(SVG_VH - 8, svg.sy))
  );
  autoShotType(dist);
}}

function onDistanceChange() {{
  var d = parseFloat(document.getElementById("shot_distance").value) || 0;
  autoShotType(d);
}}

/* ── Shot type ── */
function selectType(t) {{
  selectedType = t;
  document.getElementById("btn-2pt").classList.toggle("active", t === "2pt");
  document.getElementById("btn-3pt").classList.toggle("active", t === "3pt");
}}

function autoShotType(distFt) {{
  var is3 = distFt >= 22.5;
  document.getElementById("btn-2pt").classList.toggle("hidden", is3);
  if (is3) {{
    selectType("3pt");
  }} else if (selectedType === "3pt") {{
    selectType("2pt");
  }}
}}

/* ── Player search ── */
var searchTimer = null;
var searchInput = document.getElementById("player-search");
var playerDd    = document.getElementById("player-dd");
var retrainRow  = document.getElementById("retrain-row");
var retrainName = document.getElementById("retrain-name");
var retrainBtn  = document.getElementById("retrain-btn");
var retrainStat = document.getElementById("retrain-status");

searchInput.addEventListener("input", function() {{
  clearTimeout(searchTimer);
  var q = searchInput.value.trim();
  if (q.length < 2) {{ closeDd(); return; }}
  searchTimer = setTimeout(function() {{ fetchPlayers(q); }}, 280);
}});

searchInput.addEventListener("blur", function() {{
  setTimeout(closeDd, 180);
}});

function fetchPlayers(q) {{
  fetch("/players/search?q=" + encodeURIComponent(q))
    .then(function(r) {{ return r.json(); }})
    .then(function(list) {{
      if (!list.length) {{ closeDd(); return; }}
      playerDd.innerHTML = "";
      list.forEach(function(p) {{
        var div = document.createElement("div");
        div.className   = "dd-item";
        div.textContent = p.name;
        div.dataset.id  = p.id;
        div.dataset.name = p.name;
        div.addEventListener("mousedown", function() {{
          pickPlayer(p.id, p.name);
        }});
        playerDd.appendChild(div);
      }});
      playerDd.classList.add("open");
    }})
    .catch(closeDd);
}}

function closeDd() {{ playerDd.classList.remove("open"); }}

function pickPlayer(id, name) {{
  selectedPlayer = {{ id: id, name: name }};
  searchInput.value       = name;
  retrainName.textContent = name;
  retrainRow.classList.add("show");
  retrainStat.textContent = "";
  closeDd();
}}

/* ── Retrain ── */
function startRetrain() {{
  if (!selectedPlayer) return;
  retrainBtn.disabled = true;
  retrainStat.innerHTML = "<span class='spinner'></span>Fetching shots for " + selectedPlayer.name + "...";

  fetch("/model/retrain", {{
    method: "POST",
    headers: {{ "Content-Type": "application/json" }},
    body: JSON.stringify({{ player_name: selectedPlayer.name, season: "2023-24" }})
  }})
  .then(function() {{
    retrainPoll = setInterval(pollRetrain, 2500);
  }})
  .catch(function() {{
    retrainStat.textContent = "Failed to start retraining.";
    retrainBtn.disabled = false;
  }});
}}

function pollRetrain() {{
  fetch("/model/retrain/status")
    .then(function(r) {{ return r.json(); }})
    .then(function(d) {{
      if (d.status === "fetching" || d.status === "training") {{
        retrainStat.innerHTML = "<span class='spinner'></span>" + d.message;
      }} else if (d.status === "done") {{
        clearInterval(retrainPoll);
        retrainBtn.disabled = false;
        retrainStat.innerHTML = "&#10003; " + d.message;
        retrainRow.classList.remove("show");
        searchInput.value = "";
        selectedPlayer = null;
        refreshBadge();
      }} else if (d.status === "error") {{
        clearInterval(retrainPoll);
        retrainBtn.disabled = false;
        retrainStat.innerHTML = "&#10007; " + d.message;
      }}
    }})
    .catch(function() {{}});
}}

function refreshBadge() {{
  fetch("/health")
    .then(function(r) {{ return r.json(); }})
    .then(function(d) {{
      document.getElementById("badge-player").textContent = d.player;
      document.getElementById("badge-meta").textContent   =
        d.season + " \u00b7 " + (d.n_shots || "").toLocaleString() + " shots \u00b7 AUC " + d.roc_auc.toFixed(3);
      document.getElementById("pill-player").textContent  = d.player;
      document.getElementById("pill-season").textContent  = d.season;
      document.getElementById("pill-auc").textContent     = d.roc_auc.toFixed(3);
    }})
    .catch(function() {{}});
}}

/* ── Prediction ── */
function doPredikt() {{
  document.getElementById("err-msg").textContent = "";

  var loc_x         = parseFloat(document.getElementById("loc_x").value);
  var loc_y         = parseFloat(document.getElementById("loc_y").value);
  var shot_distance = parseFloat(document.getElementById("shot_distance").value);

  if (isNaN(loc_x) || isNaN(loc_y) || isNaN(shot_distance)) {{
    document.getElementById("err-msg").textContent = "Please set valid coordinates.";
    return;
  }}

  fetch("/predict", {{
    method: "POST",
    headers: {{ "Content-Type": "application/json" }},
    body: JSON.stringify({{ loc_x: loc_x, loc_y: loc_y, shot_distance: shot_distance, shot_type: selectedType }})
  }})
  .then(function(r) {{
    if (!r.ok) {{ return r.json().then(function(e) {{ throw new Error(e.detail || "Server error"); }}); }}
    return r.json();
  }})
  .then(showResult)
  .catch(function(e) {{
    document.getElementById("err-msg").textContent = "Error: " + e.message;
  }});
}}

function resultColor(pct) {{
  if (pct < 0.35) return "#f87171";
  if (pct < 0.50) return "#f0a500";
  return "#4ade80";
}}

function showResult(d) {{
  var rc = document.getElementById("result-card");
  rc.classList.remove("hidden");

  var color  = resultColor(d.xfg_pct);
  var arc    = document.getElementById("gauge-arc");
  var offset = 220 - d.xfg_pct * 220;
  arc.style.strokeDashoffset = offset;
  arc.style.stroke           = color;

  var num = document.getElementById("gauge-num");
  num.textContent  = d.xfg_pct_display;
  num.style.color  = color;

  document.getElementById("r-zone").textContent  = d.shot_zone;
  document.getElementById("r-angle").textContent = d.shot_angle_deg + "\u00b0";
  document.getElementById("r-type").textContent  = d.shot_type;
  document.getElementById("r-ci").textContent    =
    (d.confidence_low  * 100).toFixed(1) + "% \u2013 " +
    (d.confidence_high * 100).toFixed(1) + "%";

  /* move any existing FT data into result card */
  var sp = document.getElementById("ft-popup-standalone");
  if (sp.classList.contains("show")) {{
    document.getElementById("ft-pct-val").textContent = document.getElementById("ft-pct-val2").textContent;
    document.getElementById("ft-sub").textContent     = document.getElementById("ft-sub2").textContent;
    document.getElementById("ft-popup").classList.add("show");
    sp.classList.remove("show");
  }}
}}

/* ── Free throw % ── */
function fetchFTPct() {{
  var playerName = document.getElementById("badge-player").textContent.trim();
  var label2 = document.getElementById("ft-pct-val2");
  var sub2   = document.getElementById("ft-sub2");
  label2.textContent = "...";
  sub2.textContent   = "";

  var sp = document.getElementById("ft-popup-standalone");
  sp.classList.add("show");

  fetch("/player/freethrow?player_name=" + encodeURIComponent(playerName))
    .then(function(r) {{
      if (!r.ok) throw new Error("Not found");
      return r.json();
    }})
    .then(function(d) {{
      label2.textContent = d.ft_pct_display;
      sub2.textContent   = d.ftm + " / " + d.fta + " FTA \u00b7 " + d.season;

      var rc = document.getElementById("result-card");
      if (!rc.classList.contains("hidden")) {{
        document.getElementById("ft-pct-val").textContent = d.ft_pct_display;
        document.getElementById("ft-sub").textContent     = d.ftm + " / " + d.fta + " FTA \u00b7 " + d.season;
        document.getElementById("ft-popup").classList.add("show");
        sp.classList.remove("show");
      }}
    }})
    .catch(function() {{
      label2.textContent = "N/A";
      sub2.textContent   = "Could not load FT data";
    }});
}}
</script>
</body>
</html>"""


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    print("\n  Basketball xFG% API  v3.1")
    print("  ─────────────────────────────────────")
    print("  Browser UI  ->  http://localhost:8000/")
    print("  Swagger UI  ->  http://localhost:8000/docs")
    print("  Health      ->  http://localhost:8000/health")
    print("  ─────────────────────────────────────\n")
    uvicorn.run("phase3_api:app", host="0.0.0.0", port=8000, reload=True)
