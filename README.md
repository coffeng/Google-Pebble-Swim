# Pebble Swim Data Explorer

> Last updated: 2026-07-17

A React app for visualizing Pebble smartwatch swim sensor data (accelerometer + compass). Includes stroke detection, turn detection, and stroke type classification.

View in AI Studio: https://ai.studio/apps/ae12f1c7-8d16-4580-b80a-546832639f6f

## Data Sources

The app supports two data sources, automatically selected based on your environment:

| Mode | When | Data source |
|------|------|-------------|
| **Local** (default on desktop) | Local file server is detected | JSON files from disk |
| **Google Drive** (default in AI Studio) | Local server unavailable | Files from your Google Drive `/pebble/` folder |

You can switch between them using the **Local / Drive** toggle in the header.

## Run Locally (Windows Desktop)

**Prerequisites:** Node.js

1. Install dependencies:
   ```
   npm install
   ```

2. Start both the local file server and the frontend:
   ```
   npm run dev:local
   ```

3. Open **http://localhost:3000/** in your browser.

The local file server reads JSON files from:
```
C:\Users\100014430\Documents\GitHub\coffeng\pebble-swim-tracker\logs\PebbleSwimTracker
```

To use a different folder, set the `PEBBLE_DATA_PATH` environment variable before starting:
```
set PEBBLE_DATA_PATH=C:\path\to\your\json\files
npm run dev:local
```

No Google authentication is required in local mode — the app loads files directly from disk.

## Run in Google AI Studio

The app runs as-is in AI Studio. It uses Firebase + Google OAuth to read JSON files from your Google Drive.

1. Set the `GEMINI_API_KEY` in `.env.local` to your Gemini API key (if using AI features)
2. Run:
   ```
   npm run dev
   ```

## Available Scripts

| Script | Description |
|--------|-------------|
| `npm run dev:local` | Start local file server + Vite dev server (desktop use) |
| `npm run dev` | Start Vite dev server only (AI Studio / Google Drive mode) |
| `npm run server` | Start only the local file server on port 3001 |
| `npm run build` | Production build |
| `npm run preview` | Preview production build |

## Expected JSON Format

The parser handles Pebble swim tracker JSON files with this structure:

```json
{
  "device": "pebble-swim-tracker",
  "created_at": "2026-07-14T09:31:43.998Z",
  "pool_length": 50,
  "raw_hz": 20,
  "raw_axes": 4,
  "raw": [
    [1204, 437, -364, 174],
    [1246, 643, -700, 174]
  ]
}
```

Each entry in `raw` is `[x, y, z, compass_heading]`. The parser also supports object-based formats with named fields (`x`, `y`, `z`, `heading`, etc.).

### v19+ Format (July 2026+)

The `raw` array can contain interleaved event objects for pause/resume:

```json
{
  "raw": [
    [880, 256, -624, 316],
    [1104, 512, -400, 316],
    {"type": "pause", "time": "14:25.520"},
    {"type": "resume", "time": "15:29.992"},
    [-64, -752, -640, 196]
  ]
}
```

These are rendered as wide light-gray vertical bars on the charts.

---
## Annotated Swim Sessions (July 16, 2026)

Ground-truth annotations for 3 valid pool recordings (50m pool). Annotations stored as `.annotations.json` sidecar files, created via GUI playback+annotate tool.

**Primary success metric:** Correct lap count (= turns + 1) → determines distance accuracy.

### Distance Summary

| File | Pool | Annotated Turns | Laps | Distance | v3 Algo Turns | v3 Laps | v3 Distance | Lap Count Error |
|------|------|-----------------|------|----------|---------------|---------|-------------|-----------------|
| swim-20260716-152708.json | 50m | 11 | 12 | 600m | ~13 | ~14 | 700m | +2 over |
| swim-20260716-163913.json | 50m | 4 | 5 | 250m | ~3 | ~4 | 200m | −1 under |
| swim-20260716-165824.json | 50m | 6 | 7 | 350m | ~5 | ~6 | 300m | −1 under |

### Session 1: `swim-20260716-152708.json` (15:27 local)

Duration: 16:39 | 10 Hz | 50m pool | **Ground truth: 600m (12 laps, 11 turns)**

Stroke sequence: 10× freestyle + 2× breaststroke

Annotated turn times: `33s, 119s, 204s, 301s, 383s, 472s, 548s, 627s, 708s, 786s, 889s`

Pause events: `pause@14:25`, `resume@15:30`, `pause@19:01`

### Session 2: `swim-20260716-163913.json` (16:39 local)

Duration: 6:01 | 10 Hz | 50m pool | **Ground truth: 250m (5 laps, 4 turns)**

Stroke sequence: breaststroke, freestyle, breaststroke, freestyle, breaststroke

Annotated turn times: `102s, 179s, 277s, 357s`

Pause events: `pause@6:02`

### Session 3: `swim-20260716-165824.json` (16:58 local)

Duration: 9:03 | 10 Hz | 50m pool | **Ground truth: 350m (7 laps, 6 turns)**

Stroke sequence: freestyle, freestyle, breaststroke, freestyle, freestyle, freestyle, freestyle

Annotated turn times: `91s, 182s, 277s, 353s, 443s, 531s`

Pause events: `pause@9:03`

---

## Algorithm Change History

### v3: Stroke-Interval + Glide + Y/Z Variance Secondary (July 17, 2026 — current)

**Two-stage detection:**

**Stage 1 (Primary):** Stroke-interval gap analysis
1. Detect strokes via magnitude peak detection
2. Compute inter-stroke intervals, exclude pause regions
3. Score each interval = gap / P15(nearby 4 intervals)
4. Turn candidate if: `score ≥ 1.3` AND `gap ≥ 1800ms`
5. Validate: gap contains glide (smoothed magnitude drops below 88% of session average)
6. Post-filter: minimum 22s between turns, minimum 20s lap duration

**Stage 2 (Secondary):** Y+Z variance spike detector (catches breaststroke turns)
1. Compute Y std + Z std in 1.5s sliding windows (0.5s step)
2. Also track Z range (max−min) and magnitude minimum per window
3. Detect spikes where either:
   - Combined Y+Z std > 1.55× local baseline, OR
   - Z range > 1.4× local baseline AND magMin < 85% session average
4. Must be local maximum, ≥22s from nearest existing turn

**Parameters:**

| Parameter | Value | Stage |
|-----------|-------|-------|
| MEDIAN_WINDOW | 4 | Primary |
| MIN_TURN_GAP_RATIO | 1.3 | Primary |
| MIN_TURN_GAP_ABS | 1800ms | Primary |
| MIN_TURN_INTERVAL | 22000ms | Both |
| GLIDE_THRESHOLD | 0.88 | Primary |
| MIN_LAP_DURATION | 20000ms | Post-filter |
| PERCENTILE | 0.15 | Primary |
| SEC_COMBINED_RATIO | 1.55 | Secondary |
| SEC_ZRANGE_RATIO | 1.4 | Secondary |

**Results (±4s match threshold):**

| File | Matched/Total | Missed | Extra | Accuracy |
|------|---------------|--------|-------|----------|
| 152708 (freestyle) | 10/11 | 1 | ~3 | 91% |
| 163913 (breaststroke) | 2/4 | 2 | ~2 | 50% |
| 165824 (mixed) | 4/6 | 2 | ~1 | 67% |
| **Total** | **16/21** | **5** | **~6** | **76%** |

### v2: Stroke-Interval + Glide (July 17, 2026)

Primary stage only. Parameters: RATIO=1.8, ABS=2500ms, GLIDE=0.92, LAP=30s.
Result: 13/21 matched (62%). Breaststroke file: 0/4.

### v1: Compass Angular Difference (July 14, 2026)

Heading change detection (before/after ±14s windows, >135° difference, 35s min interval).
Problems: wrong placement, requires compass, over/under detection.

---

## Algorithm Development Notes (AI Agent Handoff)

Sister repo: `../pebble-swim-tracker/` (embedded C + watch simulator).

### Stroke Detection (unchanged)

1. Moving average (radius=4) on magnitude
2. Threshold: `max(1120, avg + (max-avg) * 0.22)`
3. Peak detection with 700ms minimum separation

### Turn Detection (v3)

See Algorithm Change History above.

### Stroke Type Classification (C classifier)

- `gz_static < 250` → Backstroke
- `ez_ey >= 1.20` → Butterfly
- `freq > 0 && freq <= 0.475` → Breaststroke
- `freq <= 0.52` (if prev=freestyle) → Freestyle
- Default → Freestyle

### Key Findings from Visual Annotation

**What a real turn looks like in the signal:**
1. Last stroke of lap ends → glide phase (magnitude drops toward 1000mG)
2. Turn itself: 2-3 rapid Y oscillations (higher freq than stroke rhythm) + Z dives deep
3. Push-off glide (another quiet phase)
4. First stroke of new lap resumes

**Why breaststroke turns remain hardest:**
- Normal inter-stroke interval: 2.5-3.0s (vs freestyle 0.8-1.5s)
- Turn gap: 3.5-5.0s → only 1.3-1.8× normal rhythm (below reliable detection)
- Must rely on Y/Z axis pattern changes rather than timing gaps alone
- The Y-oscillation burst and Z-dive ARE visible but subtle at 10Hz

**Remaining missed turns (5 of 21):**
- File 1 @889s: Last breaststroke turn before pause. Low contrast against irregullar end-of-swim activity.
- File 2 @102s: First breaststroke turn. Swimmer still settling into rhythm — no clear baseline yet.
- File 2 @357s: Last turn before stopping. Post-turn activity drops to zero (rest), breaking pattern assumptions.
- File 3 @182s: Freestyle→breaststroke transition. Z-range drops after turn (breaststroke is quieter).
- File 3 @531s: Last freestyle turn. Similar to file 2 @357s — activity drops after.

**False positive pattern:**
- Mid-freestyle peaks with high Y-crossing rate but normal Z range
- Stroke-interval gaps in breaststroke that meet 1.3× but aren't wall touches
- Post-pause/cooldown irregularities

### Screenshots (referenced in annotation sessions)

1. Breaststroke turn (5s view): 3 fast Y oscillations at wall, Z dives deep, then quiet glide
2. Breaststroke turn (30s view): Regular slow pattern → abrupt burst → quiet → resume
3. Freestyle→breaststroke transition: Fast strokes → glide → slow wide-spaced strokes
