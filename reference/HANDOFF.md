# Algorithm Handoff Document

> For AI agent or developer working on `pebble-swim-tracker` C implementation

## Overview

This document describes the swim turn detection algorithms developed and validated
in the `Google-Pebble-Swim` visualization tool, ready for porting to the Pebble
smartwatch C firmware.

**Target hardware:** Pebble Time / Time Steel (ARM Cortex-M4, 64MHz, 256KB RAM, no FPU)  
**Sensor:** 3-axis accelerometer + magnetometer at 10-25 Hz  
**Goal:** Detect pool wall turns in real-time to count laps and estimate distance

## Files in this directory

| File | Purpose |
|------|---------|
| `turn_detect_ncc.h` | C header — API and configuration |
| `turn_detect_ncc.c` | C implementation — fixed-point NCC template matching |
| `HANDOFF.md` | This document |

## Algorithm Summary (v5 — Combined)

The production algorithm uses TWO complementary detectors:

### Detector 1: Stroke-Interval Gap (best for freestyle)

1. Detect strokes via magnitude peak detection (moving average + threshold)
2. Compute inter-stroke time intervals
3. Flag intervals that are ≥1.3× the local baseline (15th percentile of nearby 4 intervals)
4. Validate: the gap must contain a "glide" (magnitude drops below 88% of session average)
5. Place turn at midpoint of the gap

**Why it works:** During a turn, the swimmer glides to the wall, touches, pushes off, and glides out. This creates a ~3-8s gap in the regular 0.8-1.5s freestyle stroke rhythm.

**Limitation:** Breaststroke has naturally long inter-stroke intervals (2.5-3.0s), making turn gaps indistinguishable from normal strokes.

### Detector 2: Template Matching via NCC (catches breaststroke turns)

1. Pre-compute a "canonical turn" template from annotated data (61 samples × 4 axes)
2. Slide the template across the incoming signal every 1 second
3. Compute Normalized Cross-Correlation (NCC) on each axis independently
4. Average the 4 per-axis NCC scores
5. Detect peaks above threshold 0.25 with minimum 22s between turns

**Why it works:** All turns (freestyle AND breaststroke) share a common accelerometer shape: magnitude dip → rapid Y oscillation → Z dive → quiet glide. The template captures this shape.

**C implementation:** See `turn_detect_ncc.c` — uses Q8 fixed-point, ring buffer, ~300 multiply-accumulates per evaluation.

### Combining

Take the union of both detector outputs. If a template-matched turn is within 22s of an already-detected stroke-interval turn, skip it (no duplicate). Post-filter to remove any turns that would create laps shorter than 20s.

## Template Data

The template was built by:
1. Extracting ±3s windows around each of 21 human-annotated turns
2. Normalizing each window (subtract mean, divide by std dev)
3. Averaging all 21 normalized windows per axis
4. Re-normalizing the averaged template

Source annotations (stored as `.annotations.json` sidecar files):
- `swim-20260716-152708.json`: 11 turns (600m, freestyle + breaststroke)
- `swim-20260716-163913.json`: 4 turns (250m, breaststroke-heavy)
- `swim-20260716-165824.json`: 6 turns (350m, mixed)

## Parameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| Sample rate | 10 Hz | Can work at 20-25Hz with adjusted template length |
| Template half-width | 30 samples (3s) | Total template = 61 samples |
| NCC threshold | 0.25 (Q8: 64) | Higher = fewer false positives, more misses |
| Min turn interval | 22s (220 samples) | Prevents double-detection |
| Min lap duration | 20s (200 samples) | Post-filter for short laps |
| Stroke gap ratio | 1.3× | Primary detector threshold |
| Stroke gap absolute min | 1800ms | Primary detector minimum gap |
| Glide threshold | 0.88× session average | Magnitude must drop this low |

## Performance Results

Tested against 21 human-annotated turns across 3 files (±4s match threshold):

| Metric | Result |
|--------|--------|
| Total annotations | 21 |
| Matched (combined v5) | ~17-18 |
| Missed | ~3-4 |
| False positives | ~3-5 |
| Distance accuracy | Correct for 1/3 files, ±50m for other 2 |

## Integration Notes for Pebble C Code

### Memory budget
- Template storage: 61 × 4 axes × 2 bytes = 488 bytes (flash/const)
- Ring buffer: 64 × 6 bytes = 384 bytes (RAM)
- State struct: ~20 bytes (RAM)
- Working arrays for NCC: 61 × 4 × 2 = 488 bytes (stack, temporary)
- **Total: ~500 bytes flash + ~900 bytes RAM**

### CPU budget (per evaluation, every 1 second)
- Extract from ring: 61 iterations
- Per-axis NCC: 61 multiplies + 61 adds × 4 axes = ~500 operations
- Peak detection: trivial (2 comparisons)
- **Total: ~600 operations per second = negligible on 64MHz ARM**

### Integration with existing stroke detector
The existing `stroke_classify_features()` in the Pebble C code runs per-lap.
The NCC detector should run independently (every 1s) and signal turn events.
The main loop can combine NCC turns with the existing stroke-interval logic.

### Suggested main loop structure:
```c
void accel_handler(int16_t x, int16_t y, int16_t z) {
    // Feed both detectors
    stroke_feed(x, y, z);      // Existing stroke counter
    turn_ncc_feed(&ncc, x, y, z);  // New template detector
    
    // Also run stroke-interval gap check (existing logic)
    if (stroke_gap_detected()) {
        register_turn(TURN_SRC_GAP);
    }
    
    // Template match detected
    if (turn_ncc_detected(&ncc)) {
        // Only register if not too close to a gap-detected turn
        if (time_since_last_turn() > MIN_TURN_INTERVAL) {
            register_turn(TURN_SRC_NCC);
        }
    }
}
```

## Compass Data Notes

The magnetometer data in these recordings is unreliable for turn detection:
- Only ~180 unique heading values (2° resolution)
- Heading changes at turns range from 11° to 173° (inconsistent)
- Pool environment causes magnetic interference
- **Recommendation:** Do NOT rely on compass for primary turn detection. Use as optional confidence booster only (if heading change > 60°, increase confidence in a turn candidate).

## Key Visual Patterns at Turns

From manual annotation analysis:

1. **Pre-turn:** Regular stroke rhythm visible as periodic magnitude peaks
2. **Glide to wall:** Magnitude drops toward ~1000mG (gravity), movement quiets
3. **Turn event:** 2-3 rapid Y-axis oscillations + Z-axis dives negative (hand goes deep)
4. **Push-off glide:** Another quiet phase (~1-2s)
5. **First strokes of new lap:** Rhythm resumes (possibly different rhythm if stroke type changes)

## Files to Modify in pebble-swim-tracker

1. Add `turn_detect_ncc.h` and `turn_detect_ncc.c` to the project
2. In the main accelerometer handler, call `turn_ncc_feed()` on every sample
3. In the turn registration logic, combine NCC detections with existing gap detections
4. Update lap counter to use the combined output

## Future Improvements

1. **Separate freestyle vs breaststroke templates** — build 2 templates from annotated data, use whichever scores higher
2. **Adaptive threshold** — start at 0.25, lower to 0.20 if fewer turns detected than expected given elapsed time and pool length
3. **Compass as tiebreaker** — if NCC score is borderline (0.20-0.25), check if heading changed >60° to confirm
4. **Higher sample rate** — at 20Hz, template would be 121 samples but more precise
