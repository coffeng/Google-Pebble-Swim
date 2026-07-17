/**
 * turn_detect_ncc.h - Template-based swim turn detection via Normalized Cross-Correlation
 * 
 * Reference C implementation for Pebble smartwatch (ARM Cortex-M, no FPU).
 * Uses fixed-point Q15 arithmetic for all computations.
 * 
 * Algorithm: Slides a pre-computed turn template (built from 21 annotated turns)
 * across the accelerometer signal and detects correlation peaks.
 * 
 * Memory: ~500 bytes template + ~250 bytes working buffer
 * CPU: ~300 multiply-accumulates per evaluation (every 1s at 10Hz)
 * 
 * Usage:
 *   1. Call turn_ncc_init() once at start of swim session
 *   2. Call turn_ncc_feed() for every new accelerometer sample (at 10Hz)
 *   3. Check turn_ncc_detected() after each feed - returns true if a turn was just detected
 */

#ifndef TURN_DETECT_NCC_H
#define TURN_DETECT_NCC_H

#include <stdint.h>
#include <stdbool.h>

/* Configuration */
#define NCC_SAMPLE_RATE     10      /* Hz */
#define NCC_TEMPLATE_HALF   30      /* samples = 3 seconds each side */
#define NCC_TEMPLATE_LEN    61      /* 2*HALF + 1 */
#define NCC_EVAL_INTERVAL   10      /* Evaluate every 10 samples = 1 second */
#define NCC_THRESHOLD       64      /* Q8 fixed point: 0.25 * 256 = 64 */
#define NCC_MIN_TURN_GAP    220     /* samples = 22 seconds at 10Hz */
#define NCC_MIN_LAP_DUR     200     /* samples = 20 seconds */

/* Ring buffer size must hold at least NCC_TEMPLATE_LEN samples */
#define NCC_RING_SIZE       64      /* Power of 2 >= NCC_TEMPLATE_LEN */
#define NCC_RING_MASK       (NCC_RING_SIZE - 1)

/* Accelerometer sample (raw int16 values in mG) */
typedef struct {
    int16_t x;
    int16_t y;
    int16_t z;
} ncc_sample_t;

/* Turn detection state */
typedef struct {
    /* Ring buffer of recent samples */
    ncc_sample_t ring[NCC_RING_SIZE];
    uint16_t ring_head;         /* Next write position */
    uint32_t sample_count;      /* Total samples fed */
    
    /* Evaluation state */
    int16_t prev_score;         /* Previous NCC score (for peak detection) */
    int16_t prev_prev_score;    /* Score before previous */
    uint32_t last_turn_sample;  /* Sample index of last detected turn */
    
    /* Output */
    bool turn_detected;         /* Set true when a turn is detected this cycle */
    uint32_t turn_count;        /* Total turns detected */
} ncc_state_t;

/**
 * Initialize the NCC turn detector. Call once at swim session start.
 */
void turn_ncc_init(ncc_state_t *state);

/**
 * Feed a new accelerometer sample. Call at NCC_SAMPLE_RATE Hz.
 * After calling, check state->turn_detected.
 */
void turn_ncc_feed(ncc_state_t *state, int16_t x, int16_t y, int16_t z);

/**
 * Check if a turn was detected on the last feed.
 */
static inline bool turn_ncc_detected(const ncc_state_t *state) {
    return state->turn_detected;
}

/**
 * Get total turn count so far.
 */
static inline uint32_t turn_ncc_count(const ncc_state_t *state) {
    return state->turn_count;
}

#endif /* TURN_DETECT_NCC_H */
