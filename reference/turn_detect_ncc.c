/**
 * turn_detect_ncc.c - Template-based swim turn detection via NCC
 * 
 * Fixed-point implementation suitable for ARM Cortex-M (Pebble watch).
 * All arithmetic uses int16/int32 — no floating point.
 * 
 * The template was built by averaging ±3s windows around 21 human-annotated
 * turns from 3 swim sessions (freestyle + breaststroke, 50m pool, 10Hz).
 * Values are stored in Q8 fixed-point (multiply by 256).
 */

#include "turn_detect_ncc.h"
#include <string.h>

/* ============================================================
 * TURN TEMPLATE (Q8 fixed-point, built from 21 annotated turns)
 * 
 * These are the normalized template values * 256.
 * Template covers ±3 seconds = 61 samples at 10Hz.
 * 4 axes: X, Y, Z, Magnitude
 * ============================================================ */

/* X-axis template (Q8) */
static const int16_t TMPL_X[NCC_TEMPLATE_LEN] = {
    323, 307, 217, 148, 226, 273, 260, 232, 258, 214,
    128, 108, 38, 118, 121, 81, 59, -11, 13, 48,
    -5, -119, -149, -196, -198, -166, -257, -221, -310, -345,
    -375, -396, -413, -387, -352, -361, -369, -338, -178, -56,
    10, 93, 85, 114, 156, 141, 208, 122, 183, 146,
    168, 172, 204, 206, 237, 234, 246, 243, 192, 237, 197
};

/* Y-axis template (Q8) */
static const int16_t TMPL_Y[NCC_TEMPLATE_LEN] = {
    460, -61, -144, 212, 550, 456, 578, 163, -45, -327,
    -334, 0, 435, 374, -96, -140, -79, 185, 191, 57,
    -46, -93, -56, -139, 54, 88, 31, -94, -198, -102,
    38, 91, -57, -120, -129, 59, 180, 84, -145, -137,
    77, 0, 19, -87, -87, -42, 133, -52, -65, -131,
    -91, -119, -52, -83, -76, -131, -95, -170, -231, -98, -176
};

/* Z-axis template (Q8) */
static const int16_t TMPL_Z[NCC_TEMPLATE_LEN] = {
    3, 51, 70, -67, 92, 161, 87, 41, 114, 317,
    327, 151, 26, 234, 272, 191, 255, 289, 200, 14,
    -82, -117, -140, -115, -179, -147, -174, -102, -260, -322,
    -253, -289, -271, -143, -129, -126, -151, -173, -88, 26,
    80, 88, 91, -17, -62, -7, 39, 155, 115, 79,
    79, 102, 21, 54, 154, 162, 122, 108, -13, -37, 36
};

/* Magnitude template (Q8) */
static const int16_t TMPL_MAG[NCC_TEMPLATE_LEN] = {
    698, 719, 312, 283, 423, 274, 459, 322, 245, -52,
    -136, -64, 177, 149, -11, -23, 19, 163, 99, -108,
    -93, -122, -80, -119, -62, -34, -87, -75, -191, -259,
    -207, -225, -241, -164, -127, -102, -122, -154, -42, 80,
    80, 106, 66, -7, 2, -8, 96, 18, 55, 27,
    38, 25, 42, 52, 95, 103, 70, 53, -13, 58, 19
};

/* ============================================================
 * IMPLEMENTATION
 * ============================================================ */

void turn_ncc_init(ncc_state_t *state) {
    memset(state, 0, sizeof(ncc_state_t));
    state->last_turn_sample = 0;
    state->prev_score = 0;
    state->prev_prev_score = 0;
}

/**
 * Compute NCC between ring buffer contents and a template axis.
 * Returns Q8 fixed-point correlation value (-256 to +256).
 * 
 * Simplified NCC: we skip the signal normalization step and just
 * compute normalized dot product. The template is already normalized.
 * This is faster and works well enough for detection threshold comparison.
 */
static int16_t compute_axis_ncc(const int16_t *signal, const int16_t *tmpl, int len) {
    /* Compute signal mean */
    int32_t sig_sum = 0;
    for (int i = 0; i < len; i++) {
        sig_sum += signal[i];
    }
    int16_t sig_mean = (int16_t)(sig_sum / len);
    
    /* Compute signal std dev (approximate, using mean absolute deviation * 1.25) */
    int32_t sig_mad = 0;
    for (int i = 0; i < len; i++) {
        int16_t diff = signal[i] - sig_mean;
        sig_mad += (diff > 0) ? diff : -diff;
    }
    /* sig_std ≈ MAD * 1.25 ≈ MAD + MAD/4 */
    int32_t sig_std = (sig_mad / len) + (sig_mad / (len * 4));
    if (sig_std < 1) sig_std = 1;
    
    /* Compute cross-correlation (signal normalized by std, template already normalized) */
    int32_t dot = 0;
    for (int i = 0; i < len; i++) {
        int32_t sig_norm = ((int32_t)(signal[i] - sig_mean) * 256) / sig_std;
        dot += (sig_norm * tmpl[i]) >> 8; /* Q8 * Q8 -> Q8 */
    }
    
    /* Normalize by length */
    return (int16_t)(dot / len);
}

void turn_ncc_feed(ncc_state_t *state, int16_t x, int16_t y, int16_t z) {
    state->turn_detected = false;
    
    /* Store sample in ring buffer */
    state->ring[state->ring_head].x = x;
    state->ring[state->ring_head].y = y;
    state->ring[state->ring_head].z = z;
    state->ring_head = (state->ring_head + 1) & NCC_RING_MASK;
    state->sample_count++;
    
    /* Only evaluate every NCC_EVAL_INTERVAL samples (1 second) */
    if (state->sample_count < NCC_TEMPLATE_LEN) return;  /* Need full buffer first */
    if ((state->sample_count % NCC_EVAL_INTERVAL) != 0) return;
    
    /* Extract signal from ring buffer into linear arrays */
    int16_t sig_x[NCC_TEMPLATE_LEN];
    int16_t sig_y[NCC_TEMPLATE_LEN];
    int16_t sig_z[NCC_TEMPLATE_LEN];
    int16_t sig_mag[NCC_TEMPLATE_LEN];
    
    uint16_t read_pos = (state->ring_head - NCC_TEMPLATE_LEN) & NCC_RING_MASK;
    for (int i = 0; i < NCC_TEMPLATE_LEN; i++) {
        uint16_t idx = (read_pos + i) & NCC_RING_MASK;
        sig_x[i] = state->ring[idx].x;
        sig_y[i] = state->ring[idx].y;
        sig_z[i] = state->ring[idx].z;
        /* Approximate magnitude: |x| + |y| + |z| (Manhattan norm, faster than sqrt) */
        int16_t ax = (state->ring[idx].x > 0) ? state->ring[idx].x : -state->ring[idx].x;
        int16_t ay = (state->ring[idx].y > 0) ? state->ring[idx].y : -state->ring[idx].y;
        int16_t az = (state->ring[idx].z > 0) ? state->ring[idx].z : -state->ring[idx].z;
        sig_mag[i] = ax + ay + az;
    }
    
    /* Compute per-axis NCC */
    int16_t ncc_x = compute_axis_ncc(sig_x, TMPL_X, NCC_TEMPLATE_LEN);
    int16_t ncc_y = compute_axis_ncc(sig_y, TMPL_Y, NCC_TEMPLATE_LEN);
    int16_t ncc_z = compute_axis_ncc(sig_z, TMPL_Z, NCC_TEMPLATE_LEN);
    int16_t ncc_mag = compute_axis_ncc(sig_mag, TMPL_MAG, NCC_TEMPLATE_LEN);
    
    /* Combined score = average of 4 axes (Q8) */
    int16_t score = (ncc_x + ncc_y + ncc_z + ncc_mag) / 4;
    
    /* Peak detection: current > previous AND previous > before that AND above threshold */
    if (state->prev_score > state->prev_prev_score &&
        state->prev_score > score &&
        state->prev_score > NCC_THRESHOLD) {
        
        /* Check minimum gap since last turn */
        uint32_t gap = state->sample_count - NCC_EVAL_INTERVAL - state->last_turn_sample;
        if (gap >= NCC_MIN_TURN_GAP) {
            /* Turn detected! (at the previous evaluation point) */
            state->turn_detected = true;
            state->turn_count++;
            state->last_turn_sample = state->sample_count - NCC_EVAL_INTERVAL;
        }
    }
    
    /* Shift scores for peak detection */
    state->prev_prev_score = state->prev_score;
    state->prev_score = score;
}
