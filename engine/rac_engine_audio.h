/*
 * rac_engine_audio.h — Spatial Audio Engine
 * 3D positional audio, HRTF, DSP pipeline — all via RAC primitives.
 */

#ifndef RAC_ENGINE_AUDIO_H
#define RAC_ENGINE_AUDIO_H

#include "rac_physics.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ── Audio format ──────────────────────────────────────────────────────── */

#define RAC_AUDIO_SAMPLE_RATE   44100
#define RAC_AUDIO_CHANNELS      2
#define RAC_AUDIO_BUFFER_SIZE   2048  /* samples per channel per callback */

/* ── Sound source ──────────────────────────────────────────────────────── */

#define RAC_MAX_AUDIO_SOURCES 32
#define RAC_MAX_AUDIO_SAMPLES (RAC_AUDIO_SAMPLE_RATE * 30)  /* max 30 seconds */

typedef struct {
    int16_t  *samples;       /* mono PCM data */
    int       num_samples;
    int       sample_rate;
    int       valid;
} rac_audio_clip;

typedef struct {
    int             clip_id;
    rac_phys_vec3   position;
    float           volume;
    float           pitch;
    int             looping;
    int             playing;
    float           play_position;   /* fractional sample index */

    /* Spatial audio state */
    float           left_gain;
    float           right_gain;
    float           left_delay;      /* ITD in samples */
    float           right_delay;
} rac_audio_source;

/* ── Audio mixer ───────────────────────────────────────────────────────── */

typedef struct {
    /* Clip library */
    rac_audio_clip    clips[RAC_MAX_AUDIO_SOURCES];
    int               num_clips;

    /* Active sources */
    rac_audio_source  sources[RAC_MAX_AUDIO_SOURCES];
    int               num_sources;

    /* Listener (camera) */
    rac_phys_vec3     listener_pos;
    rac_phys_quat     listener_orient;

    /* Output buffer (interleaved stereo S16LE) */
    int16_t          *output_buffer;
    int               buffer_size;

    /* Global volume */
    float             master_volume;
} rac_audio_engine;

/* ── API ───────────────────────────────────────────────────────────────── */

rac_audio_engine *rac_audio_create(void);
void rac_audio_destroy(rac_audio_engine *audio);

/* Load a WAV file (16-bit PCM mono/stereo) */
int rac_audio_load_wav(rac_audio_engine *audio, const char *path);

/* Generate procedural tones via rac_rotate (sin/cos) */
int rac_audio_gen_sine(rac_audio_engine *audio, float frequency,
                       float duration, float amplitude);
int rac_audio_gen_noise(rac_audio_engine *audio, float duration, float amplitude);

/* Create a sound source */
int rac_audio_create_source(rac_audio_engine *audio, int clip_id,
                            rac_phys_vec3 position);
void rac_audio_play(rac_audio_engine *audio, int source_id);
void rac_audio_stop(rac_audio_engine *audio, int source_id);

/* Update spatial audio: recompute gains/delays from listener pose */
void rac_audio_update_spatial(rac_audio_engine *audio);

/* Mix all active sources into output buffer */
void rac_audio_mix(rac_audio_engine *audio);

/* Write output to WAV file */
int rac_audio_write_wav(const rac_audio_engine *audio, const char *path,
                        const int16_t *samples, int num_samples);

/* Update listener transform */
void rac_audio_set_listener(rac_audio_engine *audio,
                            rac_phys_vec3 pos, rac_phys_quat orient);

/* DSP: frequency analysis via rac_dct */
void rac_audio_dct_analyze(const float *samples, float *spectrum, int n);

/* DSP: envelope generation via rac_exp */
void rac_audio_envelope(float *samples, int n, float attack, float decay,
                        float sustain, float release, int sample_rate);

#ifdef __cplusplus
}
#endif

#endif /* RAC_ENGINE_AUDIO_H */
