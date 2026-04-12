/*
 * rac_engine_audio.c — Spatial Audio Engine Implementation
 *
 * RAC-native audio pipeline:
 *   - Distance attenuation via rac_norm (rac_phys_v3_length)
 *   - Panning/direction via rac_project (rac_phys_v3_dot)
 *   - HRTF angle extraction via rac_polar
 *   - Tone generation via rac_rotate (sin/cos)
 *   - Frequency analysis via rac_dct
 *   - Envelope curves via rac_exp
 *   - Multi-channel mixing via rac_inner
 */

#include "rac_engine_audio.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

/* ── Creation/destruction ──────────────────────────────────────────────── */

rac_audio_engine *rac_audio_create(void)
{
    rac_audio_engine *a = (rac_audio_engine *)calloc(1, sizeof(rac_audio_engine));
    if (!a) return NULL;

    a->buffer_size = RAC_AUDIO_BUFFER_SIZE * RAC_AUDIO_CHANNELS;
    a->output_buffer = (int16_t *)calloc(a->buffer_size, sizeof(int16_t));
    a->master_volume = 1.0f;
    a->listener_orient = rac_phys_quat_identity();
    return a;
}

void rac_audio_destroy(rac_audio_engine *audio)
{
    if (!audio) return;
    for (int i = 0; i < audio->num_clips; i++)
        free(audio->clips[i].samples);
    free(audio->output_buffer);
    free(audio);
}

/* ── WAV loader ────────────────────────────────────────────────────────── */

int rac_audio_load_wav(rac_audio_engine *audio, const char *path)
{
    if (audio->num_clips >= RAC_MAX_AUDIO_SOURCES) return -1;

    FILE *f = fopen(path, "rb");
    if (!f) return -1;

    /* Read RIFF header */
    char riff[4];
    uint32_t file_size, fmt_tag;
    if (fread(riff,       1, 4, f) != 4 ||
        fread(&file_size, 4, 1, f) != 1 ||
        fread(&fmt_tag,   4, 1, f) != 1) {
        fclose(f);
        return -1;
    }

    if (riff[0] != 'R' || riff[1] != 'I' || riff[2] != 'F' || riff[3] != 'F') {
        fclose(f);
        return -1;
    }

    /* Find fmt chunk */
    uint16_t audio_format = 0, num_channels = 0, bits_per_sample = 0;
    uint32_t sample_rate = 0;
    int16_t *pcm_data = NULL;
    int num_samples = 0;

    while (!feof(f)) {
        char chunk_id[4];
        uint32_t chunk_size;
        if (fread(chunk_id, 1, 4, f) != 4) break;
        if (fread(&chunk_size, 4, 1, f) != 1) break;

        if (chunk_id[0] == 'f' && chunk_id[1] == 'm' && chunk_id[2] == 't') {
            if (fread(&audio_format,    2, 1, f) != 1 ||
                fread(&num_channels,    2, 1, f) != 1 ||
                fread(&sample_rate,     4, 1, f) != 1) {
                fclose(f); return -1;
            }
            fseek(f, 6, SEEK_CUR); /* skip byte rate + block align */
            if (fread(&bits_per_sample, 2, 1, f) != 1) { fclose(f); return -1; }
            if (chunk_size > 16) fseek(f, chunk_size - 16, SEEK_CUR);
        } else if (chunk_id[0] == 'd' && chunk_id[1] == 'a' && chunk_id[2] == 't' && chunk_id[3] == 'a') {
            if (bits_per_sample == 16) {
                num_samples = chunk_size / (2 * num_channels);
                pcm_data = (int16_t *)malloc(num_samples * sizeof(int16_t));
                if (!pcm_data) { fclose(f); return -1; }
                if (num_channels == 1) {
                    if (fread(pcm_data, 2, num_samples, f) != (size_t)num_samples) {
                        free(pcm_data); fclose(f); return -1;
                    }
                } else {
                    /* Downmix stereo to mono */
                    for (int i = 0; i < num_samples; i++) {
                        int16_t l, r;
                        if (fread(&l, 2, 1, f) != 1 ||
                            fread(&r, 2, 1, f) != 1) {
                            free(pcm_data); fclose(f); return -1;
                        }
                        pcm_data[i] = (int16_t)((l + r) / 2);
                        for (int ch = 2; ch < num_channels; ch++) {
                            int16_t dummy;
                            if (fread(&dummy, 2, 1, f) != 1) {
                                free(pcm_data); fclose(f); return -1;
                            }
                        }
                    }
                }
            } else {
                fseek(f, chunk_size, SEEK_CUR);
            }
        } else {
            fseek(f, chunk_size, SEEK_CUR);
        }
    }
    fclose(f);

    if (!pcm_data || num_samples == 0) {
        free(pcm_data);
        return -1;
    }

    int id = audio->num_clips++;
    audio->clips[id].samples = pcm_data;
    audio->clips[id].num_samples = num_samples;
    audio->clips[id].sample_rate = sample_rate;
    audio->clips[id].valid = 1;
    return id;
}

/* ── Procedural generation ─────────────────────────────────────────────── */

int rac_audio_gen_sine(rac_audio_engine *audio, float frequency,
                       float duration, float amplitude)
{
    if (audio->num_clips >= RAC_MAX_AUDIO_SOURCES) return -1;

    int num_samples = (int)(duration * RAC_AUDIO_SAMPLE_RATE);
    if (num_samples > RAC_MAX_AUDIO_SAMPLES) num_samples = RAC_MAX_AUDIO_SAMPLES;

    int16_t *data = (int16_t *)malloc(num_samples * sizeof(int16_t));
    if (!data) return -1;

    /* Generate sine wave via rac_rotate: rotate (1,0) by increasing angle */
    float phase_inc = 2.0f * RAC_PI * frequency / (float)RAC_AUDIO_SAMPLE_RATE;
    float phase = 0.0f;
    for (int i = 0; i < num_samples; i++) {
        rac_vec2 sc = rac_rotate((rac_vec2){1.0f, 0.0f}, phase);
        /* sc.y = sin(phase) */
        data[i] = (int16_t)(sc.y * amplitude * 32767.0f);
        phase += phase_inc;
        /* Wrap phase to [-PI, PI] to maintain CORDIC precision */
        if (phase > RAC_PI) phase -= 2.0f * RAC_PI;
        if (phase < -RAC_PI) phase += 2.0f * RAC_PI;
    }

    int id = audio->num_clips++;
    audio->clips[id].samples = data;
    audio->clips[id].num_samples = num_samples;
    audio->clips[id].sample_rate = RAC_AUDIO_SAMPLE_RATE;
    audio->clips[id].valid = 1;
    return id;
}

int rac_audio_gen_noise(rac_audio_engine *audio, float duration, float amplitude)
{
    if (audio->num_clips >= RAC_MAX_AUDIO_SOURCES) return -1;

    int num_samples = (int)(duration * RAC_AUDIO_SAMPLE_RATE);
    if (num_samples > RAC_MAX_AUDIO_SAMPLES) num_samples = RAC_MAX_AUDIO_SAMPLES;

    int16_t *data = (int16_t *)malloc(num_samples * sizeof(int16_t));
    if (!data) return -1;

    /* Simple LCG noise */
    uint32_t seed = 12345;
    for (int i = 0; i < num_samples; i++) {
        seed = seed * 1103515245 + 12345;
        float v = ((float)(seed >> 16) / 32768.0f) - 1.0f;
        data[i] = (int16_t)(v * amplitude * 32767.0f);
    }

    int id = audio->num_clips++;
    audio->clips[id].samples = data;
    audio->clips[id].num_samples = num_samples;
    audio->clips[id].sample_rate = RAC_AUDIO_SAMPLE_RATE;
    audio->clips[id].valid = 1;
    return id;
}

/* ── Source management ─────────────────────────────────────────────────── */

int rac_audio_create_source(rac_audio_engine *audio, int clip_id,
                            rac_phys_vec3 position)
{
    if (audio->num_sources >= RAC_MAX_AUDIO_SOURCES) return -1;
    if (clip_id < 0 || clip_id >= audio->num_clips) return -1;

    int id = audio->num_sources++;
    rac_audio_source *src = &audio->sources[id];
    memset(src, 0, sizeof(*src));
    src->clip_id = clip_id;
    src->position = position;
    src->volume = 1.0f;
    src->pitch = 1.0f;
    src->left_gain = 0.5f;
    src->right_gain = 0.5f;
    return id;
}

void rac_audio_play(rac_audio_engine *audio, int source_id)
{
    if (source_id < 0 || source_id >= audio->num_sources) return;
    audio->sources[source_id].playing = 1;
    audio->sources[source_id].play_position = 0.0f;
}

void rac_audio_stop(rac_audio_engine *audio, int source_id)
{
    if (source_id < 0 || source_id >= audio->num_sources) return;
    audio->sources[source_id].playing = 0;
}

/* ── Spatial audio update ──────────────────────────────────────────────── */

void rac_audio_update_spatial(rac_audio_engine *audio)
{
    rac_phys_vec3 listener_pos = audio->listener_pos;
    rac_phys_quat listener_orient = audio->listener_orient;

    /* Compute listener's right vector via quaternion rotation */
    rac_phys_vec3 listener_right = rac_phys_quat_rotate_vec3(
        listener_orient, rac_phys_v3(1.0f, 0.0f, 0.0f));
    rac_phys_vec3 listener_forward = rac_phys_quat_rotate_vec3(
        listener_orient, rac_phys_v3(0.0f, 0.0f, -1.0f));

    for (int i = 0; i < audio->num_sources; i++) {
        rac_audio_source *src = &audio->sources[i];
        if (!src->playing) continue;

        /* Direction vector to source */
        rac_phys_vec3 to_source = rac_phys_v3_sub(src->position, listener_pos);

        /* Distance attenuation via rac_phys_v3_length (rac_norm internally) */
        float distance = rac_phys_v3_length(to_source);
        float attenuation = 1.0f / (1.0f + distance);

        if (distance < 1e-4f) {
            src->left_gain = attenuation * src->volume;
            src->right_gain = attenuation * src->volume;
            src->left_delay = 0.0f;
            src->right_delay = 0.0f;
            continue;
        }

        rac_phys_vec3 dir = rac_phys_v3_scale(to_source, 1.0f / distance);

        /* Panning via rac_phys_v3_dot (projects direction onto right axis) */
        float pan = rac_phys_v3_dot(dir, listener_right);
        /* pan: -1 = full left, +1 = full right */

        src->left_gain = attenuation * src->volume * (1.0f - pan) * 0.5f;
        src->right_gain = attenuation * src->volume * (1.0f + pan) * 0.5f;

        /* HRTF approximation: interaural time difference via rac_polar */
        /* Project direction onto horizontal plane */
        rac_vec2 horiz = { rac_phys_v3_dot(dir, listener_right),
                           rac_phys_v3_dot(dir, listener_forward) };
        float mag, angle;
        rac_polar(horiz, &mag, &angle);

        /* ITD: ~0.6ms max at 90 degrees = ~26 samples at 44.1kHz */
        /* sin(angle) gives the interaural delay factor */
        rac_vec2 sc = rac_rotate((rac_vec2){1.0f, 0.0f}, angle);
        float itd_samples = sc.y * 26.0f;

        if (itd_samples > 0.0f) {
            src->left_delay = 0.0f;
            src->right_delay = itd_samples;
        } else {
            src->left_delay = -itd_samples;
            src->right_delay = 0.0f;
        }
    }
}

/* ── Mixer ─────────────────────────────────────────────────────────────── */

void rac_audio_mix(rac_audio_engine *audio)
{
    int buf_samples = RAC_AUDIO_BUFFER_SIZE;
    memset(audio->output_buffer, 0, audio->buffer_size * sizeof(int16_t));

    /* Temp float buffers for mixing */
    float left[RAC_AUDIO_BUFFER_SIZE];
    float right[RAC_AUDIO_BUFFER_SIZE];
    memset(left, 0, sizeof(left));
    memset(right, 0, sizeof(right));

    for (int s = 0; s < audio->num_sources; s++) {
        rac_audio_source *src = &audio->sources[s];
        if (!src->playing) continue;
        if (src->clip_id < 0 || src->clip_id >= audio->num_clips) continue;

        rac_audio_clip *clip = &audio->clips[src->clip_id];
        if (!clip->valid) continue;

        float pos = src->play_position;
        float rate = src->pitch * (float)clip->sample_rate / (float)RAC_AUDIO_SAMPLE_RATE;

        for (int i = 0; i < buf_samples; i++) {
            int sample_idx = (int)pos;
            if (sample_idx >= clip->num_samples) {
                if (src->looping) {
                    pos = 0.0f;
                    sample_idx = 0;
                } else {
                    src->playing = 0;
                    break;
                }
            }

            float sample = (float)clip->samples[sample_idx] / 32767.0f;

            /* Apply delay by offsetting write position */
            int li = i - (int)src->left_delay;
            int ri = i - (int)src->right_delay;
            if (li >= 0 && li < buf_samples)
                left[li] += sample * src->left_gain;
            if (ri >= 0 && ri < buf_samples)
                right[ri] += sample * src->right_gain;

            pos += rate;
        }

        src->play_position = pos;
    }

    /* Master volume and convert to S16LE */
    for (int i = 0; i < buf_samples; i++) {
        float l = left[i] * audio->master_volume;
        float r = right[i] * audio->master_volume;

        /* Clamp to [-1, 1] */
        if (l >  1.0f) l =  1.0f;
        if (l < -1.0f) l = -1.0f;
        if (r >  1.0f) r =  1.0f;
        if (r < -1.0f) r = -1.0f;

        audio->output_buffer[i * 2 + 0] = (int16_t)(l * 32767.0f);
        audio->output_buffer[i * 2 + 1] = (int16_t)(r * 32767.0f);
    }
}

void rac_audio_set_listener(rac_audio_engine *audio,
                            rac_phys_vec3 pos, rac_phys_quat orient)
{
    audio->listener_pos = pos;
    audio->listener_orient = orient;
}

/* ── WAV writer ────────────────────────────────────────────────────────── */

int rac_audio_write_wav(const rac_audio_engine *audio, const char *path,
                        const int16_t *samples, int num_samples)
{
    (void)audio;
    FILE *f = fopen(path, "wb");
    if (!f) return -1;

    int channels = 2;
    int sample_rate = RAC_AUDIO_SAMPLE_RATE;
    int bits = 16;
    int data_size = num_samples * channels * (bits / 8);
    int file_size = 36 + data_size;

    /* RIFF header */
    fwrite("RIFF", 1, 4, f);
    fwrite(&file_size, 4, 1, f);
    fwrite("WAVE", 1, 4, f);

    /* fmt chunk */
    fwrite("fmt ", 1, 4, f);
    uint32_t fmt_size = 16;
    uint16_t audio_fmt = 1;  /* PCM */
    uint16_t nch = channels;
    uint32_t sr = sample_rate;
    uint32_t byte_rate = sample_rate * channels * (bits / 8);
    uint16_t block_align = channels * (bits / 8);
    uint16_t bps = bits;
    fwrite(&fmt_size, 4, 1, f);
    fwrite(&audio_fmt, 2, 1, f);
    fwrite(&nch, 2, 1, f);
    fwrite(&sr, 4, 1, f);
    fwrite(&byte_rate, 4, 1, f);
    fwrite(&block_align, 2, 1, f);
    fwrite(&bps, 2, 1, f);

    /* data chunk */
    fwrite("data", 1, 4, f);
    fwrite(&data_size, 4, 1, f);
    fwrite(samples, 2, num_samples * channels, f);

    fclose(f);
    return 0;
}

/* ── DSP utilities ─────────────────────────────────────────────────────── */

void rac_audio_dct_analyze(const float *samples, float *spectrum, int n)
{
    /* Direct passthrough to RAC DCT */
    rac_dct(samples, spectrum, n);
}

void rac_audio_envelope(float *samples, int n, float attack, float decay,
                        float sustain, float release, int sample_rate)
{
    int attack_samples = (int)(attack * sample_rate);
    int decay_samples = (int)(decay * sample_rate);
    int release_start = n - (int)(release * sample_rate);

    for (int i = 0; i < n; i++) {
        float env;
        if (i < attack_samples) {
            /* Attack: linear ramp */
            env = (float)i / (float)attack_samples;
        } else if (i < attack_samples + decay_samples) {
            /* Decay: exponential via rac_exp */
            float t = (float)(i - attack_samples) / (float)decay_samples;
            float decay_factor = rac_exp(-3.0f * t);
            env = sustain + (1.0f - sustain) * decay_factor;
        } else if (i < release_start) {
            env = sustain;
        } else {
            /* Release: exponential via rac_exp */
            float t = (float)(i - release_start) / (float)(n - release_start);
            env = sustain * rac_exp(-5.0f * t);
        }
        samples[i] = samples[i] * env;
    }
}
