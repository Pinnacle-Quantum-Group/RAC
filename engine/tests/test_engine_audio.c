/*
 * test_engine_audio.c — Audio Engine BVT
 * Verifies PCM generation, spatial audio, WAV output.
 */

#include "../rac_engine_audio.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

static int tests_passed = 0;
static int tests_failed = 0;

#define TEST(name) printf("  [Audio] %-50s ", name);
#define PASS() do { printf("PASS\n"); tests_passed++; } while(0)
#define FAIL(msg) do { printf("FAIL: %s\n", msg); tests_failed++; } while(0)
#define CHECK(cond, msg) do { if (!(cond)) { FAIL(msg); return; } } while(0)

static void test_create_destroy(void)
{
    TEST("Audio engine create/destroy");
    rac_audio_engine *audio = rac_audio_create();
    CHECK(audio != NULL, "created");
    CHECK(audio->output_buffer != NULL, "output buffer");
    CHECK(audio->master_volume == 1.0f, "master volume");
    rac_audio_destroy(audio);
    PASS();
}

static void test_gen_sine(void)
{
    TEST("Sine wave generation via rac_rotate");
    rac_audio_engine *audio = rac_audio_create();

    int clip_id = rac_audio_gen_sine(audio, 440.0f, 0.1f, 0.5f);
    CHECK(clip_id >= 0, "clip created");
    CHECK(audio->clips[clip_id].valid, "clip valid");
    CHECK(audio->clips[clip_id].num_samples > 0, "has samples");

    /* Check that sine wave crosses zero */
    int16_t *samples = audio->clips[clip_id].samples;
    int num = audio->clips[clip_id].num_samples;
    int zero_crossings = 0;
    for (int i = 1; i < num; i++) {
        if ((samples[i - 1] >= 0 && samples[i] < 0) ||
            (samples[i - 1] < 0 && samples[i] >= 0))
            zero_crossings++;
    }
    /* 440 Hz for 0.1s = ~88 zero crossings */
    CHECK(zero_crossings > 50, "oscillating sine wave");

    rac_audio_destroy(audio);
    PASS();
}

static void test_spatial_audio(void)
{
    TEST("Spatial audio panning");
    rac_audio_engine *audio = rac_audio_create();

    int clip = rac_audio_gen_sine(audio, 440.0f, 0.5f, 0.5f);
    int src = rac_audio_create_source(audio, clip, rac_phys_v3(5.0f, 0.0f, 0.0f));
    CHECK(src >= 0, "source created");

    /* Listener at origin, facing -Z, source to the right (+X) */
    rac_audio_set_listener(audio, rac_phys_v3_zero(), rac_phys_quat_identity());
    rac_audio_play(audio, src);
    rac_audio_update_spatial(audio);

    /* Right gain should be higher than left */
    CHECK(audio->sources[src].right_gain > audio->sources[src].left_gain,
          "right panning for +X source");

    /* Move source to left */
    audio->sources[src].position = rac_phys_v3(-5.0f, 0.0f, 0.0f);
    rac_audio_update_spatial(audio);
    CHECK(audio->sources[src].left_gain > audio->sources[src].right_gain,
          "left panning for -X source");

    rac_audio_destroy(audio);
    PASS();
}

static void test_mixer(void)
{
    TEST("Audio mixer produces valid PCM");
    rac_audio_engine *audio = rac_audio_create();

    int clip = rac_audio_gen_sine(audio, 440.0f, 0.5f, 0.5f);
    int src = rac_audio_create_source(audio, clip, rac_phys_v3_zero());
    rac_audio_play(audio, src);
    rac_audio_update_spatial(audio);
    rac_audio_mix(audio);

    /* Check output buffer has non-zero samples */
    int nonzero = 0;
    for (int i = 0; i < audio->buffer_size; i++) {
        if (audio->output_buffer[i] != 0) nonzero++;
    }
    CHECK(nonzero > 100, "mixer produced audio");

    rac_audio_destroy(audio);
    PASS();
}

static void test_wav_output(void)
{
    TEST("WAV file output");
    rac_audio_engine *audio = rac_audio_create();

    int clip = rac_audio_gen_sine(audio, 440.0f, 0.1f, 0.5f);
    int src = rac_audio_create_source(audio, clip, rac_phys_v3_zero());
    rac_audio_play(audio, src);
    rac_audio_update_spatial(audio);
    rac_audio_mix(audio);

    int ret = rac_audio_write_wav(audio, "/tmp/rac_test_audio.wav",
                                  audio->output_buffer, RAC_AUDIO_BUFFER_SIZE);
    CHECK(ret == 0, "WAV write success");

    /* Verify RIFF header */
    FILE *f = fopen("/tmp/rac_test_audio.wav", "rb");
    CHECK(f != NULL, "file exists");
    char riff[4] = {0};
    CHECK(fread(riff, 1, 4, f) == 4, "RIFF header readable");
    CHECK(riff[0] == 'R' && riff[1] == 'I' && riff[2] == 'F' && riff[3] == 'F',
          "valid RIFF header");
    fclose(f);

    rac_audio_destroy(audio);
    PASS();
}

static void test_distance_attenuation(void)
{
    TEST("Distance attenuation");
    rac_audio_engine *audio = rac_audio_create();

    int clip = rac_audio_gen_sine(audio, 440.0f, 0.5f, 0.5f);
    int near_src = rac_audio_create_source(audio, clip, rac_phys_v3(1.0f, 0.0f, 0.0f));
    int far_src = rac_audio_create_source(audio, clip, rac_phys_v3(20.0f, 0.0f, 0.0f));
    rac_audio_play(audio, near_src);
    rac_audio_play(audio, far_src);

    rac_audio_set_listener(audio, rac_phys_v3_zero(), rac_phys_quat_identity());
    rac_audio_update_spatial(audio);

    float near_total = audio->sources[near_src].left_gain + audio->sources[near_src].right_gain;
    float far_total = audio->sources[far_src].left_gain + audio->sources[far_src].right_gain;
    CHECK(near_total > far_total, "near source louder than far");

    rac_audio_destroy(audio);
    PASS();
}

int main(void)
{
    printf("\n=== RAC Engine Audio BVT ===\n\n");

    test_create_destroy();
    test_gen_sine();
    test_spatial_audio();
    test_mixer();
    test_wav_output();
    test_distance_attenuation();

    printf("\n  Results: %d passed, %d failed\n\n", tests_passed, tests_failed);
    return tests_failed > 0 ? 1 : 0;
}
