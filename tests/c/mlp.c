/* The topple.policy/1 forward pass in float32 (docs/design/phase-2.md §3.2), checked against a
 * .bin v1 parity fixture (§3.3). policy.h comes from tests/c/genc.py. Usage: mlp FIXTURE TOL.
 * Prints "<policy sha256> <rows> <max |a_c - a_ref|>"; exit 0 iff every row is within TOL,
 * 1 if one is not, 2 for a fixture that is not .bin v1 for this policy (little-endian host). */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "policy.h"

static float clampf(float x, float lo, float hi) { return x < lo ? lo : (x > hi ? hi : x); }

static void dense(const float *w, const float *b, const float *in, int n_in, float *out,
                  int n_out, int hidden) { /* Gemm(transB=1), then the hidden activation */
  for (int i = 0; i < n_out; i++) {
    float s = 0.0f;
    for (int j = 0; j < n_in; j++) s += w[i * n_in + j] * in[j];
    s += b[i];
    out[i] = !hidden ? s : (POLICY_SAC ? (s > 0.0f ? s : 0.0f) : tanhf(s));
  }
}

static int refuse(const char *why) { fprintf(stderr, "mlp: %s\n", why); return 2; }

int main(int argc, char **argv) {
  char magic[4];
  uint32_t head[4]; /* version, N, D, A */
  FILE *f = argc == 3 ? fopen(argv[1], "rb") : NULL;
  if (!f || fread(magic, 1, 4, f) != 4 || fread(head, 4, 4, f) != 4) return refuse("truncated");
  if (memcmp(magic, "TPAR", 4) != 0) return refuse("bad magic");
  if (head[0] != 1) return refuse("unknown version");
  if (head[2] != POLICY_D || head[3] != POLICY_A) return refuse("D and A differ from policy.h");
  size_t n = head[1];
  float *obs = malloc(n * POLICY_D * sizeof(float)), *ref = malloc(n * POLICY_A * sizeof(float));
  if (!obs || !ref || fread(obs, sizeof(float), n * POLICY_D, f) != n * POLICY_D ||
      fread(ref, sizeof(float), n * POLICY_A, f) != n * POLICY_A) return refuse("truncated");
  double worst = 0.0;
  for (size_t r = 0; r < n; r++) {
    float z[POLICY_D], h1[POLICY_H1], h2[POLICY_H2], m[POLICY_A];
    for (int j = 0; j < POLICY_D; j++)
      z[j] = clampf((obs[r * POLICY_D + j] - MEAN[j]) * INV_STD[j], OBS_LO, OBS_HI);
    dense(W0, B0, z, POLICY_D, h1, POLICY_H1, 1);
    dense(W1, B1, h1, POLICY_H1, h2, POLICY_H2, 1);
    dense(W2, B2, h2, POLICY_H2, m, POLICY_A, 0);
    for (int k = 0; k < POLICY_A; k++) {
      float a = clampf(POLICY_SAC ? tanhf(m[k]) : m[k], ACT_LO, ACT_HI);
      double d = fabs((double)a - (double)ref[r * POLICY_A + k]);
      if (isnan(d) || d > worst) worst = d; /* a NaN sticks: nothing is > NaN */
    }
  }
  printf("%s %zu %.9g\n", POLICY_SHA256, n, worst);
  free(obs);
  free(ref);
  fclose(f);
  return worst <= atof(argv[2]) ? 0 : 1;
}
