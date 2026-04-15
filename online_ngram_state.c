#include <stdint.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
  uint64_t key;
  uint16_t next_tok;
  uint32_t count;
  uint8_t used;
} Slot;

typedef struct {
  int order;
  uint32_t capacity;
  Slot *slots;
} OnlineNgramState;

static uint64_t mix64(uint64_t x) {
  x ^= x >> 33;
  x *= 0xff51afd7ed558ccdULL;
  x ^= x >> 33;
  x *= 0xc4ceb9fe1a85ec53ULL;
  x ^= x >> 33;
  return x;
}

static uint64_t ctx_key(const uint16_t *ctx, int n) {
  uint64_t h = 1469598103934665603ULL;
  for (int i = 0; i < n; ++i) {
    h ^= (uint64_t)ctx[i] + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
    h = mix64(h);
  }
  return h;
}

OnlineNgramState *online_ngram_create(int order, uint32_t capacity) {
  OnlineNgramState *st = (OnlineNgramState *)calloc(1, sizeof(OnlineNgramState));
  if (!st) return NULL;
  st->order = order;
  st->capacity = capacity;
  st->slots = (Slot *)calloc(capacity, sizeof(Slot));
  if (!st->slots) {
    free(st);
    return NULL;
  }
  return st;
}

void online_ngram_free(OnlineNgramState *st) {
  if (!st) return;
  free(st->slots);
  free(st);
}

void online_ngram_update(OnlineNgramState *st, uint16_t prev_tok, uint16_t tok) {
  if (!st) return;
  uint64_t key = mix64((uint64_t)prev_tok);
  uint32_t idx = (uint32_t)(key % st->capacity);
  for (uint32_t probe = 0; probe < st->capacity; ++probe) {
    Slot *slot = &st->slots[(idx + probe) % st->capacity];
    if (!slot->used) {
      slot->used = 1;
      slot->key = key;
      slot->next_tok = tok;
      slot->count = 1;
      return;
    }
    if (slot->key == key && slot->next_tok == tok) {
      slot->count += 1;
      return;
    }
  }
}

int online_ngram_best_hint(
    OnlineNgramState *st,
    const uint16_t *ctx,
    int ctx_len,
    int min_order,
    int max_order,
    uint16_t *out_tok,
    float *out_score
) {
  if (!st || ctx_len <= 0) return 0;
  if (max_order > st->order) max_order = st->order;

  uint32_t best_count = 0;
  uint16_t best_tok = 0;
  for (int order = max_order; order >= min_order; --order) {
    if (ctx_len < order - 1) continue;
    uint64_t key = ctx_key(ctx + (ctx_len - (order - 1)), order - 1);
    uint32_t idx = (uint32_t)(key % st->capacity);
    for (uint32_t probe = 0; probe < st->capacity; ++probe) {
      Slot *slot = &st->slots[(idx + probe) % st->capacity];
      if (!slot->used) break;
      if (slot->key == key && slot->count > best_count) {
        best_count = slot->count;
        best_tok = slot->next_tok;
      }
    }
    if (best_count > 0) break;
  }

  if (best_count == 0) return 0;
  *out_tok = best_tok;
  *out_score = (float)best_count / (float)(best_count + 4U);
  return 1;
}
