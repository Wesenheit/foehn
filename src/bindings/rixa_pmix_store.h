#pragma once
#include <pmix.h>
#include <pmix_common.h>
#include <stdint.h>

typedef enum {
  RIXA_SUCCESS,
  RIXA_COMMIT_ERROR,
  RIXA_PUBLISH_ERROR,
  RIXA_LOOKUP_ERROR,
  RIXA_TIMEOUT,
  RIXA_NOT_INIT,
  RIXA_OTHER_ERROR,
} Rixa_Error;

typedef struct {
  char *bytes;
  uint32_t size;
} Rixa_bytes;

#define RIXA_BYTES_INIT(str) {.bytes = (str), .size = (str) ? strlen(str) : 0}

#define RIXA_BYTES_FREE(rb_ptr)                                                \
  do {                                                                         \
    if ((rb_ptr) && (rb_ptr)->bytes) {                                         \
      free((rb_ptr)->bytes);                                                   \
      (rb_ptr)->bytes = NULL;                                                  \
    }                                                                          \
    if (rb_ptr) {                                                              \
      (rb_ptr)->size = 0;                                                      \
    }                                                                          \
  } while (0)

// Singleton
typedef struct {
  pmix_proc_t proc;
  int init;
} GlobalPMIxState;

typedef struct {
  int timeout; // Seconds
} rixa_store;

int rixa_get_rank(GlobalPMIxState *state);
int rixa_get_world(GlobalPMIxState *state);

Rixa_Error rixa_set(GlobalPMIxState *state, rixa_store *store, const char *key,
                    const char *val, uint32_t val_len);

Rixa_Error rixa_get(GlobalPMIxState *state, rixa_store *store, const char *ket,
                    Rixa_bytes *out);

Rixa_Error rixa_get(GlobalPMIxState *state, rixa_store *store, const char *ket,
                    Rixa_bytes *out);
Rixa_Error rixa_wait(GlobalPMIxState *state, rixa_store *store,
                     const char keys[][PMIX_MAX_KEYLEN], uint32_t n,
                     uint32_t timout);
