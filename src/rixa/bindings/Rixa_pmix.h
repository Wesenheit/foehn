#pragma once

#include "pmix_headers/pmix.h"
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <stdexcept>

enum class RixaError {
  Success,
  CommitError,
  PublishError,
  LookupError,
  Timeout,
  NotInit,
  OtherError,
};

struct RixaBytes {
  char *data = nullptr;
  uint32_t size = 0;

  static RixaBytes from_raw(char *ptr, uint32_t len) noexcept {
    return {ptr, len};
  }

  static RixaBytes from_cstr(char *ptr) noexcept {
    return {ptr, ptr ? static_cast<uint32_t>(std::strlen(ptr)) : 0};
  }

  void free() noexcept {
    std::free(data);
    data = nullptr;
    size = 0;
  }

  [[nodiscard]] char *release() noexcept {
    char *p = data;
    data = nullptr;
    size = 0;
    return p;
  }
};

struct GlobalPMIxState {
  pmix_proc_t proc{};
  bool init = false;
};

struct RixaStore {
  int timeout = 30; // seconds
};

int rixa_get_rank(GlobalPMIxState *state) noexcept;
int rixa_get_world(GlobalPMIxState *state) noexcept;
int rixa_get_local_rank(GlobalPMIxState *state) noexcept;

RixaError rixa_set(GlobalPMIxState *state, RixaStore *store, const char *key,
                   const char *val, uint32_t val_len) noexcept;

RixaError rixa_get(GlobalPMIxState *state, RixaStore *store, const char *key,
                   RixaBytes *out) noexcept;

RixaError rixa_wait(GlobalPMIxState *state, RixaStore *store,
                    const char keys[][PMIX_MAX_KEYLEN], uint32_t n,
                    uint32_t timeout) noexcept;

RixaError rixa_check(GlobalPMIxState *state, RixaStore *store, const char *key,
                     uint32_t *out) noexcept;
