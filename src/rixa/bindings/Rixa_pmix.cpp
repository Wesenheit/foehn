#include "Rixa_pmix.h"
#include <chrono>
#include <cstdint>
#include <cstring>
#include <thread>
#include <vector>

template <std::size_t N> struct PmixInfoGuard {
  pmix_info_t data[N];

  PmixInfoGuard() {
    for (auto &i : data)
      PMIX_INFO_CONSTRUCT(&i);
  }
  ~PmixInfoGuard() {
    for (auto &i : data)
      PMIX_INFO_DESTRUCT(&i);
  }

  pmix_info_t *get() { return data; }
  pmix_info_t &operator[](std::size_t i) { return data[i]; }
};

int rixa_get_rank(GlobalPMIxState *state) noexcept {
  if (!state->init)
    return -1;
  return static_cast<int>(state->proc.rank);
}

int rixa_get_local_rank(GlobalPMIxState *state) noexcept {
  if (!state->init)
    return -1;

  pmix_value_t *val = nullptr;
  pmix_status_t status =
      PMIx_Get(&state->proc, PMIX_LOCAL_RANK, nullptr, 0, &val);

  if (status != PMIX_SUCCESS || val == nullptr)
    return -1;

  uint16_t rc = val->data.uint16;
  PMIX_VALUE_RELEASE(val);
  return static_cast<int>(rc);
}

int rixa_get_world(GlobalPMIxState *state) noexcept {
  if (!state->init)
    return -1;

  pmix_value_t *val = nullptr;
  pmix_proc_t job_info;
  PMIX_LOAD_NSPACE(job_info.nspace, state->proc.nspace);
  job_info.rank = PMIX_RANK_WILDCARD;

  pmix_status_t rc = PMIx_Get(&job_info, PMIX_JOB_SIZE, nullptr, 0, &val);

  if (rc != PMIX_SUCCESS || val == nullptr)
    return -1;

  uint32_t world_size = val->data.uint32;
  PMIX_VALUE_RELEASE(val);
  return static_cast<int>(world_size);
}

RixaError rixa_set(GlobalPMIxState *state, RixaStore *store, const char *key,
                   const char *val, uint32_t val_len) noexcept {

  char *val_copy = static_cast<char *>(std::malloc(val_len));
  if (!val_copy)
    return RixaError::OtherError;
  std::memcpy(val_copy, val, val_len);

  pmix_info_t info[1];
  PMIX_INFO_CONSTRUCT(&info[0]);
  std::strncpy(info[0].key, key, PMIX_MAX_KEYLEN);
  info[0].value.type = PMIX_BYTE_OBJECT;
  info[0].value.data.bo.bytes = val_copy;
  info[0].value.data.bo.size = val_len;

  pmix_status_t status = PMIx_Publish(info, 1);
  PMIX_INFO_DESTRUCT(&info[0]);

  if (status != PMIX_SUCCESS)
    return RixaError::PublishError;

  status = PMIx_Commit();
  if (status != PMIX_SUCCESS)
    return RixaError::CommitError;

  return RixaError::Success;
}

// ── get
// ───────────────────────────────────────────────────────────────────────

RixaError rixa_get(GlobalPMIxState *state, RixaStore *store, const char *key,
                   RixaBytes *out) noexcept {

  PmixInfoGuard<2> info; // DESTRUCT called automatically on all exits
  int wait_flag = 1;
  PMIx_Info_load(&info[0], PMIX_WAIT, &wait_flag, PMIX_INT);
  PMIx_Info_load(&info[1], PMIX_TIMEOUT, &store->timeout, PMIX_INT);

  pmix_pdata_t pdata[1];
  PMIX_PDATA_CONSTRUCT(&pdata[0]);
  std::strncpy(pdata[0].key, key, PMIX_MAX_KEYLEN);

  pmix_status_t status = PMIx_Lookup(pdata, 1, info.get(), 2);

  if (status == PMIX_ERR_TIMEOUT) {
    PMIX_PDATA_DESTRUCT(&pdata[0]);
    return RixaError::Timeout;
  }
  if (status != PMIX_SUCCESS) {
    PMIX_PDATA_DESTRUCT(&pdata[0]);
    return RixaError::OtherError;
  }

  pmix_byte_object_t *bo = &pdata[0].value.data.bo;
  if (!bo->bytes) {
    PMIX_PDATA_DESTRUCT(&pdata[0]);
    return RixaError::OtherError;
  }

  out->size = static_cast<uint32_t>(bo->size);
  out->data = static_cast<char *>(std::malloc(out->size));
  if (!out->data) {
    PMIX_PDATA_DESTRUCT(&pdata[0]);
    return RixaError::OtherError;
  }
  std::memcpy(out->data, bo->bytes, out->size);

  PMIX_PDATA_DESTRUCT(&pdata[0]);
  return RixaError::Success;
}

// ── wait
// ──────────────────────────────────────────────────────────────────────

RixaError rixa_wait(GlobalPMIxState *state, RixaStore *store,
                    const char keys[][PMIX_MAX_KEYLEN], uint32_t n,
                    uint32_t timeout) noexcept {

  using namespace std::chrono;
  using namespace std::chrono_literals;

  if (timeout == 0)
    timeout = static_cast<uint32_t>(store->timeout);

  PmixInfoGuard<2> info;
  int wait_flag = 0;
  PMIx_Info_load(&info[0], PMIX_WAIT, &wait_flag, PMIX_INT);
  PMIx_Info_load(&info[1], PMIX_TIMEOUT, &timeout, PMIX_INT);

  std::vector<bool> found(n, false);
  uint32_t total_found = 0;

  constexpr auto delta = 100ms;
  auto deadline = steady_clock::now() + seconds(timeout);

  while (total_found < n) {
    if (steady_clock::now() > deadline)
      return RixaError::Timeout;

    for (uint32_t i = 0; i < n; i++) {
      if (found[i])
        continue;

      pmix_pdata_t pdata[1];
      PMIX_PDATA_CONSTRUCT(&pdata[0]);
      std::strncpy(pdata[0].key, keys[i], PMIX_MAX_KEYLEN);
      pmix_status_t rc = PMIx_Lookup(pdata, 1, info.get(), 2);
      PMIX_PDATA_DESTRUCT(&pdata[0]);

      if (rc == PMIX_ERR_TIMEOUT)
        return RixaError::Timeout;
      if (rc == PMIX_ERR_NOT_FOUND)
        continue;
      if (rc != PMIX_SUCCESS)
        return RixaError::OtherError;

      found[i] = true;
      ++total_found;
    }

    std::this_thread::sleep_for(delta);
  }

  return RixaError::Success;
}

RixaError rixa_check(GlobalPMIxState *state, RixaStore *store, const char *key,
                     uint32_t *out) noexcept {

  pmix_pdata_t pdata[1];
  PMIX_PDATA_CONSTRUCT(&pdata[0]);
  std::strncpy(pdata[0].key, key, PMIX_MAX_KEYLEN);
  pmix_status_t status = PMIx_Lookup(pdata, 1, nullptr, 0);
  PMIX_PDATA_DESTRUCT(&pdata[0]);

  if (status == PMIX_SUCCESS) {
    *out = 1;
    return RixaError::Success;
  }
  if (status == PMIX_ERR_NOT_FOUND) {
    *out = 0;
    return RixaError::Success;
  }
  return RixaError::LookupError;
}
