#include "rixa_pmix_store.h"
#include <string.h>

int rixa_get_rank(GlobalPMIxState *GlobState) {
  if (GlobState->init) {
    return GlobState->proc.rank;
  } else
    return -1;
}

int rixa_get_world(GlobalPMIxState *GlobState) {

  if (!GlobState->init) {
    return -1;
  }
  pmix_value_t *val = NULL;
  pmix_proc_t job_info;
  PMIX_LOAD_NSPACE(job_info.nspace, GlobState->proc.nspace);
  job_info.rank = PMIX_RANK_WILDCARD;

  pmix_status_t rc = PMIx_Get(&job_info, PMIX_JOB_SIZE, NULL, 0, &val);

  uint32_t world_size = -1;
  if (PMIX_SUCCESS == rc && val != NULL) {
    world_size = val->data.uint32;
    PMIX_VALUE_RELEASE(val);
  }
  return world_size;
}

Rixa_Error rixa_set(GlobalPMIxState *state, rixa_store *store, const char *key,
                    const char *val, uint32_t val_len) {

  pmix_info_t info[1];
  pmix_byte_object_t bo;

  char *val_copy = malloc(val_len * sizeof(char));
  memcpy(val_copy, val, val_len);
  bo.bytes = val_copy;
  bo.size = val_len;

  PMIX_INFO_CONSTRUCT(&info[0]);
  strncpy(info[0].key, key, PMIX_MAX_KEYLEN);
  info[0].value.type = PMIX_BYTE_OBJECT;
  info[0].value.data.bo = bo;

  pmix_status_t status = PMIx_Publish(info, 1);

  PMIX_INFO_DESTRUCT(&info[0]);
  if (status != PMIX_SUCCESS) {
    return RIXA_PUBLISH_ERROR;
  }
  status = PMIx_Commit();
  if (status != PMIX_SUCCESS) {
    return RIXA_COMMIT_ERROR;
  }
  return RIXA_SUCCESS;
}

Rixa_Error rixa_get(GlobalPMIxState *state, rixa_store *store, const char *key,
                    Rixa_bytes *out) {

  pmix_pdata_t pdata[1];
  pmix_info_t info[2];

  PMIX_INFO_CONSTRUCT(&info[0]);
  PMIX_INFO_CONSTRUCT(&info[1]);
  int wait_flag = 1; // NOTE: for Lookup, PMIX_WAIT=1 means "block until found"
  PMIx_Info_load(&info[0], PMIX_WAIT, &wait_flag, PMIX_INT);
  PMIx_Info_load(&info[1], PMIX_TIMEOUT, &store->timeout, PMIX_INT);

  PMIX_PDATA_CONSTRUCT(&pdata[0]);
  strncpy(pdata[0].key, key, PMIX_MAX_KEYLEN);

  pmix_status_t status = PMIx_Lookup(pdata, 1, info, 2);

  if (status == PMIX_ERR_TIMEOUT) {
    PMIX_INFO_DESTRUCT(&info[0]);
    PMIX_INFO_DESTRUCT(&info[1]);
    return RIXA_TIMEOUT;
  }
  if (status != PMIX_SUCCESS) {
    PMIX_INFO_DESTRUCT(&info[0]);
    PMIX_INFO_DESTRUCT(&info[1]);
    return RIXA_OTHER_ERROR;
  }

  pmix_byte_object_t *bo = &pdata[0].value.data.bo;
  if (bo->bytes == NULL) {
    PMIX_INFO_DESTRUCT(&info[0]);
    PMIX_INFO_DESTRUCT(&info[1]);
    PMIX_PDATA_DESTRUCT(&pdata[0]);
    return RIXA_OTHER_ERROR;
  }
  out->size = bo->size;
  out->bytes = malloc(sizeof(char) * out->size);
  memcpy(out->bytes, bo->bytes, out->size);

  PMIX_INFO_DESTRUCT(&info[0]);
  PMIX_INFO_DESTRUCT(&info[1]);
  PMIX_PDATA_DESTRUCT(&pdata[0]);
  return RIXA_SUCCESS;
}
Rixa_Error rixa_wait(GlobalPMIxState *state, rixa_store *store,
                     const char keys[][PMIX_MAX_KEYLEN], uint32_t n,
                     uint32_t timeout) {
  float delta_T = 0.1;   // Fraction of second for every retry
  float total_sleep = 0; // total amount of time spend on sleeping
  if (timeout < 0)
    timeout = store->timeout;

  pmix_info_t info[2];

  PMIX_INFO_CONSTRUCT(&info[0]);
  PMIX_INFO_CONSTRUCT(&info[1]);
  int wait_flag = 0;
  PMIx_Info_load(&info[0], PMIX_WAIT, &wait_flag, PMIX_INT);
  PMIx_Info_load(&info[1], PMIX_TIMEOUT, &timeout, PMIX_INT);
  bool found[n];
  for (uint32_t i = 0; i < n; i++) {
    found[i] = 0;
  }
  uint32_t total_found = 0;

  while (total_found < n) {
    if (total_sleep > timeout) {
      PMIX_INFO_DESTRUCT(&info[0]);
      PMIX_INFO_DESTRUCT(&info[1]);
      return RIXA_TIMEOUT;
    }
    for (uint32_t i = 0; i < n; i++) {
      if (found[i])
        continue;
      const char *key = keys[i];
      pmix_pdata_t pdata[1];
      PMIX_PDATA_CONSTRUCT(&pdata[0]);
      strncpy(pdata[0].key, key, PMIX_MAX_KEYLEN);
      pmix_status_t rc = PMIx_Lookup(pdata, 1, info, 2);
      PMIX_PDATA_DESTRUCT(&pdata[0]);

      if (rc == PMIX_ERR_TIMEOUT) {
        PMIX_INFO_DESTRUCT(&info[0]);
        PMIX_INFO_DESTRUCT(&info[1]);
        return RIXA_TIMEOUT;
      } else if (rc == PMIX_ERR_NOT_FOUND) {
        continue;
      } else if (rc != PMIX_SUCCESS) {
        PMIX_INFO_DESTRUCT(&info[0]);
        PMIX_INFO_DESTRUCT(&info[1]);
        return RIXA_OTHER_ERROR;
      }
      found[i] = 1;
      total_found += 1;
    }
    total_sleep += delta_T;
    usleep(delta_T * 1000);
  }

  PMIX_INFO_DESTRUCT(&info[0]);
  PMIX_INFO_DESTRUCT(&info[1]);
  return RIXA_SUCCESS;
}
