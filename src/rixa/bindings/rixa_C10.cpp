#include <chrono>
#include <cstdint>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <stdexcept>
#include <string>
#include <torch/csrc/distributed/c10d/Store.hpp>
#include <torch/csrc/utils/pybind.h>

namespace py = pybind11;
extern "C" {
#include "rixa_pmix_store.h"
}

GlobalPMIxState state = {.init = 0};

class PMIxC10dStore : public c10d::Store {
private:
  rixa_store store;
  std::chrono::milliseconds timeout;

public:
  PMIxC10dStore() { init(); }

  ~PMIxC10dStore() override { finalize(); }

  void init() {
    pmix_status_t rc = PMIx_Init(&state.proc, NULL, 0);
    if (rc != PMIX_SUCCESS)
      throw std::runtime_error("PMIX faild to initialize!");
    state.init = 1;
    store.timeout = 30;
    this->timeout = std::chrono::milliseconds{30000};
  }

  void finalize() {
    if (state.init) {
      PMIx_Finalize(NULL, 0);
      state.init = 0;
    };
  }
  c10::intrusive_ptr<c10d::Store> clone() override {
    throw std::runtime_error("PMIxC10dStore does not support clone()");
  }

  // ----- required overrides -----
  void set(const std::string &key, const std::vector<uint8_t> &value) override {
    Rixa_Error rc =
        rixa_set(&state, NULL, key.c_str(),
                 reinterpret_cast<const char *>(value.data()), value.size());
    if (rc != RIXA_SUCCESS)
      throw std::runtime_error("PMIxStore::set failed for key: " + key);
  }

  std::vector<uint8_t> get(const std::string &key) override {
    Rixa_bytes out;
    Rixa_Error rc = rixa_get(&state, &store, key.c_str(), &out);
    if (rc != RIXA_SUCCESS)
      throw std::runtime_error("PMIxStore::get failed for key: " + key);
    std::vector<uint8_t> result(out.bytes, out.bytes + out.size);
    free(out.bytes);

    std::string s(result.begin(), result.end());
    return result;
  }
  void append(const std::string &key,
              const std::vector<uint8_t> &value) override {
    // get current value, append, set back
    try {
      auto current = get(key);
      current.insert(current.end(), value.begin(), value.end());
      set(key, current);
    } catch (...) {
      set(key, value);
    }
  }

  int64_t add(const std::string &key, int64_t value) override {
    throw std::runtime_error("Not Implemented");
  }

  bool deleteKey(const std::string &key) override {
    throw std::runtime_error("Not Implemented");
  }

  bool check(const std::vector<std::string> &keys) override {
    uint32_t flag = 0;
    for (const std::string &key : keys) {
      Rixa_Error status = rixa_check(&state, &store, key.data(), &flag);
      if (status != RIXA_SUCCESS) {
        throw std::runtime_error("Error during lookup!");
      }
      if (!flag)
        return false;
    }
    return true;
  }

  void wait(const std::vector<std::string> &keys) override {
    wait(keys, timeout_);
  }
  int64_t getNumKeys() override { return 0; }

  void wait(const std::vector<std::string> &keys,
            const std::chrono::milliseconds &timeout) override {

    std::vector<std::array<char, PMIX_MAX_KEYLEN>> ckeys(keys.size());
    for (size_t i = 0; i < keys.size(); i++) {
      strncpy(ckeys[i].data(), keys[i].c_str(), PMIX_MAX_KEYLEN);
    }

    Rixa_Error rc =
        rixa_wait(&state, &store,
                  reinterpret_cast<char (*)[PMIX_MAX_KEYLEN]>(ckeys.data()),
                  ckeys.size(), static_cast<uint32_t>(timeout.count() / 1000));
    if (rc == RIXA_TIMEOUT)
      throw std::runtime_error("PMIxStore::wait timed out for "
                               "key: " +
                               keys[0]);
    if (rc != RIXA_SUCCESS)
      throw std::runtime_error("PMIxStore::wait failed for "
                               "key: " +
                               keys[0]);
  }

  void setTimeout(const std::chrono::milliseconds &timeout) override {
    this->store.timeout = static_cast<uint32_t>(timeout.count() / 1000);
    this->timeout = timeout;
  }

  std::vector<std::vector<uint8_t>>
  multiGet(const std::vector<std::string> &keys) override {
    std::vector<std::vector<uint8_t>> result;
    result.reserve(keys.size());
    for (const auto &key : keys)
      result.push_back(get(key));
    return result;
  }

  void multiSet(const std::vector<std::string> &keys,
                const std::vector<std::vector<uint8_t>> &values) override {

    for (size_t i = 0; i < keys.size(); i++)
      set(keys[i], values[i]);
  }

  const std::chrono::milliseconds &getTimeout() const noexcept override {
    return this->timeout;
  }

  bool hasExtendedApi() const override { return false; }

  int rank() { return rixa_get_rank(&state); }

  int world_size() { return rixa_get_world(&state); }
  int local_rank() { return rixa_get_local_rank(&state); }
};

PYBIND11_MODULE(_rixa_torch, m) {
  py::module_::import("torch.distributed.distributed_c10d");

  py::class_<PMIxC10dStore, c10d::Store, c10::intrusive_ptr<PMIxC10dStore>>(
      m, "PMIxC10dStore")
      .def(py::init<>())
      .def("rank", &PMIxC10dStore::rank)
      .def("local_rank", &PMIxC10dStore::local_rank)
      .def("world_size", &PMIxC10dStore::world_size)
      .def("finalize", &PMIxC10dStore::finalize)
      .def("init", &PMIxC10dStore::init);
}
