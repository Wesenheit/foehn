#include <array>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/csrc/distributed/c10d/Store.hpp>
#include <torch/csrc/utils/pybind.h>

#include "Rixa_pmix.hpp"

namespace py = pybind11;

// ── global singleton
// ──────────────────────────────────────────────────────────

static GlobalPMIxState state{};

// ── PMIxC10dStore
// ─────────────────────────────────────────────────────────────

class PMIxC10dStore : public c10d::Store {
public:
  PMIxC10dStore() { init(); }
  ~PMIxC10dStore() override { finalize(); }

  PMIxC10dStore(const PMIxC10dStore &) = delete;
  PMIxC10dStore &operator=(const PMIxC10dStore &) = delete;

  void init() {
    pmix_status_t rc = PMIx_Init(&state.proc, nullptr, 0);
    if (rc != PMIX_SUCCESS)
      throw std::runtime_error("PMIx failed to initialize!");
    state.init = true;
    store_.timeout = std::chrono::milliseconds{30'000};
  }

  void finalize() {
    if (state.init) {
      PMIx_Finalize(nullptr, 0);
      state.init = false;
    }
  }
  #if (TORCH_VERSION_MAJOR > 2) || \
      (TORCH_VERSION_MAJOR == 2 && TORCH_VERSION_MINOR > 8) || \
      (TORCH_VERSION_MAJOR == 2 && TORCH_VERSION_MINOR == 8 && TORCH_VERSION_PATCH > 0)
      c10::intrusive_ptr<c10d::Store> clone() override
  #else
    c10::intrusive_ptr<c10d::Store> clone()
  #endif
  {
    throw std::runtime_error("PMIxC10dStore does not support clone()");
  }

  // ── set ───────────────────────────────────────────────────────────────────

  void set(const std::string &key, const std::vector<uint8_t> &value) override {
    RixaError rc = rixa_set(state, store_, key.c_str(),
                            reinterpret_cast<const char *>(value.data()),
                            static_cast<uint32_t>(value.size()));
    if (rc != RixaError::Success)
      throw std::runtime_error("PMIxStore::set failed for key: " + key);
  }

  // ── get ───────────────────────────────────────────────────────────────────

  std::vector<uint8_t> get(const std::string &key) override {
    RixaBytes out{};
    RixaError rc = rixa_get(state, store_, key.c_str(), out);
    if (rc != RixaError::Success)
      throw std::runtime_error("PMIxStore::get failed for key: " + key);

    std::vector<uint8_t> result(out.data, out.data + out.size);
    out.free();
    return result;
  }

  // ── append ────────────────────────────────────────────────────────────────

  void append(const std::string &key,
              const std::vector<uint8_t> &value) override {
    try {
      auto current = get(key);
      current.insert(current.end(), value.begin(), value.end());
      set(key, current);
    } catch (...) {
      set(key, value);
    }
  }

  // ── check ─────────────────────────────────────────────────────────────────

  bool check(const std::vector<std::string> &keys) override {
    for (const auto &key : keys) {
      bool flag = false;
      RixaError status = rixa_check(state, store_, key.c_str(), flag);
      if (status != RixaError::Success)
        throw std::runtime_error("PMIxStore::check error on key: " + key);
      if (!flag)
        return false;
    }
    return true;
  }

  // ── wait ──────────────────────────────────────────────────────────────────

  void wait(const std::vector<std::string> &keys) override {
    wait(keys, timeout_);
  }

  void wait(const std::vector<std::string> &keys,
            const std::chrono::milliseconds &timeout) override {

    std::vector<std::array<char, PMIX_MAX_KEYLEN>> ckeys(keys.size());
    for (std::size_t i = 0; i < keys.size(); i++) {
      std::strncpy(ckeys[i].data(), keys[i].c_str(), PMIX_MAX_KEYLEN - 1);
      ckeys[i][PMIX_MAX_KEYLEN - 1] = '\0';
    }

    RixaError rc = rixa_wait(
        state, store_,
        reinterpret_cast<const char (*)[PMIX_MAX_KEYLEN]>(ckeys[0].data()),
        static_cast<uint32_t>(keys.size()),
        static_cast<uint32_t>(timeout.count() / 1000));

    if (rc == RixaError::Timeout)
      throw std::runtime_error("PMIxStore::wait timed out for key: " + keys[0]);
    if (rc != RixaError::Success)
      throw std::runtime_error("PMIxStore::wait failed for key: " + keys[0]);
  }

  // ── multi get/set ─────────────────────────────────────────────────────────

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
    for (std::size_t i = 0; i < keys.size(); i++)
      set(keys[i], values[i]);
  }

  // ── misc overrides ────────────────────────────────────────────────────────

  int64_t add(const std::string &key, int64_t value) override {
    throw std::runtime_error("PMIxStore::add not implemented");
  }

  bool deleteKey(const std::string &key) override {
    throw std::runtime_error("PMIxStore::deleteKey not implemented");
  }

  int64_t getNumKeys() override { return 0; }

  void setTimeout(const std::chrono::milliseconds &timeout) override {
    store_.timeout = timeout;
  }

  const std::chrono::milliseconds &getTimeout() const noexcept override {
    return store_.timeout;
  }

  bool hasExtendedApi() const override { return false; }

  // ── rixa extras ───────────────────────────────────────────────────────────

  int rank() const noexcept { return rixa_get_rank(state); }
  int world_size() const noexcept { return rixa_get_world(state); }
  int local_rank() const noexcept { return rixa_get_local_rank(state); }

private:
  RixaStore store_{};
};

// ── pybind11 module
// ───────────────────────────────────────────────────────────

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
