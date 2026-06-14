#include <Python.h>
#include <array>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <vector>

#include "Rixa_pmix.hpp"
#include "unicodeobject.h"

// ── Python object
// ─────────────────────────────────────────────────────────────

struct PyPMIx {
  PyObject_HEAD RixaStore store;
};

static GlobalPMIxState state;

// ── helpers
// ───────────────────────────────────────────────────────────────────

static Py_ssize_t get_string_from_python(PyObject *val_obj, const char **out) {
  if (PyBytes_Check(val_obj)) {
    Py_ssize_t len = 0;
    if (PyBytes_AsStringAndSize(val_obj, const_cast<char **>(out), &len) < 0)
      return -1;
    return len;
  }

  if (PyUnicode_Check(val_obj)) {
    Py_ssize_t len = 0;
    const char *s = PyUnicode_AsUTF8AndSize(val_obj, &len);
    if (!s)
      return -1;
    *out = s;
    return len;
  }

  return -1;
}

// ── init / cleanup
// ────────────────────────────────────────────────────────────

static int PMIxObjInit(PyObject *self, PyObject *args, PyObject *kwargs) {
  auto *self_pmix = reinterpret_cast<PyPMIx *>(self);

  int timeout;
  if (!PyArg_ParseTuple(args, "i", &timeout))
    return -1;

  self_pmix->store.timeout = std::chrono::seconds(timeout);

  if (state.init) {
    PyErr_SetString(PyExc_RuntimeError, "PMIx already started!");
    return -1;
  }

  pmix_status_t rc = PMIx_Init(&state.proc, nullptr, 0);
  if (rc != PMIX_SUCCESS) {
    PyErr_SetString(PyExc_RuntimeError, "Failed to init PMIx!");
    return -1;
  }

  state.init = true;
  return 0;
}

static void PMIxCleanup() {
  if (state.init) {
    PMIx_Finalize(nullptr, 0);
    state.init = false;
  }
}

// ── get_rank
// ──────────────────────────────────────────────────────────────────

static PyObject *get_rank_python(PyObject *self, PyObject *) {
  int rank = rixa_get_rank(state);
  if (rank < 0) {
    PyErr_SetString(PyExc_RuntimeError,
                    "PMIx runtime not started, failed to query rank!");
    return nullptr;
  }
  return PyLong_FromLong(rank);
}

static PyObject *get_local_rank_python(PyObject *self, PyObject *) {
  int rank = rixa_get_local_rank(state);
  if (rank < 0) {
    PyErr_SetString(PyExc_RuntimeError,
                    "PMIx runtime not started, failed to query local rank!");
    return nullptr;
  }
  return PyLong_FromLong(rank);
}

// ── get_world
// ─────────────────────────────────────────────────────────────────

static PyObject *get_world_python(PyObject *self, PyObject *) {
  int world = rixa_get_world(state);
  if (world < 0) {
    PyErr_SetString(PyExc_RuntimeError,
                    "PMIx runtime not started, failed to query world!");
    return nullptr;
  }
  return PyLong_FromLong(world);
}

// ── set
// ───────────────────────────────────────────────────────────────────────

static PyObject *pmix_set(PyObject *self, PyObject *args) {
  auto *self_pmix = reinterpret_cast<PyPMIx *>(self);
  PyObject *key_obj, *val_obj;
  if (!PyArg_ParseTuple(args, "OO", &key_obj, &val_obj))
    return nullptr;

  const char *key = nullptr, *val = nullptr;
  if (get_string_from_python(key_obj, &key) < 0) {
    PyErr_SetString(PyExc_TypeError, "key must be str or bytes");
    return nullptr;
  }
  Py_ssize_t size_val = get_string_from_python(val_obj, &val);
  if (size_val < 0) {
    PyErr_SetString(PyExc_TypeError, "val must be str or bytes");
    return nullptr;
  }

  RixaError status = rixa_set(state, self_pmix->store, key, val,
                              static_cast<uint32_t>(size_val));
  if (status != RixaError::Success) {
    PyErr_Format(PyExc_RuntimeError, "(set) failed to push key '%s'", key);
    return nullptr;
  }

  Py_RETURN_NONE;
}

// ── get
// ───────────────────────────────────────────────────────────────────────

static PyObject *pmix_get(PyObject *self, PyObject *args) {
  auto *self_pmix = reinterpret_cast<PyPMIx *>(self);

  PyObject *key_obj;
  if (!PyArg_ParseTuple(args, "O", &key_obj))
    return nullptr;

  const char *key = nullptr;
  if (get_string_from_python(key_obj, &key) < 0) {
    PyErr_SetString(PyExc_TypeError, "key must be str or bytes");
    return nullptr;
  }

  RixaBytes out{};
  RixaError status = rixa_get(state, self_pmix->store, key, out);

  if (status == RixaError::Timeout) {
    PyErr_Format(PyExc_TimeoutError, "(get) timeout waiting for key '%s'", key);
    return nullptr;
  }
  if (status != RixaError::Success) {
    PyErr_Format(PyExc_RuntimeError, "(get) failed to get key '%s'", key);
    return nullptr;
  }

  PyObject *result = PyBytes_FromStringAndSize(out.data, out.size);
  out.free(); // RixaBytes RAII method from rewritten header
  return result;
}

// ── wait
// ──────────────────────────────────────────────────────────────────────

static PyObject *wait_for_keys(PyObject *self, PyObject *args) {
  auto *self_pmix = reinterpret_cast<PyPMIx *>(self);

  PyObject *keys_list;
  int timeout = -1;
  if (!PyArg_ParseTuple(args, "O|i", &keys_list, &timeout))
    return nullptr;

  if (!PyList_Check(keys_list)) {
    PyErr_SetString(PyExc_TypeError, "keys must be a list");
    return nullptr;
  }

  Py_ssize_t n = PyList_Size(keys_list);
  std::vector<std::array<char, PMIX_MAX_KEYLEN>> keys(n);

  for (Py_ssize_t i = 0; i < n; i++) {
    PyObject *key_obj = PyList_GetItem(keys_list, i);
    const char *key = nullptr;

    if (PyUnicode_Check(key_obj)) {
      key = PyUnicode_AsUTF8AndSize(key_obj, nullptr);
    } else if (PyBytes_Check(key_obj)) {
      key = PyBytes_AsString(key_obj);
    } else {
      PyErr_SetString(PyExc_TypeError, "keys must be str or bytes");
      return nullptr;
    }

    std::strncpy(keys[i].data(), key, PMIX_MAX_KEYLEN - 1);
    keys[i][PMIX_MAX_KEYLEN - 1] = '\0'; // guarantee null termination
  }

  RixaError status = rixa_wait(
      state, self_pmix->store,
      reinterpret_cast<const char (*)[PMIX_MAX_KEYLEN]>(keys[0].data()),
      static_cast<uint32_t>(n), static_cast<uint32_t>(timeout));

  if (status == RixaError::Timeout) {
    PyErr_SetString(PyExc_TimeoutError, "Timeout reached waiting for keys!");
    return nullptr;
  }
  if (status != RixaError::Success) {
    PyErr_SetString(PyExc_RuntimeError, "Error encountered in wait!");
    return nullptr;
  }

  Py_RETURN_NONE;
}

// ── method table
// ──────────────────────────────────────────────────────────────

static PyMethodDef PMIx_methods[] = {
    {"get_rank", get_rank_python, METH_NOARGS, "Get the process rank"},
    {"get_world", get_world_python, METH_NOARGS, "Get the world size"},
    {"get_local_rank", get_local_rank_python, METH_NOARGS,
     "Get the local process rank"},
    {"set", pmix_set, METH_VARARGS, "Set a key-value pair"},
    {"get", pmix_get, METH_VARARGS, "Get a value for a given key"},
    {"wait", wait_for_keys, METH_VARARGS, "Wait for a list of keys"},
    {nullptr, nullptr, 0, nullptr}};

// ── type spec (stable ABI)
// ────────────────────────────────────────────────────

static PyType_Slot PMIx_slots[] = {
    {Py_tp_doc, const_cast<char *>("PMIx key-value store")},
    {Py_tp_init, reinterpret_cast<void *>(PMIxObjInit)},
    {Py_tp_new, reinterpret_cast<void *>(PyType_GenericNew)},
    {Py_tp_methods, PMIx_methods},
    {0, nullptr}};

static PyType_Spec PMIxType_spec = {
    .name = "PMIx_core.PMIxStore",
    .basicsize = static_cast<int>(sizeof(PyPMIx)),
    .itemsize = 0,
    .flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
    .slots = PMIx_slots,
};

// ── module init
// ───────────────────────────────────────────────────────────────

static struct PyModuleDef coremodule = {PyModuleDef_HEAD_INIT,
                                        "PMIx_core",
                                        nullptr,
                                        -1,
                                        nullptr,
                                        nullptr,
                                        nullptr,
                                        nullptr,
                                        nullptr};

PyMODINIT_FUNC PyInit_PMIx_core() {
  PyObject *m = PyModule_Create(&coremodule);
  if (!m)
    return nullptr;

  PyObject *PMIxType = PyType_FromSpec(&PMIxType_spec);
  if (!PMIxType) {
    Py_DECREF(m);
    return nullptr;
  }

  if (PyModule_AddObjectRef(m, "PMIxStore", PMIxType) < 0) {
    Py_DECREF(PMIxType);
    Py_DECREF(m);
    return nullptr;
  }

  Py_AtExit(PMIxCleanup);
  state.init = false;
  return m;
}
