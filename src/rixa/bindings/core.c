#include <Python.h>
#include <stdio.h>
#include <string.h>
#include <structmember.h>
#include <unistd.h>

#include "pyerrors.h"
#include "pyport.h"
#include "rixa_pmix_store.h"

typedef struct {
  PyObject_HEAD;
  rixa_store store;
} PyPMIx;
static GlobalPMIxState state;

Py_ssize_t get_string_from_python(PyObject *val_obj, const char **out) {
  Py_ssize_t return_val;
  if (PyBytes_Check(val_obj)) {
    PyBytes_AsStringAndSize(val_obj, (char **)out, &return_val);
  } else if (PyUnicode_Check(val_obj)) {
    *out = PyUnicode_AsUTF8(val_obj);
    return_val = strlen(*out);
  } else {
    return 0;
  }
  return return_val;
}

static int PMIxObjInit(PyObject *self, PyObject *args) {
  PyPMIx *self_pmix = (PyPMIx *)self;
  if (!PyArg_ParseTuple(args, "i", &self_pmix->store.timeout)) {
    return -1;
  }
  // self_pmix->timeout = 30;

  if (state.init) {
    PyErr_SetString(PyExc_TypeError, "PMIx already started!");
    return -1;
  }

  pmix_status_t rc = PMIx_Init(&state.proc, NULL, 0);
  if (rc != PMIX_SUCCESS) {
    PyErr_SetString(PyExc_TypeError, "Failed to init PMIx!");
    return -1;
  }

  state.init = 1;
  return 0;
}

static PyTypeObject PMIxType = {
    PyVarObject_HEAD_INIT(NULL, 0).tp_name = "PMIx_core.PMIxStore",
    .tp_basicsize = sizeof(PyPMIx),
    .tp_itemsize = 0,
    .tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
    .tp_doc = "Custom PMX storage",
    .tp_init = (initproc)PMIxObjInit, // Default constructor
    .tp_new = PyType_GenericNew,
};
// IMPLEMENTATIONS

// 1. GET RANK
static PyObject *get_rank_python(PyObject *self, PyObject *Py_UNUSED(ignored)) {
  int rank = rixa_get_rank(&state);
  if (rank < 0) {
    PyErr_SetString(PyExc_RuntimeError,
                    "Pmix runtime not started, failed to query rank!");
    return NULL;
  }

  PyObject *result = PyLong_FromLong((long)rank);
  return result;
}

static PyObject *get_local_rank_python(PyObject *self,
                                       PyObject *Py_UNUSED(ignored)) {
  int rank = rixa_get_local_rank(&state);
  if (rank < 0) {
    PyErr_SetString(PyExc_RuntimeError,
                    "Pmix runtime not started, failed to query rank!");
    return NULL;
  }

  PyObject *result = PyLong_FromLong((long)rank);
  return result;
}
// 2. GET WORLD
static PyObject *get_world_python(PyObject *self,
                                  PyObject *Py_UNUSED(ignored)) {

  int world = rixa_get_world(&state);
  if (world < 0) {
    PyErr_SetString(PyExc_RuntimeError,
                    "Pmix runtime not started, failed to query world!");
    return NULL;
  }

  PyObject *result = PyLong_FromLong((long)world);
  return result;
}

// 3. SET
static PyObject *set(PyObject *self, PyObject *args) {

  PyObject *key_obj, *val_obj;
  const char *key, *val;
  if (!PyArg_ParseTuple(args, "OO", &key_obj, &val_obj)) {
    return NULL;
  }
  Py_ssize_t size_key = get_string_from_python(key_obj, &key);
  if (!size_key) {
    return NULL;
  }

  Py_ssize_t size_val = get_string_from_python(val_obj, &val);
  if (!size_val) {
    return NULL;
  }

  Rixa_Error status = rixa_set(&state, NULL, key, val, size_val);
  if (status != RIXA_SUCCESS) {
    PyErr_Format(PyExc_RuntimeError, "(set) failed to push key '%s': %s", key,
                 PMIx_Error_string(status));
    return NULL;
  }

  Py_INCREF(Py_None);
  return Py_None;
}

// 4. GET
static PyObject *get(PyObject *self, PyObject *args) {
  PyPMIx *self_pmix = (PyPMIx *)self;

  PyObject *key_obj;
  const char *key;
  if (!PyArg_ParseTuple(args, "O", &key_obj)) {
    return NULL;
  }
  Py_ssize_t size_key = get_string_from_python(key_obj, &key);
  if (!size_key) {
    return NULL;
  }

  Rixa_bytes out;
  Rixa_Error status = rixa_get(&state, &self_pmix->store, key, &out);

  if (status == RIXA_TIMEOUT) {
    PyErr_Format(PyExc_TimeoutError, "(get) Timeout to get key '%s'!", key);
    return NULL;
  }
  if (status != RIXA_SUCCESS) {
    PyErr_Format(PyExc_TypeError, "(get) Failed to get key '%s'!", key);
    return NULL;
  }

  PyObject *result;
  result = PyBytes_FromStringAndSize(out.bytes, out.size);
  free(out.bytes);
  return result;
}

// 5. WATI
static PyObject *wait_for_keys(PyObject *self, PyObject *args) {
  PyPMIx *self_pmix = (PyPMIx *)self;
  PyObject *keys_list;
  int timeout;
  if (!PyArg_ParseTuple(args, "O|i", &keys_list, &timeout)) {
    return NULL;
  }
  if (timeout < 0) {
    timeout = self_pmix->store.timeout;
  }
  if (!PyList_Check(keys_list)) {
    PyErr_SetString(PyExc_TypeError, "keys must be a list");
    return NULL;
  }

  Py_ssize_t n = PyList_Size(keys_list);
  char keys[n][PMIX_MAX_KEYLEN];

  for (Py_ssize_t i = 0; i < n; i++) {
    PyObject *key_obj = PyList_GetItem(keys_list, i);
    const char *key;
    if (PyUnicode_Check(key_obj)) {
      key = PyUnicode_AsUTF8(key_obj);
    } else if (PyBytes_Check(key_obj)) {
      key = PyBytes_AsString(key_obj);
    } else {
      PyErr_SetString(PyExc_TypeError, "key must be str or bytes");
      return NULL;
    }
    strncpy(keys[i], key, PMIX_MAX_KEYLEN);
  }
  Rixa_Error status = rixa_wait(&state, &self_pmix->store, keys, n, timeout);
  if (status == RIXA_TIMEOUT) {
    PyErr_SetString(PyExc_RuntimeError, "Timout reached!");
    return NULL;
  } else if (status != RIXA_SUCCESS) {
    PyErr_SetString(PyExc_RuntimeError, "Error encoutered, failed!");
    return NULL;
  }

  Py_INCREF(Py_None);
  return Py_None;
}

// -1. CLEAN UP
void PMIxCleanup(void) {
  if (state.init == 1) {
    PMIx_Finalize(NULL, 0);
  }
}

static PyMethodDef Custom_methods[] = {
    {"get_rank", get_rank_python, METH_NOARGS, "Get the process rank"},
    {"get_world", get_world_python, METH_NOARGS, "Get the world size"},
    {"get_local_rank", get_local_rank_python, METH_NOARGS,
     "Get the local process rank"},
    {"set", set, METH_VARARGS, "set a key-value pair"},
    {"get", get, METH_VARARGS, "get a value for given key"},
    {"wait", wait_for_keys, METH_VARARGS, "wait for arrays of keys"},
    {NULL}};

static struct PyModuleDef coremodule = {
    PyModuleDef_HEAD_INIT, "_core", NULL, -1, NULL, NULL, NULL, NULL, NULL};

PyMODINIT_FUNC PyInit_PMIx_core(void) {
  PyObject *m;

  PMIxType.tp_methods = Custom_methods;

  if (PyType_Ready(&PMIxType) < 0)
    return NULL;

  m = PyModule_Create(&coremodule);
  if (m == NULL)
    return NULL;

  Py_INCREF(&PMIxType);
  if (PyModule_AddObject(m, "PMIxStore", (PyObject *)&PMIxType) < 0) {
    Py_DECREF(&PMIxType);
    Py_DECREF(m);
    return NULL;
  }

  Py_AtExit(PMIxCleanup);

  state.init = 0; // set that we can init PMIX
  return m;
}
