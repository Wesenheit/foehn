#include "pmix_macros.h"
#include "pmix_types.h"
#include <Python.h>
#include <pmix.h>
#include <string.h>
#include <structmember.h>

static struct {
  pmix_proc_t proc;
  int init;
} GlobState;

typedef struct {
  PyObject_HEAD;
  int timeout;
} PyPMIx;

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
  PyPMIx *self_pmix = self;
  if (!PyArg_ParseTuple(args, "i", &self_pmix->timeout)) {
    return -1;
  }

  if (GlobState.init) {
    PyErr_SetString(PyExc_TypeError, "PMIx already started!");
    return -1;
  }

  pmix_status_t rc = PMIx_Init(&GlobState.proc, NULL, 0);
  if (rc != PMIX_SUCCESS) {
    PyErr_SetString(PyExc_TypeError, "Failed to init PMIx!");
    return -1;
  }

  GlobState.init = 1;
  return 0;
}

static PyTypeObject PMIxType = {
    PyVarObject_HEAD_INIT(NULL, 0).tp_name = "PMIx_core.PMIxStore",
    .tp_doc = "Custom PMX storage for pytorch",
    .tp_basicsize = sizeof(PyPMIx),
    .tp_itemsize = 0,
    .tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
    .tp_new = PyType_GenericNew,
    .tp_init = (initproc)PMIxObjInit, // Default constructor
};
// IMPLEMENTATIONS

// 1. GET RANK
static PyObject *get_rank(PyObject *self, PyObject *Py_UNUSED(ignored)) {
  int rank = GlobState.proc.rank;
  PyObject *result = PyLong_FromLong((long)rank);
  return result;
}

// 2. GET WORLD
static PyObject *get_world(PyObject *self, PyObject *Py_UNUSED(ignored)) {
  pmix_value_t *val = NULL;
  pmix_proc_t job_info;
  PMIX_LOAD_NSPACE(job_info.nspace, GlobState.proc.nspace);
  job_info.rank = PMIX_RANK_WILDCARD;

  pmix_status_t rc = PMIx_Get(&job_info, PMIX_JOB_SIZE, NULL, 0, &val);

  uint32_t world_size = -1;
  if (PMIX_SUCCESS == rc && val != NULL) {
    world_size = val->data.uint32;
    PMIX_VALUE_RELEASE(val);
  }
  PyObject *result = PyLong_FromLong((long)world_size);
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

  pmix_value_t PMIX_value;
  pmix_byte_object_t PMIX_bytes;
  char *val_copy = malloc(size_val);
  memcpy(val_copy, val, size_val);
  PMIX_bytes.bytes = val_copy;
  PMIX_bytes.size = size_val;

  PMIX_value.type = PMIX_BYTE_OBJECT;
  PMIX_value.data.bo = PMIX_bytes;
  pmix_status_t status = PMIx_Put(PMIX_GLOBAL, key, &PMIX_value);
  PMIX_VALUE_DESTRUCT(&PMIX_value);

  if (status != PMIX_SUCCESS) {
    PyErr_Format(PyExc_RuntimeError, "(set) failed to push key '%s': %s", key,
                 PMIx_Error_string(status));
    return NULL;
  }
  status = PMIx_Commit();
  if (status != PMIX_SUCCESS) {
    PyErr_Format(PyExc_RuntimeError, "(set) failed to commit key '%s': %s", key,
                 PMIx_Error_string(status));
    return NULL;
  }

  Py_INCREF(Py_None);
  return Py_None;
}

// 4. GET
static PyObject *get(PyObject *self, PyObject *args) {
  PyPMIx *self_pmix = self;

  PyObject *key_obj;
  const char *key;
  if (!PyArg_ParseTuple(args, "O", &key_obj)) {
    return NULL;
  }
  Py_ssize_t size_key = get_string_from_python(key_obj, &key);
  if (!size_key) {
    return NULL;
  }

  pmix_value_t *return_val;
  pmix_info_t info[2];
  PMIX_INFO_CONSTRUCT(&info[0]);
  PMIX_INFO_CONSTRUCT(&info[1]);

  PMIx_Info_load(&info[0], PMIX_WAIT, NULL, PMIX_BOOL);
  PMIx_Info_load(&info[1], PMIX_TIMEOUT, &self_pmix->timeout, PMIX_INT);

  pmix_proc_t proc;
  PMIX_PROC_CONSTRUCT(&proc);
  PMIX_PROC_LOAD(&proc, GlobState.proc.nspace, PMIX_RANK_UNDEF);

  pmix_status_t status = PMIx_Get(&proc, key, info, 2, &return_val);
  PMIX_INFO_DESTRUCT(&info[0]);
  PMIX_INFO_DESTRUCT(&info[1]);
  PMIX_PROC_DESTRUCT(&proc);

  if (status == PMIX_ERR_TIMEOUT) {
    PyErr_Format(PyExc_TimeoutError, "(get) Timeout to get key '%s'!", key);
    return NULL;
  }
  if (status != PMIX_SUCCESS) {
    PyErr_Format(PyExc_TypeError, "(get) Failed to get key '%s'!", key);
    return NULL;
  }
  if (return_val == NULL) {
    PyErr_SetString(PyExc_RuntimeError, "PMIx_Get returned NULL value");
    return NULL;
  }

  if (return_val->type != PMIX_BYTE_OBJECT) {
    PyErr_SetString(PyExc_TypeError,
                    "Posted something different than byte object!");
    return NULL;
  }

  if (return_val->data.bo.bytes == NULL) {
    PyErr_SetString(PyExc_RuntimeError, "PMIx_Get returned NULL bytes");
    PMIX_VALUE_RELEASE(return_val);
    return NULL;
  }

  PyObject *result;
  // PyErr_Format(PyExc_RuntimeError, "%i", return_val->data.bo.size);
  // return NULL;
  result = PyBytes_FromStringAndSize(return_val->data.bo.bytes,
                                     return_val->data.bo.size);

  PMIX_VALUE_RELEASE(return_val);
  return result;
}

// 5. WATI
static PyObject *wait_for_keys(PyObject *self, PyObject *args) {
  PyPMIx *self_pmix = self;
  PyObject *keys_list;
  int _dummy_timeout;
  if (!PyArg_ParseTuple(args, "O|i", &keys_list, &_dummy_timeout)) {
    return NULL;
  }

  if (!PyList_Check(keys_list)) {
    PyErr_SetString(PyExc_TypeError, "keys must be a list");
    return NULL;
  }

  Py_ssize_t n = PyList_Size(keys_list);

  pmix_proc_t proc;
  pmix_info_t info[2];

  PMIX_INFO_CONSTRUCT(&info[0]);
  PMIX_INFO_CONSTRUCT(&info[1]);
  PMIx_Info_load(&info[0], PMIX_WAIT, NULL, PMIX_BOOL);
  PMIx_Info_load(&info[1], PMIX_TIMEOUT, &self_pmix->timeout, PMIX_INT);

  PMIX_PROC_CONSTRUCT(&proc);
  PMIX_PROC_LOAD(&proc, GlobState.proc.nspace, PMIX_RANK_UNDEF);

  for (Py_ssize_t i = 0; i < n; i++) {
    PyObject *key_obj = PyList_GetItem(keys_list, i);
    const char *key;
    if (PyUnicode_Check(key_obj)) {
      key = PyUnicode_AsUTF8(key_obj);
    } else if (PyBytes_Check(key_obj)) {
      key = PyBytes_AsString(key_obj);
    } else {
      PyErr_SetString(PyExc_TypeError, "key must be str or bytes");
      goto err_cleanup;
    }

    pmix_value_t *val;
    pmix_status_t rc = PMIx_Get(&proc, key, info, 2, &val);

    if (rc == PMIX_ERR_TIMEOUT) {
      PyErr_Format(PyExc_TimeoutError, "key '%s' not available within timeout",
                   key);
      goto err_cleanup;
    } else if (rc != PMIX_SUCCESS) {
      PyErr_SetString(PyExc_RuntimeError, PMIx_Error_string(rc));
      goto err_cleanup;
    }

    PMIX_VALUE_RELEASE(val);
  }

  PMIX_INFO_DESTRUCT(&info[0]);
  PMIX_INFO_DESTRUCT(&info[1]);
  PMIX_PROC_DESTRUCT(&proc);
  Py_INCREF(Py_None);
  return Py_None;

err_cleanup: {
  PMIX_INFO_DESTRUCT(&info[0]);
  PMIX_INFO_DESTRUCT(&info[1]);
  PMIX_PROC_DESTRUCT(&proc);
  return NULL;
}
}

// -1. CLEAN UP
void PMIxCleanup(void) {
  if (GlobState.init == 1) {
    PMIx_Finalize(NULL, 0);
  }
}

static PyMethodDef Custom_methods[] = {
    {"get_rank", get_rank, METH_NOARGS, "Get the process rank"},
    {"get_world", get_world, METH_NOARGS, "Get the world size"},
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

  GlobState.init = 0; // set that we can init PMIX
  return m;
}
