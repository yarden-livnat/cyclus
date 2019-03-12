"""Python wrapper for cyclus."""
# Cython imports
from libcpp.utility cimport pair as cpp_pair
from libcpp.set cimport set as cpp_set
from libcpp.map cimport map as cpp_map
from libcpp.vector cimport vector as cpp_vector
from libcpp.string cimport string as std_string
from cython.operator cimport dereference as deref
from cython.operator cimport preincrement as inc
from libc.stdlib cimport malloc, free
from libcpp cimport bool as cpp_bool
from cpython cimport PyObject

# local imports
from cyclus_backend cimport cpp_cyclus_backend


cdef class _Datum:
    cdef void * ptx
    cdef bint _free
    cdef list _fieldnames

cdef object query_result_to_py(cpp_cyclus_backend.QueryResult)
cdef object single_query_result_to_py(cpp_cyclus_backend.QueryResult qr, int row)

cdef class _FullBackend:
    cdef void * ptx

cdef class _SqliteBack(_FullBackend):
    pass

cdef class _Hdf5Back(_FullBackend):
    pass

cdef class _Recorder:
    cdef void * ptx


cdef class _Logger:
    # everything that we use on this class is static
    pass

cdef class _ColumnInfo:
    cdef cpp_cyclus_backend.ColumnInfo * ptx
    cdef void copy_from(_ColumnInfo, cpp_cyclus_backend.ColumnInfo)


