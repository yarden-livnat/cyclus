"""Python wrapper for cyclus."""
from __future__ import division, unicode_literals, print_function
from cpython.pycapsule cimport PyCapsule_New, PyCapsule_GetPointer

# Cython imports
from libc.stdint cimport intptr_t
from libcpp.utility cimport pair as std_pair
from libcpp.set cimport set as std_set
from libcpp.map cimport map as std_map
from libcpp.vector cimport vector as std_vector
from libcpp.string cimport string as std_string
from libcpp.list cimport list as std_list
from cython.operator cimport dereference as deref
from cython.operator cimport preincrement as inc
from libc.stdlib cimport malloc, free
from libc.string cimport memcpy
from libcpp cimport bool as cpp_bool
from libcpp.cast cimport reinterpret_cast, dynamic_cast
from cpython cimport PyObject
from cpython.pycapsule cimport PyCapsule_GetPointer

from binascii import hexlify
import uuid
import os
from collections import Mapping, Sequence, Iterable, defaultdict
from importlib import import_module

cimport numpy as np
import numpy as np
import pandas as pd

# local imports

from cyclus_backend cimport cpp_cyclus_backend
from cyclus_backend.cpp_cyclus_backend cimport shared_ptr
from cyclus_backend cimport cpp_typesystem
from cyclus_backend cimport typesystem as ts
#from cyclus_backend import typesystem as ts
from cyclus_backend.typesystem cimport py_to_any, db_to_py, uuid_cpp_to_py, \
    str_py_to_cpp, std_string_to_py, std_vector_std_string_to_py, \
    bool_to_py, bool_to_cpp, int_to_py, std_set_std_string_to_py, uuid_cpp_to_py, \
    std_vector_std_string_to_py, C_IDS, blob_to_bytes, std_vector_int_to_py


# startup numpy
np.import_array()
np.import_ufunc()

cdef class _Datum:

    def __cinit__(self):
        """Constructor for Datum type conversion."""
        self._free = False
        self.ptx = NULL
        self._fields = []

    def __dealloc__(self):
        """Datum destructor."""
        if self.ptx == NULL:
            return
        cdef cpp_cyclus_backend.Datum* cpp_ptx
        if self._free:
            cpp_ptx = <cpp_cyclus_backend.Datum*> self.ptx
            del cpp_ptx
            self.ptx = NULL

    def add_val(self, field, value, shape=None, type=None):
        """Adds Datum value to current record as the corresponding cyclus data type.

        Parameters
        ----------
        field : str or bytes
            The column name.
        value : object
            Value in table column, optional
        shape : list or tuple of ints
            Length of value.
        type : dbtype or norm type
            Data type as defined by cyclus typesystem

        Returns
        -------
        self : Datum
        """
        cdef int i, n
        cdef std_vector[int] cpp_shape
        if type is None:
            raise TypeError('a database or C++ type must be supplied to add a '
                            'value to the datum, got None.')
        cdef cpp_cyclus_backend.hold_any v = py_to_any(value, type)
        if isinstance(field, str):
            field = field.encode()
        elif isinstance(field, bytes):
            pass
        else:
            raise ValueError('field name must be str or bytes.')
        # have to keep refs around so don't dealloc field names
        self._fields.append(field)
        cdef char* cpp_field = <char*> field
        if shape is None:
            (<cpp_cyclus_backend.Datum*> self.ptx).AddVal(cpp_field, v)
        else:
            n = len(shape)
            cpp_shape.resize(n)
            for i in range(n):
                cpp_shape[i] = <int> shape[i]
            (<cpp_cyclus_backend.Datum*> self.ptx).AddVal(cpp_field, v, &cpp_shape)
        return self

    def record(self):
        """Records the Datum."""
        (<cpp_cyclus_backend.Datum*> self.ptx).Record()
        self._fields.clear()

    @property
    def title(self):
        """The datum name."""
        s = (<cpp_cyclus_backend.Datum*> self.ptx).title()
        return s


class Datum(_Datum):
    """Datum class."""


cdef object query_result_to_py(cpp_cyclus_backend.QueryResult qr):
    """Converts a query result object to a dictionary mapping fields to value lists
    and a list of field names in order.
    """
    cdef int i, j
    cdef int nrows, ncols
    nrows = qr.rows.size()
    ncols = qr.fields.size()
    cdef dict res = {}
    cdef list fields = []
    for j in range(ncols):
        res[j] = []
        f = qr.fields[j]
        fields.append(f.decode())
    for i in range(nrows):
        for j in range(ncols):
            res[j].append(db_to_py(qr.rows[i][j], qr.types[j]))
    res = {fields[j]: v for j, v in res.items()}
    rtn = (res, fields)
    return rtn


cdef object single_query_result_to_py(cpp_cyclus_backend.QueryResult qr, int row):
    """Converts a query result object with only one row to a dictionary mapping
    fields to values and a list of field names in order.
    """
    cdef int i, j
    cdef int nrows, ncols
    nrows = qr.rows.size()
    ncols = qr.fields.size()
    if nrows < row:
        raise ValueError("query result does not have enough rows!")
    cdef dict res = {}
    cdef list fields = []
    for j in range(ncols):
        f = qr.fields[j]
        fields.append(f.decode())
        res[fields[j]] = db_to_py(qr.rows[row][j], qr.types[j])
    rtn = (res, fields)
    return rtn


cdef class _FullBackend:

    def __cinit__(self):
        """Full backend C++ constructor"""
        self._tables = None

    def __dealloc__(self):
        """Full backend C++ destructor."""
        # Note that we have to do it this way since self.ptx is void*
        if self.ptx == NULL:
            return
        cdef cpp_cyclus_backend.FullBackend * cpp_ptx = <cpp_cyclus_backend.FullBackend *> self.ptx
        del cpp_ptx
        self.ptx = NULL

    def query(self, table, conds=None):
        """Queries a database table.

        Parameters
        ----------
        table : str
            The table name.
        conds : iterable, optional
            A list of conditions.

        Returns
        -------
        results : pd.DataFrame
            Pandas DataFrame the represents the table
        """
        cdef std_string tab = str(table).encode()
        cdef std_string field
        cdef cpp_cyclus_backend.QueryResult qr
        cdef std_vector[cpp_cyclus_backend.Cond] cpp_conds
        cdef std_vector[cpp_cyclus_backend.Cond]* conds_ptx
        cdef std_map[std_string, cpp_cyclus_backend.DbTypes] coltypes
        # set up the conditions
        if conds is None:
            conds_ptx = NULL
        else:
            coltypes = (<cpp_cyclus_backend.FullBackend*> self.ptx).ColumnTypes(tab)
            for cond in conds:
                cond0 = cond[0].encode()
                cond1 = cond[1].encode()
                field = std_string(<const char*> cond0)
                if coltypes.count(field) == 0:
                    continue  # skips non-existent columns
                cpp_conds.push_back(cpp_cyclus_backend.Cond(field, cond1,
                    py_to_any(cond[2], coltypes[field])))
            if cpp_conds.size() == 0:
                conds_ptx = NULL
            else:
                conds_ptx = &cpp_conds
        # query, convert, and return
        qr = (<cpp_cyclus_backend.FullBackend*> self.ptx).Query(tab, conds_ptx)
        res, fields = query_result_to_py(qr)
        results = pd.DataFrame(res, columns=fields)
        return results

    def schema(self, table):
        cdef std_string ctable = str_py_to_cpp(table)
        cdef std_list[cpp_cyclus_backend.ColumnInfo] cis = (<cpp_cyclus_backend.QueryableBackend*> self.ptx).Schema(ctable)
        rtn = []
        for ci in cis:
            py_ci = ColumnInfo()
            (<_ColumnInfo> py_ci).copy_from(ci)
            rtn.append(py_ci)
        return rtn

    @property
    def tables(self):
        """Retrieves the set of tables present in the database."""
        if self._tables is not None:
            return self._tables
        cdef std_set[std_string] ctabs = \
            (<cpp_cyclus_backend.FullBackend*> self.ptx).Tables()
        cdef std_set[std_string].iterator it = ctabs.begin()
        cdef set tabs = set()
        while it != ctabs.end():
            tab = deref(it)
            tabs.add(tab.decode())
            inc(it)
        self._tables = tabs
        return self._tables

    @tables.setter
    def tables(self, value):
        self._tables = value


class FullBackend(_FullBackend, object):
    """Full backend cyclus database interface."""

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()


cdef class _SqliteBack(_FullBackend):

    def __cinit__(self, path):
        """Full backend C++ constructor"""
        cdef std_string cpp_path = str(path).encode()
        self.ptx = new cpp_cyclus_backend.SqliteBack(cpp_path)

    def __dealloc__(self):
        """Full backend C++ destructor."""
        # Note that we have to do it this way since self.ptx is void*
        if self.ptx == NULL:
            return
        cdef cpp_cyclus_backend.SqliteBack * cpp_ptx = <cpp_cyclus_backend.SqliteBack *> self.ptx
        del cpp_ptx
        self.ptx = NULL

    def flush(self):
        """Flushes the database to disk."""
        (<cpp_cyclus_backend.SqliteBack*> self.ptx).Flush()

    def close(self):
        """Closes the backend, flushing it in the process."""
        self.flush()  # just in case
        (<cpp_cyclus_backend.SqliteBack*> self.ptx).Close()

    @property
    def name(self):
        """The name of the database."""
        name = (<cpp_cyclus_backend.SqliteBack*> self.ptx).Name()
        name = name.decode()
        return name


class SqliteBack(_SqliteBack, FullBackend):
    """SQLite backend cyclus database interface."""


cdef class _Hdf5Back(_FullBackend):

    def __cinit__(self, path):
        """Hdf5 backend C++ constructor"""
        cdef std_string cpp_path = str(path).encode()
        self.ptx = new cpp_cyclus_backend.Hdf5Back(cpp_path)

    def __dealloc__(self):
        """Full backend C++ destructor."""
        # Note that we have to do it this way since self.ptx is void*
        if self.ptx == NULL:
            return
        cdef cpp_cyclus_backend.Hdf5Back * cpp_ptx = <cpp_cyclus_backend.Hdf5Back *> self.ptx
        del cpp_ptx
        self.ptx = NULL

    def flush(self):
        """Flushes the database to disk."""
        (<cpp_cyclus_backend.Hdf5Back*> self.ptx).Flush()

    def close(self):
        """Closes the backend, flushing it in the process."""
        (<cpp_cyclus_backend.Hdf5Back*> self.ptx).Close()

    @property
    def name(self):
        """The name of the database."""
        name = (<cpp_cyclus_backend.Hdf5Back*> self.ptx).Name()
        name = name.decode()
        return name


class Hdf5Back(_Hdf5Back, FullBackend):
    """HDF5 backend cyclus database interface."""


cdef class _Recorder:

    def __cinit__(self, bint inject_sim_id=True):
        """Recorder C++ constructor"""
        self.ptx = new cpp_cyclus_backend.Recorder(<cpp_bool> inject_sim_id)

    def __dealloc__(self):
        """Recorder C++ destructor."""
        if self.ptx == NULL:
            return
        self.close()
        # Note that we have to do it this way since self.ptx is void*
        cdef cpp_cyclus_backend.Recorder * cpp_ptx = <cpp_cyclus_backend.Recorder *> self.ptx
        del cpp_ptx
        self.ptx = NULL

    @property
    def dump_count(self):
        """The frequency of recording."""
        return (<cpp_cyclus_backend.Recorder*> self.ptx).dump_count()

    @dump_count.setter
    def dump_count(self, value):
        (<cpp_cyclus_backend.Recorder*> self.ptx).set_dump_count(<unsigned int> value)

    @property
    def sim_id(self):
        """The simulation id of the recorder."""
        return uuid_cpp_to_py((<cpp_cyclus_backend.Recorder*> self.ptx).sim_id())

    @property
    def inject_sim_id(self):
        """Whether or not to inject the simulation id into the tables."""
        return (<cpp_cyclus_backend.Recorder*> self.ptx).inject_sim_id()

    @inject_sim_id.setter
    def inject_sim_id(self, value):
        (<cpp_cyclus_backend.Recorder*> self.ptx).inject_sim_id(<bint> value)

    def new_datum(self, title):
        """Returns a new datum instance."""
        cdef std_string cpp_title = str_py_to_cpp(title)
        cdef _Datum d = Datum(new=False)
        (<_Datum> d).ptx = (<cpp_cyclus_backend.Recorder*> self.ptx).NewDatum(cpp_title)
        return d

    def register_backend(self, backend):
        """Registers a backend with the recorder."""
        cdef cpp_cyclus_backend.RecBackend* b
        if isinstance(backend, Hdf5Back):
            b = <cpp_cyclus_backend.RecBackend*> (
                <cpp_cyclus_backend.Hdf5Back*> (<_Hdf5Back> backend).ptx)
        elif isinstance(backend, SqliteBack):
            b = <cpp_cyclus_backend.RecBackend*> (
                <cpp_cyclus_backend.SqliteBack*> (<_SqliteBack> backend).ptx)
        elif isinstance(backend, FullBackend):
            b = <cpp_cyclus_backend.RecBackend*> ((<_FullBackend> backend).ptx)
        else:
            raise ValueError("type of backend not recognized for " +
                             str(type(backend)))
        (<cpp_cyclus_backend.Recorder*> self.ptx).RegisterBackend(b)

    def flush(self):
        """Flushes the recorder to disk."""
        (<cpp_cyclus_backend.Recorder*> self.ptx).Flush()

    def close(self):
        """Closes the recorder."""
        (<cpp_cyclus_backend.Recorder*> self.ptx).Close()


class Recorder(_Recorder, object):
    """Cyclus recorder interface."""


#
# Logger
#

# LogLevel
LEV_ERROR = cpp_cyclus_backend.LEV_ERROR
LEV_WARN = cpp_cyclus_backend.LEV_WARN
LEV_INFO1 = cpp_cyclus_backend.LEV_INFO1
LEV_INFO2 = cpp_cyclus_backend.LEV_INFO2
LEV_INFO3 = cpp_cyclus_backend.LEV_INFO3
LEV_INFO4 = cpp_cyclus_backend.LEV_INFO4
LEV_INFO5 = cpp_cyclus_backend.LEV_INFO5
LEV_DEBUG1 = cpp_cyclus_backend.LEV_INFO5
LEV_DEBUG2 = cpp_cyclus_backend.LEV_DEBUG2
LEV_DEBUG3 = cpp_cyclus_backend.LEV_DEBUG3
LEV_DEBUG4 = cpp_cyclus_backend.LEV_DEBUG4
LEV_DEBUG5 = cpp_cyclus_backend.LEV_DEBUG5


cdef class _Logger:

    @property
    def report_level(self):
        """Use to get/set the (global) log level report cutoff."""
        cdef cpp_cyclus_backend.LogLevel cpp_rtn = cpp_cyclus_backend.Logger.ReportLevel()
        rtn = int_to_py(cpp_rtn)
        return rtn

    @report_level.setter
    def report_level(self, int level):
        cpp_cyclus_backend.Logger.SetReportLevel(<cpp_cyclus_backend.LogLevel> level)

    @property
    def no_agent(self):
        """Set whether or not agent/agent log entries should be printed"""
        cdef cpp_bool cpp_rtn = cpp_cyclus_backend.Logger.NoAgent()
        rtn = bool_to_py(cpp_rtn)
        return rtn

    @no_agent.setter
    def no_agent(self, bint na):
        cpp_cyclus_backend.Logger.SetNoAgent(na)

    @property
    def no_mem(self):
        cdef cpp_bool cpp_rtn = cpp_cyclus_backend.Logger.NoMem()
        rtn = bool_to_py(cpp_rtn)
        return rtn

    @no_mem.setter
    def no_mem(self, bint nm):
        cpp_cyclus_backend.Logger.SetNoMem(nm)

    @staticmethod
    def to_log_level(text):
        """Converts a string into a corresponding LogLevel value.

        For strings that do not correspond to any particular LogLevel enum value
        the method returns the LogLevel value `LEV_ERROR`.  This method is
        primarily intended for translating command line verbosity argument(s) in
        appropriate report levels.  LOG(level) statements
        """
        cdef std_string cpp_text = str_py_to_cpp(text)
        cdef cpp_cyclus_backend.LogLevel cpp_rtn = cpp_cyclus_backend.Logger.ToLogLevel(cpp_text)
        rtn = <int> cpp_rtn
        return rtn

    @staticmethod
    def to_string(level):
        """Converts a LogLevel enum value into a corrsponding string.

        For a level argments that have no corresponding string value, the string
        `BAD_LEVEL` is returned.  This method is primarily intended for translating
        LOG(level) statement levels into appropriate strings for output to stdout.
        """
        cdef cpp_cyclus_backend.LogLevel cpp_level = <cpp_cyclus_backend.LogLevel> level
        cdef std_string cpp_rtn = cpp_cyclus_backend.Logger.ToString(cpp_level)
        rtn = std_string_to_py(cpp_rtn)
        return rtn


class Logger(_Logger):
    """A logging tool providing finer grained control over standard output
    for debugging and other purposes.
    """

#
# Errors
#
def get_warn_limit():
    """Returns the current warning limit."""
    wl = cpp_cyclus_backend.warn_limit
    return wl


def set_warn_limit(unsigned int wl):
    """Sets the warning limit."""
    cpp_cyclus_backend.warn_limit = wl


def get_warn_as_error():
    """Returns the current value for wether warnings should be treated
    as errors.
    """
    wae = bool_to_py(cpp_cyclus_backend.warn_as_error)
    return wae


def set_warn_as_error(bint wae):
    """Sets whether warnings should be treated as errors."""
    cpp_cyclus_backend.warn_as_error = wae



cdef class _ColumnInfo:
    def __cinit__(self):
        self.ptx = NULL

    cdef void copy_from(self, cpp_cyclus_backend.ColumnInfo cinfo):
        self.ptx = new cpp_cyclus_backend.ColumnInfo(cinfo.table, cinfo.col, cinfo.index,
                                             cinfo.dbtype, cinfo.shape)

    def __dealloc__(self):
        if self.ptx == NULL:
            pass
        else:
            del self.ptx

    def __repr__(self):
        s = 'ColumnInfo(table=' + self.table + ', col=' + self.col + ', index='\
            + str(self.index) + ', dbtype=' + str(self.dbtype) + ', shape=' + repr(self.shape) + ')'
        return s

    @property
    def table(self):
        table = std_string_to_py(self.ptx.table)
        return table

    @property
    def col(self):
        col = std_string_to_py(self.ptx.col)
        return col

    @property
    def index(self):
        return self.ptx.index

    @property
    def dbtype(self):
        return self.ptx.dbtype

    @property
    def shape(self):
        shape = std_vector_int_to_py(self.ptx.shape)
        return shape

class ColumnInfo(_ColumnInfo):
    """Python wrapper for ColumnInfo"""

