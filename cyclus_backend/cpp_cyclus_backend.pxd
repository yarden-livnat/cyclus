"""C++ header wrapper for specific parts of cyclus."""
from libc.stdint cimport uint64_t
from libcpp.map cimport map
from libcpp.set cimport set
from libcpp.vector cimport vector
from libcpp.list cimport list
from libcpp.utility cimport pair
from libcpp.string cimport string as std_string
from libcpp cimport bool as cpp_bool
from libcpp.typeinfo cimport type_info

# we use boost shared_ptrs
#from libcpp.memory cimport shared_ptr

from .cpp_typesystem cimport DbTypes


cdef extern from "<functional>" namespace "std":

    cdef cppclass function[T]:
        function()
        function(T)

cdef extern from "cyclus.h" namespace "boost::spirit":

    cdef cppclass hold_any:
        hold_any() except +
        hold_any(const char*) except +
        hold_any assign[T](T) except +
        T cast[T]() except +
        const type_info& type() except +

cdef extern from "cyclus.h" namespace "boost::uuids":

    cdef cppclass uuid:
        unsigned char data[16]


cdef extern from "cyclus.h" namespace "boost":

    cdef cppclass shared_ptr[T]:
        shared_ptr()
        shared_ptr(T*)
        T* get()
        T& operator*()
        cpp_bool unique()
        long use_count()
        swap(shared_ptr&)

    shared_ptr[T] reinterpret_pointer_cast[T,U](shared_ptr[U])

cdef extern from "version.h" namespace "cyclus::version":

    const char* describe() except +
    const char* core() except +
    const char* boost() except +
    const char* sqlite3() except +
    const char* hdf5() except +
    const char* xml2() except +
    const char* xmlpp() except +
    const char* coincbc() except +
    const char* coinclp() except +


cdef extern from "cyc_limits.h" namespace "cyclus":

    cdef double eps()
    cdef double eps_rsrc()


cdef extern from "cyclus.h" namespace "cyclus":

    cdef cppclass Datum:
        ctypedef pair[const char*, hold_any] Entry
        ctypedef vector[Entry] Vals
        ctypedef vector[int] Shape
        ctypedef vector[Shape] Shapes
        ctypedef vector[std_string] Fields

        Datum* AddVal(const char*, hold_any) except +
        Datum* AddVal(const char*, hold_any, vector[int]*) except +
        Datum* AddVal(std_string, hold_any) except +
        Datum* AddVal(std_string, hold_any, vector[int]*) except +
        void Record() except +
        std_string title() except +
        vector[Entry] vals() except +
        vector[vector[int]] shapes() except +
        vector[std_string] fields() except +


cdef extern from "rec_backend.h" namespace "cyclus":

    ctypedef vector[Datum*] DatumList

    cdef cppclass RecBackend:
        void Notify(DatumList) except +
        std_string Name() except +
        void Flush() except +
        void Close() except +


cdef extern from "cyclus.h" namespace "cyclus":

    ctypedef vector[hold_any] QueryRow

    cdef enum CmpOpCode:
        LT
        GT
        LE
        GE
        EQ
        NE

    cdef cppclass Blob:
        Blob() except +
        Blob(std_string) except +
        const std_string str() except +

    cdef cppclass Cond:
        Cond() except +
        Cond(std_string, std_string, hold_any) except +

        std_string field
        std_string op
        CmpOpCode opcode
        hold_any val

    cdef cppclass QueryResult:
        QueryResult() except +

        void Reset() except +
        T GetVal[T](std_string) except +
        T GetVal[T](std_string, int) except +

        vector[std_string] fields
        vector[DbTypes] types
        vector[QueryRow] rows

    cdef cppclass ColumnInfo:
        ColumnInfo()
        ColumnInfo(std_string, std_string, int, DbTypes, vector[int])
        std_string table
        std_string col
        int index
        DbTypes dbtype
        vector[int] shape

    cdef cppclass QueryableBackend:
        QueryResult Query(std_string, vector[Cond]*) except +
        map[std_string, DbTypes] ColumnTypes(std_string) except +
        list[ColumnInfo] Schema(std_string)
        set[std_string] Tables() except +

    cdef cppclass FullBackend(QueryableBackend, RecBackend):
        FullBackend() except +

    cdef cppclass Recorder:
        Recorder() except +
        Recorder(cpp_bool) except +

        unsigned int dump_count() except +
        void set_dump_count(unsigned int) except +
        cpp_bool inject_sim_id() except +
        void inject_sim_id(cpp_bool) except +
        uuid sim_id() except +
        Datum* NewDatum(std_string)
        void RegisterBackend(RecBackend*) except +
        void Flush() except +
        void Close() except +


cdef extern from "sqlite_back.h" namespace "cyclus":

    cdef cppclass SqliteBack(FullBackend):
        SqliteBack(std_string) except +


cdef extern from "hdf5_back.h" namespace "cyclus":

    cdef cppclass Hdf5Back(FullBackend):
        Hdf5Back(std_string) except +


cdef extern from "logger.h" namespace "cyclus":

    cdef enum LogLevel:
        LEV_ERROR
        LEV_WARN
        LEV_INFO1
        LEV_INFO2
        LEV_INFO3
        LEV_INFO4
        LEV_INFO5
        LEV_DEBUG1
        LEV_DEBUG2
        LEV_DEBUG3
        LEV_DEBUG4
        LEV_DEBUG5

    cdef cppclass Logger:
        Logger() except +
        @staticmethod
        LogLevel& ReportLevel() except +
        @staticmethod
        void SetReportLevel(LogLevel) except +
        @staticmethod
        cpp_bool& NoAgent() except +
        @staticmethod
        void SetNoAgent(cpp_bool) except +
        @staticmethod
        cpp_bool& NoMem() except +
        @staticmethod
        void SetNoMem(cpp_bool) except +
        @staticmethod
        LogLevel ToLogLevel(std_string) except +
        @staticmethod
        std_string ToString(LogLevel) except +


cdef extern from "error.h" namespace "cyclus":

    cdef unsigned int warn_limit
    cdef cpp_bool warn_as_error




