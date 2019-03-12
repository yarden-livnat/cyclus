#!/usr/bin/env python
"""Generates Cyclus Type System bindings.

Module history:

- 2016-10-12: scopatz: This file used to be called genapi.py in cymetric.
"""
from __future__ import print_function, unicode_literals

import io
import os
import sys
import imp
import json
import argparse
import platform
import warnings
import itertools
import subprocess
from glob import glob
from distutils import core, dir_util
from pprint import pprint, pformat
from collections import defaultdict
if sys.version_info[0] > 2:
    from urllib.request import urlopen
    str_types = (str, bytes)
    unicode_types = (str,)
else:
    from urllib2 import urlopen
    str_types = (str, unicode)
    unicode_types = (str, unicode)

import jinja2

#
# Type System
#

class TypeSystem(object):
    """A type system for cyclus code generation."""

    def __init__(self, table, cycver, rawver=None, cpp_typesystem='cpp_typesystem'):
        """Parameters
        ----------
        table : list
            A table of possible types. The first row must be the column names.
        cycver : tuple of ints
            Cyclus version number.
        rawver : string, optional
            A full, raw version string, if available
        cpp_typesystem : str, optional
            The namespace of the C++ wrapper header.

        Attributes
        ----------
        table : list
            A stripped down table of type information.
        cols : dict
            Maps column names to column number in table.
        cycver : tuple of ints
            Cyclus version number.
        verstr : str
            A version string of the format 'vX.X'.
        types : set of str
            The type names in the type system.
        ids : dict
            Maps types to integer identifier.
        cpptypes : dict
            Maps types to C++ type.
        ranks : dict
            Maps types to shape rank.
        norms : dict
            Maps types to programmatic normal form, ie INT -> 'int' and
            VECTOR_STRING -> ('std::vector', 'std::string').
        dbtypes : list of str
            The type names in the type system, sorted by id.
        uniquetypes : list of str
            The type names in the type system, sorted by id,
            which map to a unique C++ type.
        """
        self.cpp_typesystem = cpp_typesystem
        self.cycver = cycver
        self.rawver = rawver
        self.verstr = verstr = 'v{0}.{1}'.format(*cycver)
        self.cols = cols = {x: i for i, x in enumerate(table[0])}
        id, name, version = cols['id'], cols['name'], cols['version']
        cpptype, rank = cols['C++ type'], cols['shape rank']
        tab = []
        if rawver is not None:
            tab = [row for row in table if row[version] == rawver]
        if len(tab) == 0:
            tab = [row for row in table if row[version].startswith(verstr)]
            if len(tab) == 0:
                raise ValueError("Cyclus version could not be found in table!")
        self.table = table = tab
        self.types = types = set()
        self.ids = ids = {}
        self.cpptypes = cpptypes = {}
        self.ranks = ranks = {}
        for row in table:
            if (("RES_MAP" not in row[name]) and 
                    ("RESOURCE_BUFF" not in row[name]) and 
                    ("MATERIAL" not in row[name]) and 
                    ("RES_BUF" not in row[name]) and 
                    ("PRODUCT" not in row[name])):
                t = row[name]
                types.add(t)
                ids[t] = row[id]
                cpptypes[t] = row[cpptype]
                ranks[t] = row[rank]
        self.norms = {t: parse_template(c) for t, c in cpptypes.items()}
        self.dbtypes = sorted(ids.keys(), key=lambda t: ids[t])
        # find unique types
        seen = set()
        self.uniquetypes = uniquetypes = []
        for t in self.dbtypes:
            normt = self.norms[t]
            if normt in seen:
                continue
            else:
                uniquetypes.append(t)
                seen.add(normt)
        self.resources = tuple(RESOURCES)
        self.inventory_types = inventory_types = []
        for t in uniquetypes:
            normt = self.norms[t]
            if normt in INVENTORIES or normt[0] in INVENTORIES:
                inventory_types.append(t)

        # caches
        self._cython_cpp_name = {}
        self._cython_types = dict(CYTHON_TYPES)
        self._shared_ptrs = {}
        self._funcnames = dict(FUNCNAMES)
        self._classnames = dict(CLASSNAMES)
        self._vars_to_py = dict(VARS_TO_PY)
        self._vars_to_cpp = dict(VARS_TO_CPP)
        self._nptypes = dict(NPTYPES)
        self._new_py_insts = dict(NEW_PY_INSTS)
        self._to_py_converters = dict(TO_PY_CONVERTERS)
        self._to_cpp_converters = dict(TO_CPP_CONVERTERS)
        self._use_shared_ptr = defaultdict(lambda: False,
                                           {k: True for k in USE_SHARED_PTR})


    def cython_cpp_name(self, t):
        """Returns the C++ name of the type, eg INT -> cpp_typesystem.INT."""
        if t not in self._cython_cpp_name:
            self._cython_cpp_name[t] = '{0}.{1}'.format(self.cpp_typesystem, t)
        return self._cython_cpp_name[t]

    def cython_type(self, t):
        """Returns the Cython spelling of the type."""
        if t in self._cython_types:
            return self._cython_types[t]
        if isinstance(t, str_types):
            n = self.norms[t]
            return self.cython_type(n)
        # must be teplate type
        cyt = list(map(self.cython_type, t))
        cyt = '{0}[{1}]'.format(cyt[0], ', '.join(cyt[1:]))
        self._cython_types[t] = cyt
        return cyt

    def possibly_shared_cython_type(self, t):
        """Returns the Cython type, or if it is a shared pointer type,
        return the shared pointer version.
        """
        if self._use_shared_ptr[t]:
            cyt = self._shared_ptrs.get(t, None)
            if cyt is None:
                self._shared_ptrs[t] = 'shared_ptr[' + self.cython_type(t) + ']'
                cyt = self._shared_ptrs[t]
        else:
            cyt = self.cython_type(t)
        return cyt

    def funcname(self, t):
        """Returns a version of the type name suitable for use in a function name.
        """
        if t in self._funcnames:
            return self._funcnames[t]
        if isinstance(t, str_types):
            n = self.norms[t]
            return self.funcname(n)
        f = '_'.join(map(self.funcname, t))
        self._funcnames[t] = f
        return f

    def classname(self, t):
        """Returns a version of the type name suitable for use in a class name.
        """
        if t in self._classnames:
            return self._classnames[t]
        if isinstance(t, str_types):
            n = self.norms[t]
            return self.classname(n)
        c = ''.join(map(self.classname, t))
        self._classnames[t] = c
        return c

    def var_to_py(self, x, t):
        """Returns an expression for converting an object to Python."""
        n = self.norms.get(t, t)
        expr = self._vars_to_py.get(n, None)
        if expr is None:
            f = self.funcname(t)
            expr = f + '_to_py({var})'
            self._vars_to_py[n] = expr
        return expr.format(var=x)

    def hold_any_to_py(self, x, t):
        """Returns an expression for converting a hold_any object to Python."""
        cyt = self.possibly_shared_cython_type(t)
        cast = '{0}.cast[{1}]()'.format(x, cyt)
        return self.var_to_py(cast, t)

    def var_to_cpp(self, x, t):
        """Returns an expression for converting a Python object to C++."""
        n = self.norms.get(t, t)
        expr = self._vars_to_cpp.get(n, None)
        if expr is None:
            f = self.funcname(t)
            expr = f + '_to_cpp({var})'
            self._vars_to_cpp[n] = expr
        return expr.format(var=x)

    def py_to_any(self, a, val, t):
        """Returns an expression for assigning a Python object (val) to an any
        object (a)."""
        cyt = self.possibly_shared_cython_type(t)
        cpp = self.var_to_cpp(val, t)
        rtn = '{a}.assign[{cyt}]({cpp})'.format(a=a, cyt=cyt, cpp=cpp)
        return rtn

    def nptype(self, n):
        """Returns the numpy type for a normal form element."""
        npt = self._nptypes.get(n, None)
        if npt is None:
            npt = 'np.NPY_OBJECT'
            self._nptypes[n] = npt
        return npt

    def new_py_inst(self, t):
        """Returns the new instance for the type."""
        n = self.norms.get(t, t)
        inst = self._new_py_insts.get(n, None)
        if inst is None:
            # look up base type
            inst = self._new_py_insts.get(n[0], None)
        if inst is None:
            inst = 'None'
            self._new_py_insts[n] = inst
        return inst

    def convert_to_py(self, x, t):
        """Converts a C++ variable to python.

        Parameters
        ----------
        x : str
            variable name
        t : str
            variable type

        Returns
        -------
        decl : str
            Declarations needed for conversion, may be many lines.
        body : str
            Actual conversion implementation, may be many lines.
        rtn : str
            Return expression.
        """
        n = self.norms.get(t, t)
        ctx = {'type': self.cython_type(t), 'var': x, 'nptypes': [],
               'classname': self.classname(t), 'funcname': self.funcname(t)}
        if n in self._to_py_converters:
            # basic type or str
            n0 = ()
            decl, body, expr = self._to_py_converters[n]
        elif n[0] == 'std::vector' and self.nptype(n[1]) == 'np.NPY_OBJECT':
            # vector of type that should become an object
            n0 = n[0]
            decl, body, expr = self._to_py_converters['np.ndarray', 'np.NPY_OBJECT']
            ctx['elem_to_py'] = self.var_to_py(x + '[i]', n[1])
        else:
            # must be a template already
            n0 = n[0]
            decl, body, expr = self._to_py_converters[n0]
        for targ, n_i in zip(TEMPLATE_ARGS.get(n0, ()), n[1:]):
            x_i = x + '_' +  targ
            ctx[targ+'name'] = x_i
            ctx[targ+'type'] = self.cython_type(n_i)
            dbe_i = self.convert_to_py(x_i, n_i)
            dbe_i = map(Indenter, dbe_i)
            ctx[targ+'decl'], ctx[targ+'body'], ctx[targ+'expr'] = dbe_i
            ctx['nptypes'].append(self.nptype(n_i))
        decl = decl.format(**ctx)
        body = body.format(**ctx)
        expr = expr.format(**ctx)
        return decl, body, expr

    def convert_to_cpp(self, x, t):
        """Converts a Python variable to C++.

        Parameters
        ----------
        x : str
            variable name
        t : str
            variable type

        Returns
        -------
        decl : str
            Declarations needed for conversion, may be many lines.
        body : str
            Actual conversion implementation, may be many lines.
        rtn : str
            Return expression.
        """
        n = self.norms.get(t, t)
        ctx = {'type': self.cython_type(t), 'var': x, 'nptypes': [],
               'classname': self.classname(t), 'funcname': self.funcname(t)}
        if n in self._to_cpp_converters:
            # basic type or str
            n0 = ()
            decl, body, expr = self._to_cpp_converters[n]
        elif n[0] == 'std::vector' and self.nptype(n[1]) in ('np.NPY_OBJECT',
                                                             'np.NPY_BOOL'):
            # vector of type that should become an object
            n0 = n[0]
            decl, body, expr = self._to_cpp_converters['np.ndarray', 'np.NPY_OBJECT']
        else:
            # must be a template already
            n0 = n[0]
            decl, body, expr = self._to_cpp_converters[n0]
        for targ, n_i in zip(TEMPLATE_ARGS.get(n0, ()), n[1:]):
            x_i = x + '_' +  targ
            ctx[targ+'name'] = x_i
            ctx[targ+'type'] = self.cython_type(n_i)
            dbe_i = self.convert_to_cpp(x_i, n_i)
            dbe_i = map(Indenter, dbe_i)
            ctx[targ+'decl'], ctx[targ+'body'], ctx[targ+'expr'] = dbe_i
            ctx['nptypes'].append(self.nptype(n_i))
            ctx[targ+'_to_cpp'] = self.var_to_cpp(x_i, n_i)
        decl = decl.format(**ctx)
        body = body.format(**ctx)
        expr = expr.format(**ctx)
        return decl, body, expr


CYTHON_TYPES = {
    # type system types
    'BOOL': 'cpp_bool',
    'INT': 'int',
    'FLOAT': 'float',
    'DOUBLE': 'double',
    'STRING': 'std_string',
    'VL_STRING': 'std_string',
    'BLOB': 'cpp_cyclus_backend.Blob',
    'UUID': 'cpp_cyclus_backend.uuid',
    'MATERIAL': 'cpp_cyclus_backend.Material',
    'PRODUCT': 'cpp_cyclus_backend.Product',
    'RESOURCE_BUFF': 'cpp_cyclus_backend.ResourceBuff',
    # C++ normal types
    'bool': 'cpp_bool',
    'int': 'int',
    'float': 'float',
    'double': 'double',
    'std::string': 'std_string',
    'std::string': 'std_string',
    'cyclus::Blob': 'cpp_cyclus_backend.Blob',
    'boost::uuids::uuid': 'cpp_cyclus_backend.uuid',
    'cyclus::Material': 'cpp_cyclus_backend.Material',
    'cyclus::Product': 'cpp_cyclus_backend.Product',
    'cyclus::toolkit::ResourceBuff': 'cpp_cyclus_backend.ResourceBuff',
    # Template Types
    'std::set': 'std_set',
    'std::map': 'std_map',
    'std::pair': 'std_pair',
    'std::list': 'std_list',
    'std::vector': 'std_vector',
    'cyclus::toolkit::ResBuf': 'cpp_cyclus_backend.ResBuf',
    'cyclus::toolkit::ResMap': 'cpp_cyclus_backend.ResMap',
    }

# Don't include the base resource class here since it is pure virtual.
RESOURCES = ['MATERIAL', 'PRODUCT']

INVENTORIES = ['cyclus::toolkit::ResourceBuff', 'cyclus::toolkit::ResBuf',
               'cyclus::toolkit::ResMap']

USE_SHARED_PTR = ('MATERIAL', 'PRODUCT', 'cyclus::Material', 'cyclus::Product')

FUNCNAMES = {
    # type system types
    'BOOL': 'bool',
    'INT': 'int',
    'FLOAT': 'float',
    'DOUBLE': 'double',
    'STRING': 'std_string',
    'VL_STRING': 'std_string',
    'BLOB': 'blob',
    'UUID': 'uuid',
    'MATERIAL': 'material',
    'PRODUCT': 'product',
    'RESOURCE_BUFF': 'resource_buff',
    # C++ normal types
    'bool': 'bool',
    'int': 'int',
    'float': 'float',
    'double': 'double',
    'std::string': 'std_string',
    'cyclus::Blob': 'blob',
    'boost::uuids::uuid': 'uuid',
    'cyclus::Material': 'material',
    'cyclus::Product': 'product',
    'cyclus::toolkit::ResourceBuff': 'resource_buff',
    # Template Types
    'std::set': 'std_set',
    'std::map': 'std_map',
    'std::pair': 'std_pair',
    'std::list': 'std_list',
    'std::vector': 'std_vector',
    'cyclus::toolkit::ResBuf': 'res_buf',
    'cyclus::toolkit::ResMap': 'res_map',
    }

CLASSNAMES = {
    # type system types
    'BOOL': 'Bool',
    'INT': 'Int',
    'FLOAT': 'Float',
    'DOUBLE': 'Double',
    'STRING': 'String',
    'VL_STRING': 'String',
    'BLOB': 'Blob',
    'UUID': 'Uuid',
    'MATERIAL': 'Material',
    'PRODUCT': 'Product',
    'RESOURCE_BUFF': 'ResourceBuff',
    # C++ normal types
    'bool': 'Bool',
    'int': 'Int',
    'float': 'Float',
    'double': 'Double',
    'std::string': 'String',
    'cyclus::Blob': 'Blob',
    'boost::uuids::uuid': 'Uuid',
    'cyclus::Material': 'Material',
    'cyclus::Product': 'Product',
    'cyclus::toolkit::ResourceBuff': 'ResourceBuff',
    # Template Types
    'std::set': 'Set',
    'std::map': 'Map',
    'std::pair': 'Pair',
    'std::list': 'List',
    'std::vector': 'Vector',
    'cyclus::toolkit::ResBuf': 'ResBuf',
    'cyclus::toolkit::ResMap': 'ResMap',
    }



# note that this maps normal forms to python
VARS_TO_PY = {
    'bool': '{var}',
    'int': '{var}',
    'float': '{var}',
    'double': '{var}',
    'std::string': 'bytes({var}).decode()',
    'cyclus::Blob': 'blob_to_bytes({var})',
    'boost::uuids::uuid': 'uuid_cpp_to_py({var})',
    'cyclus::toolkit::ResourceBuff': 'None',
    }

# note that this maps normal forms to python
VARS_TO_CPP = {
    'bool': '<bint> {var}',
    'int': '<int> {var}',
    'float': '<float> {var}',
    'double': '<double> {var}',
    'std::string': 'str_py_to_cpp({var})',
    'cyclus::Blob': 'cpp_cyclus_backend.Blob(std_string(<const char*> {var}))',
    'boost::uuids::uuid': 'uuid_py_to_cpp({var})',
    'cyclus::toolkit::ResourceBuff': 'resource_buff_to_cpp({var})',
    }

TEMPLATE_ARGS = {
    'std::set': ('val',),
    'std::map': ('key', 'val'),
    'std::pair': ('first', 'second'),
    'std::list': ('val',),
    'std::vector': ('val',),
    'cyclus::toolkit::ResBuf': ('val',),
    'cyclus::toolkit::ResMap': ('key', 'val'),
    }

NPTYPES = {
    'bool': 'np.NPY_BOOL',
    'int': 'np.NPY_INT32',
    'float': 'np.NPY_FLOAT32',
    'double': 'np.NPY_FLOAT64',
    'std::string': 'np.NPY_OBJECT',
    'cyclus::Blob': 'np.NPY_OBJECT',
    'boost::uuids::uuid': 'np.NPY_OBJECT',
    'cyclus::Material': 'np.NPY_OBJECT',
    'cyclus::Product': 'np.NPY_OBJECT',
    'cyclus::toolkit::ResourceBuff': 'None',
    'std::set': 'np.NPY_OBJECT',
    'std::map': 'np.NPY_OBJECT',
    'std::pair': 'np.NPY_OBJECT',
    'std::list': 'np.NPY_OBJECT',
    'std::vector': 'np.NPY_OBJECT',
    'cyclus::toolkit::ResBuf': 'np.NPY_OBJECT',
    'cyclus::toolkit::ResMap': 'np.NPY_OBJECT',
    }

NEW_PY_INSTS = {
    'bool': 'False',
    'int': '0',
    'float': '0.0',
    'double': '0.0',
    'std::string': '""',
    'cyclus::Blob': 'b""',
    'boost::uuids::uuid': 'uuid.UUID(int=0)',
    'cyclus::Material': 'None',
    'cyclus::Product': 'None',
    'cyclus::toolkit::ResourceBuff': 'None',
    'std::set': 'set()',
    'std::map': '{}',
    'std::pair': '(None, None)',
    'std::list': '[]',
    'std::vector': '[]',
    'cyclus::toolkit::ResBuf': 'None',
    'cyclus::toolkit::ResMap': 'None',
    }

# note that this maps normal forms to python
TO_PY_CONVERTERS = {
    # base types
    'bool': ('', '', '{var}'),
    'int': ('', '', '{var}'),
    'float': ('', '', '{var}'),
    'double': ('', '', '{var}'),
    'std::string': ('\n', '\npy{var} = {var}\npy{var} = py{var}.decode()\n',
                    'py{var}'),
    'cyclus::Blob': ('', '', 'blob_to_bytes({var})'),
    'boost::uuids::uuid': ('', '', 'uuid_cpp_to_py({var})'),
    'cyclus::Material': (
        'cdef _Material pyx_{var}',
        'pyx_{var} = Material()\n'
        'pyx_{var}.ptx = cpp_cyclus_backend.reinterpret_pointer_cast[cpp_cyclus_backend.Resource, '
                            'cpp_cyclus_backend.Material]({var})\n'
        'py_{var} = pyx_{var}\n',
        'py_{var}'),
    'cyclus::Product': (
        'cdef _Product pyx_{var}',
        'pyx_{var} = Product()\n'
        'pyx_{var}.ptx = cpp_cyclus_backend.reinterpret_pointer_cast[cpp_cyclus_backend.Resource, '
                            'cpp_cyclus_backend.Product]({var})\n'
        'py_{var} = pyx_{var}\n',
        'py_{var}'),
    'cyclus::toolkit::ResourceBuff': ('', '', 'None'),
    # templates
    'std::set': (
        '{valdecl}\n'
        'cdef {valtype} {valname}\n'
        'cdef std_set[{valtype}].iterator it{var}\n'
        'cdef set py{var}\n',
        'py{var} = set()\n'
        'it{var} = {var}.begin()\n'
        'while it{var} != {var}.end():\n'
        '    {valname} = deref(it{var})\n'
        '    {valbody.indent4}\n'
        '    pyval = {valexpr}\n'
        '    py{var}.add(pyval)\n'
        '    inc(it{var})\n',
        'py{var}'),
    'std::map': (
        '{keydecl}\n'
        '{valdecl}\n'
        'cdef {keytype} {keyname}\n'
        'cdef {valtype} {valname}\n'
        'cdef {type}.iterator it{var}\n'
        'cdef dict py{var}\n',
        'py{var} = {{}}\n'
        'it{var} = {var}.begin()\n'
        'while it{var} != {var}.end():\n'
        '    {keyname} = deref(it{var}).first\n'
        '    {keybody.indent4}\n'
        '    pykey = {keyexpr}\n'
        '    {valname} = deref(it{var}).second\n'
        '    {valbody.indent4}\n'
        '    pyval = {valexpr}\n'
        '    pykey = {keyexpr}\n'
        '    py{var}[pykey] = pyval\n'
        '    inc(it{var})\n',
        'py{var}'),
    'std::pair': (
        '{firstdecl}\n'
        '{seconddecl}\n'
        'cdef {firsttype} {firstname}\n'
        'cdef {secondtype} {secondname}\n',
        '{firstname} = {var}.first\n'
        '{firstbody}\n'
        'pyfirst = {firstexpr}\n'
        '{secondname} = {var}.second\n'
        '{secondbody}\n'
        'pyfirst = {firstexpr}\n'
        'pysecond = {secondexpr}\n'
        'py{var} = (pyfirst, pysecond)\n',
        'py{var}'),
    'std::list': (
        '{valdecl}\n'
        'cdef {valtype} {valname}\n'
        'cdef std_list[{valtype}].iterator it{var}\n'
        'cdef list py{var}\n',
        'py{var} = []\n'
        'it{var} = {var}.begin()\n'
        'while it{var} != {var}.end():\n'
        '    {valname} = deref(it{var})\n'
        '    {valbody.indent4}\n'
        '    pyval = {valexpr}\n'
        '    py{var}.append(pyval)\n'
        '    inc(it{var})\n',
        'py{var}'),
    'std::vector': (
        'cdef np.npy_intp {var}_shape[1]\n',
        '{var}_shape[0] = <np.npy_intp> {var}.size()\n'
        'py{var} = np.PyArray_SimpleNewFromData(1, {var}_shape, {nptypes[0]}, '
            '&{var}[0])\n'
        'py{var} = np.PyArray_Copy(py{var})\n',
        'py{var}'),
    ('std::vector', 'bool'): (
        'cdef int i\n'
        'cdef np.npy_intp {var}_shape[1]\n',
        '{var}_shape[0] = <np.npy_intp> {var}.size()\n'
        'py{var} = np.PyArray_SimpleNew(1, {var}_shape, np.NPY_BOOL)\n'
        'for i in range({var}_shape[0]):\n'
        '    py{var}[i] = {var}[i]\n',
        'py{var}'),
    ('np.ndarray', 'np.NPY_OBJECT'): (
        'cdef int i\n'
        'cdef np.npy_intp {var}_shape[1]\n',
        '{var}_shape[0] = <np.npy_intp> {var}.size()\n'
        'py{var} = np.PyArray_SimpleNew(1, {var}_shape, np.NPY_OBJECT)\n'
        'for i in range({var}_shape[0]):\n'
        '    {var}_i = {elem_to_py}\n'
        '    py{var}[i] = {var}_i\n',
        'py{var}'),
    'cyclus::toolkit::ResBuf': ('', '', 'None'),
    'cyclus::toolkit::ResMap': ('', '', 'None'),
    }

TO_CPP_CONVERTERS = {
    # base types
    'bool': ('', '', '<bint> {var}'),
    'int': ('', '', '<int> {var}'),
    'float': ('', '', '<float> {var}'),
    'double': ('', '', '<double> {var}'),
    'std::string': ('cdef bytes b_{var}',
        'if isinstance({var}, str):\n'
        '   b_{var} = {var}.encode()\n'
        'elif isinstance({var}, str):\n'
        '   b_{var} = {var}\n'
        'else:\n'
        '   b_{var} = bytes({var})\n',
        'std_string(<const char*> b_{var})'),
    'cyclus::Blob': ('', '', 'cpp_cyclus_backend.Blob(std_string(<const char*> {var}))'),
    'boost::uuids::uuid': ('', '', 'uuid_py_to_cpp({var})'),
    'cyclus::Material': (
        'cdef _Material py{var}\n'
        'cdef shared_ptr[cpp_cyclus_backend.Material] cpp{var}\n',
        'py{var} = <_Material> {var}\n'
        'cpp{var} = reinterpret_pointer_cast[cpp_cyclus_backend.Material, '
                         'cpp_cyclus_backend.Resource](py{var}.ptx)\n',
        'cpp{var}'),
    'cyclus::Product': (
        'cdef _Material py{var}\n'
        'cdef shared_ptr[cpp_cyclus_backend.Product] cpp{var}\n',
        'py{var} = <_Product> {var}\n'
        'cpp{var} = reinterpret_pointer_cast[cpp_cyclus_backend.Product, '
                         'cpp_cyclus_backend.Resource](py{var}.ptx)\n',
        'cpp{var}'),
    'cyclus::toolkit::ResourceBuff': (
        'cdef _{classname} py{var}\n'
        'cdef cpp_cyclus_backend.ResourceBuff cpp{var}\n',
        'py{var} = <_{classname}> {var}\n'
        'cpp{var} = deref(py{var}.ptx)\n',
        'cpp{var}'),
    # templates
    'std::set': (
        '{valdecl}\n'
        'cdef std_set[{valtype}] cpp{var}\n',
        'cpp{var} = std_set[{valtype}]()\n'
        'for {valname} in {var}:\n'
        '    {valbody.indent4}\n'
        '    cpp{var}.insert({valexpr})\n',
        'cpp{var}'),
    'std::map': (
        '{keydecl}\n'
        '{valdecl}\n'
        'cdef {type} cpp{var}\n',
        'cpp{var} = {type}()\n'
        'if not isinstance({var}, collections.Mapping):\n'
        '    {var} = dict({var})\n'
        'for {keyname}, {valname} in {var}.items():\n'
        '    {keybody.indent4}\n'
        '    {valbody.indent4}\n'
        '    cpp{var}[{keyexpr}] = {valexpr}\n',
        'cpp{var}'),
    'std::pair': (
        '{firstdecl}\n'
        '{seconddecl}\n'
        'cdef {type} cpp{var}\n',
        '{firstname} = {var}[0]\n'
        '{firstbody}\n'
        'cpp{var}.first = {firstexpr}\n'
        '{secondname} = {var}[1]\n'
        '{secondbody}\n'
        'cpp{var}.second = {secondexpr}\n',
        'cpp{var}'),
    'std::list': (
        '{valdecl}\n'
        'cdef std_list[{valtype}] cpp{var}\n',
        'cpp{var} = std_list[{valtype}]()\n'
        'for {valname} in {var}:\n'
        '    {valbody.indent4}\n'
        '    cpp{var}.push_back({valexpr})\n',
        'cpp{var}'),
    'std::vector': (
        'cdef int i\n'
        'cdef int {var}_size\n'
        'cdef {type} cpp{var}\n'
        'cdef {valtype} * {var}_data\n',
        'cpp{var} = {type}()\n'
        '{var}_size = len({var})\n'
        'if isinstance({var}, np.ndarray) and '
        '(<np.ndarray> {var}).descr.type_num == {nptypes[0]}:\n'
        '    {var}_data = <{valtype} *> np.PyArray_DATA(<np.ndarray> {var})\n'
        '    cpp{var}.resize(<size_t> {var}_size)\n'
        '    memcpy(<void*> &cpp{var}[0], {var}_data, sizeof({valtype}) * {var}_size)\n'
        'else:\n'
        '    cpp{var}.resize(<size_t> {var}_size)\n'
        '    for i, {valname} in enumerate({var}):\n'
        '        cpp{var}[i] = {val_to_cpp}\n',
        'cpp{var}'),
    ('np.ndarray', 'np.NPY_OBJECT'): (
        'cdef int i\n'
        'cdef int {var}_size\n'
        'cdef {type} cpp{var}\n',
        'cpp{var} = {type}()\n'
        '{var}_size = len({var})\n'
        'cpp{var}.resize(<size_t> {var}_size)\n'
        'for i, {valname} in enumerate({var}):\n'
        '    cpp{var}[i] = {val_to_cpp}\n',
        'cpp{var}'),
    'cyclus::toolkit::ResBuf': (
        'cdef _{classname} py{var}\n'
        'cdef cpp_cyclus_backend.ResBuf[{valtype}] cpp{var}\n',
        'py{var} = <_{classname}> {var}\n'
        'cpp{var} = deref(py{var}.ptx)\n',
        'cpp{var}'),
    'cyclus::toolkit::ResMap': (
        'cdef _{classname} py{var}\n'
        'cdef cpp_cyclus_backend.ResMap[{keytype}, {valtype}] cpp{var}\n',
        'py{var} = <_{classname}> {var}\n'
        'cpp{var} = deref(py{var}.ptx)\n',
        'cpp{var}'),
    }

# annotation info key (pyname), C++ name,  cython type names, init snippet
ANNOTATIONS = [
    ('name', 'name', 'object', 'None'),
    ('type', 'type', 'object', 'None'),
    ('index', 'index', 'int', '-1'),
    ('default', 'dflt', 'object', 'None'),
    ('internal', 'internal', 'bint', 'False'),
    ('shape', 'shape', 'object', 'None'),
    ('doc', 'doc', 'str', '""'),
    ('tooltip', 'tooltip', 'object', 'None'),
    ('units', 'units', 'str', '""'),
    ('userlevel', 'userlevel', 'int', '0'),
    ('alias', 'alias', 'object', 'None'),
    ('uilabel', 'uilabel', 'object', 'None'),
    ('uitype', 'uitype', 'object', 'None'),
    ('range', 'range', 'object', 'None'),
    ('categorical', 'categorical', 'object', 'None'),
    ('schematype', 'schematype', 'object', 'None'),
    ('initfromcopy', 'initfromcopy', 'str', '""'),
    ('initfromdb', 'initfromdb', 'str', '""'),
    ('infiletodb', 'infiletodb', 'str', '""'),
    ('schema', 'schema', 'str', '""'),
    ('snapshot', 'snapshot', 'str', '""'),
    ('snapshotinv', 'snapshotinv', 'str', '""'),
    ('initinv', 'initinv', 'str', '""'),
    ('uniquetypeid', 'uniquetypeid', 'int', '-1'),
    ]


def split_template_args(s, open_brace='<', close_brace='>', separator=','):
    """Takes a string with template specialization and returns a list
    of the argument values as strings. Mostly cribbed from xdress.
    """
    targs = []
    ns = s.split(open_brace, 1)[-1].rsplit(close_brace, 1)[0].split(separator)
    count = 0
    targ_name = ''
    for n in ns:
        count += n.count(open_brace)
        count -= n.count(close_brace)
        if len(targ_name) > 0:
            targ_name += separator
        targ_name += n
        if count == 0:
            targs.append(targ_name.strip())
            targ_name = ''
    return targs


def parse_template(s, open_brace='<', close_brace='>', separator=','):
    """Takes a string -- which may represent a template specialization --
    and returns the corresponding type. Mostly cribbed from xdress.
    """
    if open_brace not in s and close_brace not in s:
        return s
    t = [s.split(open_brace, 1)[0]]
    targs = split_template_args(s, open_brace=open_brace,
                                close_brace=close_brace, separator=separator)
    for targ in targs:
        t.append(parse_template(targ, open_brace=open_brace,
                                close_brace=close_brace, separator=separator))
    t = tuple(t)
    return t


class Indenter(object):
    """Handles indentations."""
    def __init__(self, s):
        """Constructor for string object."""
        self._s = s

    def __str__(self):
        """Returns a string."""
        return self._s

    def __getattr__(self, key):
        """Replaces an indentation with a newline and spaces."""
        if key.startswith('indent'):
            n = int(key[6:])
            return self._s.replace('\n', '\n' + ' '*n)
        return self.__dict__[key]

def safe_output(cmd, shell=False, *args, **kwargs):
    """Checks that a command successfully runs with/without shell=True.
    Returns the output.
    """
    try:
        out = subprocess.check_output(cmd, shell=False, *args, **kwargs)
    except (subprocess.CalledProcessError, OSError):
        cmd = ' '.join(cmd)
        out = subprocess.check_output(cmd, shell=True, *args, **kwargs)
    return out

#
# Code Generation
#

JENV = jinja2.Environment(undefined=jinja2.StrictUndefined)

CG_WARNING = """
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# !!!!! WARNING - THIS FILE HAS BEEN !!!!!!
# !!!!!   AUTOGENERATED BY CYCLUS    !!!!!!
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
""".strip()

STL_CIMPORTS = """
# Cython standard library imports
from libcpp.map cimport map as std_map
from libcpp.set cimport set as std_set
from libcpp.list cimport list as std_list
from libcpp.vector cimport vector as std_vector
from libcpp.utility cimport pair as std_pair
from libcpp.string cimport string as std_string
from libcpp.typeinfo cimport type_info
from cython.operator cimport dereference as deref
from cython.operator cimport preincrement as inc
from cython.operator cimport typeid
from libc.stdlib cimport malloc, free
from libc.string cimport memcpy
from libcpp cimport bool as cpp_bool
from libcpp.cast cimport const_cast
from libcpp.cast cimport reinterpret_cast, dynamic_cast
""".strip()

NPY_IMPORTS = """
# numpy imports & startup
cimport numpy as np
import numpy as np
np.import_array()
np.import_ufunc()
""".strip()

CPP_TYPESYSTEM = JENV.from_string("""
{{ cg_warning }}

cdef extern from "cyclus.h" namespace "cyclus":

    cdef enum DbTypes:
        {{ dbtypes | join('\n') | indent(8) }}

""".strip())

def cpp_typesystem(ts, ns):
    """Creates the Cython header that wraps the Cyclus type system."""
    temp = '{0} = {1}'
    dbtypes = [temp.format(t, ts.ids[t]) for t in ts.dbtypes]
    ctx = dict(
        dbtypes=dbtypes,
        cg_warning=CG_WARNING,
        stl_cimports=STL_CIMPORTS,
        )
    rtn = CPP_TYPESYSTEM.render(ctx)
    return rtn


TYPESYSTEM_PYX = JENV.from_string('''
{{ cg_warning }}

{{ stl_cimports }}

{{ npy_imports }}

from cpython.pycapsule cimport PyCapsule_New, PyCapsule_GetPointer

# local imports
from cyclus_backend cimport cpp_typesystem
from cyclus_backend cimport cpp_cyclus_backend
from cyclus_backend.cpp_cyclus_backend cimport shared_ptr, reinterpret_pointer_cast
from cyclus_backend cimport lib


# pure python imports
import uuid
import collections
from binascii import hexlify

#
# raw type definitions
#
{% for t in dbtypes %}
{{ t }} = {{ ts.cython_cpp_name(t) }}
{%- endfor %}

cdef dict C_RANKS = {
{%- for t in dbtypes %}
    {{ ts.cython_cpp_name(t) }}: {{ ts.ranks[t] }},
{%- endfor %}
    }
RANKS = C_RANKS

cdef dict C_NAMES = {
{%- for t in dbtypes %}
    {{ ts.cython_cpp_name(t) }}: '{{ t }}',
{%- endfor %}
    }
NAMES = C_NAMES

cdef dict C_IDS = {
{%- for t in dbtypes %}
    '{{ t }}': {{ ts.cython_cpp_name(t) }},
{%- endfor %}
{%- for t in ts.uniquetypes %}
    {{ repr(ts.norms[t]) }}: {{ ts.cython_cpp_name(t) }},
{%- endfor %}
    }
IDS = C_IDS

cdef dict C_CPPTYPES = {
{%- for t in dbtypes %}
    {{ ts.cython_cpp_name(t) }}: '{{ ts.cpptypes[t] }}',
{%- endfor %}
    }
CPPTYPES = C_CPPTYPES

cdef dict C_NORMS = {
{%- for t in dbtypes %}
    {{ ts.cython_cpp_name(t) }}: {{ repr(ts.norms[t]) }},
{%- endfor %}
    }
NORMS = C_NORMS

#
# converters
#
cdef bytes blob_to_bytes(cpp_cyclus_backend.Blob value):
    rtn = value.str()
    return bytes(rtn)


cdef object uuid_cpp_to_py(cpp_cyclus_backend.uuid x):
    cdef int i
    cdef list d = []
    for i in range(16):
        d.append(<unsigned int> x.data[i])
    rtn = uuid.UUID(hex=hexlify(bytearray(d)).decode())
    return rtn


cdef cpp_cyclus_backend.uuid uuid_py_to_cpp(object x):
    cdef char * c
    cdef cpp_cyclus_backend.uuid u
    if isinstance(x, uuid.UUID):
        c = x.bytes
    else:
        c = x
    memcpy(u.data, c, 16)
    return u

cdef std_string str_py_to_cpp(object x):
    cdef std_string s
    x = x.encode()
    s = std_string(<const char*> x)
    return s


{% for n in sorted(set(ts.norms.values()), key=ts.funcname) %}
{% set decl, body, expr = ts.convert_to_py('x', n) %}
cdef object {{ ts.funcname(n) }}_to_py({{ ts.possibly_shared_cython_type(n) }} x):
    {{ decl | indent(4) }}
    {{ body | indent(4) }}
    return {{ expr }}
{%- endfor %}

{% for n in sorted(set(ts.norms.values()), key=ts.funcname) %}
{% set decl, body, expr = ts.convert_to_py('x', n) %}
cdef object any_{{ ts.funcname(n) }}_to_py(cpp_cyclus_backend.hold_any value):
    cdef {{ ts.possibly_shared_cython_type(n) }} x = value.cast[{{ ts.possibly_shared_cython_type(n) }}]()
    {{ decl | indent(4) }}
    {{ body | indent(4) }}
    return {{ expr }}
{%- endfor %}

{% for n in sorted(set(ts.norms.values()), key=ts.funcname) %}
{% set decl, body, expr = ts.convert_to_cpp('x', n) %}
cdef {{ ts.possibly_shared_cython_type(n) }} {{ ts.funcname(n) }}_to_cpp(object x):
    {{ decl | indent(4) }}
    {{ body | indent(4) }}
    return {{ expr }}
{%- endfor %}


#
# type system functions
#
cdef object db_to_py(cpp_cyclus_backend.hold_any& value, cpp_cyclus_backend.DbTypes dbtype):
    """Converts database types to python objects."""
    cdef object rtn
    {%- for i, t in enumerate(dbtypes) %}
    {% if i > 0 %}el{% endif %}if dbtype == {{ ts.cython_cpp_name(t) }}:
        #rtn = {{ ts.hold_any_to_py('value', t) }}
        rtn = any_{{ ts.funcname(t) }}_to_py(value)
    {%- endfor %}
    else:
        msg = "dbtype {0} could not be found while converting to Python"
        raise TypeError(msg.format(dbtype))
    return rtn


cdef cpp_cyclus_backend.hold_any py_to_any(object value, object t):
    """Converts a Python object into int a hold_any instance by inspecting the
    type.

    Parameters
    ----------
    value : object
        A Python object to encapsulate.
    t : dbtype or norm type (str or tupe of str)
        The type to use in the conversion.
    """
    if isinstance(t, int):
        return py_to_any_by_dbtype(value, t)
    else:
        return py_to_any_by_norm(value, t)


cdef cpp_cyclus_backend.hold_any py_to_any_by_dbtype(object value, cpp_cyclus_backend.DbTypes dbtype):
    """Converts Python object to a hold_any instance by knowing the dbtype."""
    cdef cpp_cyclus_backend.hold_any rtn
    {%- for i, t in enumerate(dbtypes) %}
    {% if i > 0 %}el{% endif %}if dbtype == {{ ts.cython_cpp_name(t) }}:
        rtn = {{ ts.py_to_any('rtn', 'value', t) }}
    {%- endfor %}
    else:
        msg = "dbtype {0} could not be found while converting from Python"
        raise TypeError(msg.format(dbtype))
    return rtn


cdef cpp_cyclus_backend.hold_any py_to_any_by_norm(object value, object norm):
    """Converts Python object to a hold_any instance by knowing the dbtype."""
    cdef cpp_cyclus_backend.hold_any rtn
    if isinstance(norm, str):
        {%- for i, t in enumerate(uniquestrtypes) %}
        {% if i > 0 %}el{% endif %}if norm == {{ repr(ts.norms[t]) }}:
            rtn = {{ ts.py_to_any('rtn', 'value', t) }}
        {%- endfor %}
        else:
            msg = "norm type {0} could not be found while converting from Python"
            raise TypeError(msg.format(norm))
    else:
        norm0 = norm[0]
        normrest = norm[1:]
        {% for i, (key, group) in enumerate(groupby(uniquetuptypes, key=firstfirst)) %}
        {% if i > 0 %}el{% endif %}if norm0 == {{ repr(key) }}:
            {%- for n, (tnorm, t) in enumerate(group) %}
            {% if n > 0 %}el{% endif %}if normrest == {{ repr(tnorm[1:]) }}:
                rtn = {{ ts.py_to_any('rtn', 'value', t) }}
            {%- endfor %}
            else:
                msg = "norm type {0} could not be found while converting from Python"
                raise TypeError(msg.format(norm))
        {% endfor %}
        else:
            msg = "norm type {0} could not be found while converting from Python"
            raise TypeError(msg.format(norm))
    return rtn


cdef object any_to_py(cpp_cyclus_backend.hold_any value):
    """Converts any C++ object to its Python equivalent."""
    cdef object rtn = None
    cdef size_t valhash = value.type().hash_code()
    # Note that we need to use the *_t tyedefs here because of
    # Bug #1561 in Cython
    {%- for i, t in enumerate(ts.uniquetypes) %}
    {% if i > 0 %}el{% endif %}if valhash == typeid({{ ts.funcname(t) }}_t).hash_code():
        rtn = any_{{ ts.funcname(t) }}_to_py(value)
    {%- endfor %}
    else:
        msg = "C++ type could not be found while converting to Python"
        raise TypeError(msg)
    return rtn

def capsule_any_to_py(value):
    """Converts a PyCapsule that holds a boost::spirit::hold_any to a python
    value.
    """
    cdef cpp_cyclus_backend.hold_any* cpp_value = <cpp_cyclus_backend.hold_any*>PyCapsule_GetPointer(value, <char*> b"value")
    py_value = any_to_py(deref(cpp_value))
    return py_value

cdef object new_py_inst(cpp_cyclus_backend.DbTypes dbtype):
    """Creates a new, empty Python instance of a database type."""
    cdef object rtn
    {%- for i, t in enumerate(dbtypes) %}
    {% if i > 0 %}el{% endif %}if dbtype == {{ ts.cython_cpp_name(t) }}:
        rtn = {{ ts.new_py_inst(t) }}
    {%- endfor %}
    else:
        msg = "dbtype {0} could not be found while making a new Python instance"
        raise TypeError(msg.format(dbtype))
    return rtn




#
# Helpers
#
def prepare_type_representation(cpptype, othertype):
    """Updates othertype to conform to the length of cpptype using None's.
    """
    cdef int i, n
    if not isinstance(cpptype, str):
        n = len(cpptype)
        if isinstance(othertype, str):
            othertype = [othertype]
        if othertype is None:
            othertype = [None] * n
        elif len(othertype) < n:
            othertype.extend([None] * (n - len(othertype)))
        # recurse down
        for i in range(1, n):
            othertype[i] = prepare_type_representation(cpptype[i], othertype[i])
        return othertype
    else:
        return othertype


'''.lstrip())


def typesystem_pyx(ts, ns):
    """Creates the Cython wrapper for the Cyclus type system."""
    nonuser_annotations = ('type', 'uniquetypeid')
    ctx = dict(
        ts=ts,
        dbtypes=ts.dbtypes,
        cg_warning=CG_WARNING,
        npy_imports=NPY_IMPORTS,
        stl_cimports=STL_CIMPORTS,
        set=set,
        repr=repr,
        sorted=sorted,
        enumerate=enumerate,
        annotations=ANNOTATIONS,
        nonuser_annotations=nonuser_annotations,
        uniquestrtypes = [t for t in ts.uniquetypes
                          if isinstance(ts.norms[t], unicode_types)],
        uniquetuptypes = sorted([(ts.norms[t], t) for t in ts.uniquetypes
                                 if not isinstance(ts.norms[t], unicode_types)], reverse=True,
                                key=lambda x: (x[0][0], x[1])),
        groupby=itertools.groupby,
        firstfirst=lambda x: x[0][0],
        )
    rtn = TYPESYSTEM_PYX.render(ctx)
    return rtn

TYPESYSTEM_PXD = JENV.from_string('''
{{ cg_warning }}

{{ stl_cimports }}

# local imports
from cyclus_backend cimport cpp_typesystem
from cyclus_backend cimport cpp_cyclus_backend


#
# raw
#
cpdef dict C_RANKS
cpdef dict C_NAMES
cpdef dict C_IDS
cpdef dict C_CPPTYPES
cpdef dict C_NORMS

#
# typedefs
#
{% for t in ts.uniquetypes %}
ctypedef {{ ts.cython_type(t) }} {{ ts.funcname(t) }}_t
{%- endfor %}

#
# converters
#
cdef bytes blob_to_bytes(cpp_cyclus_backend.Blob value)

cdef object uuid_cpp_to_py(cpp_cyclus_backend.uuid x)


cdef cpp_cyclus_backend.uuid uuid_py_to_cpp(object x)

cdef std_string str_py_to_cpp(object x)

{% for n in sorted(set(ts.norms.values()), key=ts.funcname) %}
cdef object {{ ts.funcname(n) }}_to_py({{ ts.possibly_shared_cython_type(n) }} x)
{%- endfor %}

{% for n in sorted(set(ts.norms.values()), key=ts.funcname) %}
cdef object any_{{ ts.funcname(n) }}_to_py(cpp_cyclus_backend.hold_any value)
{%- endfor %}

{% for n in sorted(set(ts.norms.values()), key=ts.funcname) %}
cdef {{ ts.possibly_shared_cython_type(n) }} {{ ts.funcname(n) }}_to_cpp(object x)
{%- endfor %}


#
# type system functions
#
cdef object db_to_py(cpp_cyclus_backend.hold_any value, cpp_cyclus_backend.DbTypes dbtype)

cdef cpp_cyclus_backend.hold_any py_to_any(object value, object t)

cdef cpp_cyclus_backend.hold_any py_to_any_by_dbtype(object value, cpp_cyclus_backend.DbTypes dbtype)

cdef cpp_cyclus_backend.hold_any py_to_any_by_norm(object value, object norm)

cdef object any_to_py(cpp_cyclus_backend.hold_any value)

cdef object new_py_inst(cpp_cyclus_backend.DbTypes dbtype)

''')

def typesystem_pxd(ts, ns):
    """Creates the Cython wrapper header for the Cyclus type system."""
    ctx = dict(
        ts=ts,
        dbtypes=ts.dbtypes,
        cg_warning=CG_WARNING,
        npy_imports=NPY_IMPORTS,
        stl_cimports=STL_CIMPORTS,
        set=set,
        sorted=sorted,
        enumerate=enumerate,
        annotations=ANNOTATIONS,
        )
    rtn = TYPESYSTEM_PXD.render(ctx)
    return rtn


#
# CLI
#

def parse_args(argv):
    """Parses typesystem arguments for code generation."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--verbose', default=False, action='store_true',
                        dest='verbose',
                        help="whether to give extra information at run time.")
    parser.add_argument('--src-dir', default='.', dest='src_dir',
                        help="the local source directory, default '.'")
    parser.add_argument('--test-dir', default='tests', dest='test_dir',
                        help="the local tests directory, default 'tests'")
    parser.add_argument('--build-dir', default='build', dest='build_dir',
                        help="the local build directory, default 'build'")
    parser.add_argument('--cpp-typesystem', default='cpp_typesystem.pxd',
                        dest='cpp_typesystem',
                        help="the name of the C++ typesystem header, "
                             "default 'cpp_typesystem.pxd'")
    parser.add_argument('--typesystem-pyx', default='typesystem.pyx',
                        dest='typesystem_pyx',
                        help="the name of the Cython typesystem wrapper, "
                             "default 'typesystem.pyx'")
    parser.add_argument('--typesystem-pxd', default='typesystem.pxd',
                        dest='typesystem_pxd',
                        help="the name of the Cython typesystem wrapper header, "
                             "default 'typesystem.pxd'")
    dbtd = os.path.join(os.path.dirname(__file__), '..', 'share', 'dbtypes.json')
    parser.add_argument('--dbtypes-json', default=dbtd,
                        dest='dbtypes_json',
                        help="the path to dbtypes.json file, "
                             "default " + dbtd)
    parser.add_argument('--cyclus-version', default=None,
                        dest='cyclus_version',
                        help="The Cyclus API version to target."
                        )
    ns = parser.parse_args(argv)
    return ns


def setup(ns):
    """Ensure that we are ready to perform code generation. Returns typesystem."""
    if not os.path.exists(ns.build_dir):
        os.mkdir(ns.build_dir)
    if not os.path.isfile(ns.dbtypes_json):
        try:
            instdir = safe_output(['cyclus', '--install-path'])
        except (subprocess.CalledProcessError, OSError):
            # fallback for conda version of cyclus
            instdir = safe_output(['cyclus_base', '--install-path'])
        ns.dbtypes_json = os.path.join(instdir.strip().decode(), 'share',
                                       'cyclus', 'dbtypes.json')
    with io.open(ns.dbtypes_json, 'r') as f:
        tab = json.load(f)
    # get cyclus version
    verstr = ns.cyclus_version
    if verstr is None:
        try:
            verstr = safe_output(['cyclus', '--version']).split()[2]
        except (subprocess.CalledProcessError, OSError):
            # fallback for conda version of cyclus
            try:
                verstr = safe_output(['cyclus_base', '--version']).split()[2]
            except (subprocess.CalledProcessError, OSError):
                # fallback using the most recent value in JSON
                ver = set([row[5] for row in tab[1:]])
                ver = max([tuple(map(int, s[1:].partition('-')[0].split('.'))) for s in ver])
    if verstr is not None:
        if isinstance(verstr, bytes):
            verstr = verstr.decode()
        ns.cyclus_version = verstr
        ver = tuple(map(int, verstr.partition('-')[0].split('.')))
    if ns.verbose:
        print('Found cyclus version: ' + verstr, file=sys.stderr)
    # make and return a type system
    ts = TypeSystem(table=tab, cycver=ver, rawver=verstr,
            cpp_typesystem=os.path.splitext(ns.cpp_typesystem)[0])
    return ts


def code_gen(ts, ns):
    """Generates code given a type system and a namespace."""
    cases = [(cpp_typesystem, ns.cpp_typesystem),
             (typesystem_pyx, ns.typesystem_pyx),
             (typesystem_pxd, ns.typesystem_pxd),]
    for func, basename in cases:
        s = func(ts, ns)
        fname = os.path.join(ns.src_dir, basename)
        orig = None
        if os.path.isfile(fname):
            with io.open(fname, 'r') as f:
                orig = f.read()
        if orig is None or orig != s:
            with io.open(fname, 'w') as f:
                f.write(s)


def main(argv=None):
    """Entry point into the code generation. Accepts list of command line arguments."""
    if argv is None:
        argv = sys.argv[1:]
    ns = parse_args(argv)
    ts = setup(ns)
    code_gen(ts, ns)


if __name__ == "__main__":
    main()
