# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.11.5
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# ## capnproto interface

import capnp
capnp.remove_import_hook()
addressbook_capnp = capnp.load('addressbook.capnp')

addressbook_capnp.qux

addresses = addressbook_capnp.AddressBook.new_message()

addresses

alice = addressbook_capnp.Person.new_message(name='alice')
alice

alice.name

people = addresses.init('people', 2) # don't call init more than once!

people[0] = alice

people[1] = addressbook_capnp.Person.new_message(name='bob')

alicePhone = alice.init('phones', 1)[0]

# ## enum type

alicePhone.type == 'mobile'

try:
    alicePhone.type = 'pager'
except AttributeError as m:
    print(m)

# ## unions

# unions are like enum structs. employment has type union, making it unique among a (fixed?) set of options.

alice.employment.which()

try:
    alice.employment.spaceship = 'Enterprise'
except AttributeError as m:
    print(m)

alice.employment.school = 'Rice'
print(alice.employment.which())

alice.employment.unemployed = None
print(alice.employment.which())

# ## i/o
# The whole point of rpc is that we can communicate in binary.

with open('example.bin', 'wb') as f:
    addresses.write(f)

# cat example.bin

with open('example.bin', 'rb') as f:
    addresses = addressbook_capnp.AddressBook.read(f)

first_person = addresses.people[0]

first_person.name

type(first_person)

employment = first_person.employment.which()
print('{} is employed at {}'.format(first_person.name, employment))
getattr(first_person.employment, employment)

# ## Dict/list

addresses.to_dict()

alice

# ## builders vs readers
# When you create a message, you are making a builder. When you read a message, you're using a reader.

type(alice)

type(addresses.people[0])

# builders have a to_bytes method

alice.to_bytes()

addressbook_capnp.Person.from_bytes(alice.to_bytes())

# ## packed format
#
# The binary data can be sent in compressed format

addressbook_capnp.Person.from_bytes_packed(alice.to_bytes_packed())

# ## RPC
#
# Specification here https://capnproto.org/rpc.html
#
# The key idea is minimizing the number of trips to/from the server by chaining dependent calls. This fits really well with kamodo's functional style, where the user is encouraged to use function composition in their pipelines! We want to be able to leverage these capabilities in our `KamodoRPC` class.

# ### Calculator test
#
# The calculator spec is located in `calculator.capnp` and is copied from the pycapnp repo.
#
# Run the calculator server in a separate window before executing these cells.
#
# `python calculator_server.py 127.0.0.1:6000`

calculator_capnp = capnp.load('calculator.capnp')
client = capnp.TwoPartyClient('127.0.0.1:6000')

# ### bootstrapping
# There could be many interfaces defined within a given service. The client's `bootstrap` method will get the interface marked for bootstrapping by server.
#
# First bootstrap the Calculator interface

calculator = client.bootstrap().cast_as(calculator_capnp.Calculator)

calculator2 = client.bootstrap().cast_as(calculator_capnp.Calculator)

# The server defines which interface to bootstrap: `TwoPartyServer(address, bootstrap=CalculatorImpl()`.

# ### methods
# Ways to call an RPC method

# +
request = calculator.evaluate_request()
request.expression.literal = 123
eval_promise = request.send()

# result = eval_promise.wait().value.read().wait() # blocking?
read_result = eval_promise.then(lambda ret: ret.value.read()).wait() # chained
read_result.value
# -

read_result

# You may also interogate available rpc methods:

calculator.schema.method_names

calculator.schema.methods

# ### test rpc
# * can test rpc with socket pair:
#
# ```python
#
# class ServerImpl():
#     ...
#
# read, write = socket.socketpair()
#
# _ = capnp.TwoPartyServer(write, bootstrap=ServerImpl())
# client = capnp.TwoPartyClient(read)
# ```
#

# ### Type Ids
#
# To generate file ids, make sure you have capnp command line tool installed:
# ```sh
# conda install -c conda-forge capnp
# capnp id # generates unique file id
# capnp compile -ocapnp calculator.capnp # returns a schema filled in with ids for all new types
# ```
#
# The unique type identifiers aid in backward compatibility and schema flexibility.
#
# capnproto schema language reference https://capnproto.org/language.html

# ## RPC Parameters
# These variables can be created by the server/client and wrap numpy arrays.

# +
import capnp
# capnp.remove_import_hook()
kamodo_capnp = capnp.load('kamodo.capnp')
import numpy as np

def class_name(obj):
    """get fully qualified class name of object"""
    return ".".join([obj.__class__.__module__, obj.__class__.__name__])

def param_to_array(param):
    """convert from parameter to numpy array
    assume input is numpy binary
    """
    if len(param.data) > 0:
        return np.frombuffer(param.data, dtype=param.dtype).reshape(param.shape)
    else:
        return np.array([], dtype=param.dtype)

def array_to_param(arr):
    """convert an array to an rpc parameter"""
    param = kamodo_capnp.Kamodo.Variable.new_message()
    if len(arr) > 0:
        param.data = arr.tobytes()
        param.shape = arr.shape
        param.dtype = str(arr.dtype)
    return param


# -

a = np.linspace(-5,5,12).reshape(3,4)

b = array_to_param(np.array([3,4,5]))

b.to_dict()

param_to_array(b)

b.to_dict()

# ## RPC Functions
#
# These functions execute on the server.

# +
import capnp
# capnp.remove_import_hook()
kamodo_capnp = capnp.load('kamodo.capnp')

# import kamodo_capnp
# -

import numpy as np


class Poly(kamodo_capnp.Kamodo.Function.Server):
    def __init__(self):
        pass
        
    def call(self, params, **kwargs):
        if len(kwargs) == 0:
            return kamodo_capnp.Kamodo.Variable.new_message()
        print('serverside function called with {} params'.format(len(params)))
        param_arrays = [param_to_array(v) for v in params]
        x = sum(param_arrays)
        print(x)
        result = x**2 - x - 1
        result_ = array_to_param(result)
        return result_


# Set up a client/server socket for testing.

import socket
read, write = socket.socketpair()

# instantiate a server with a Poly object

server = capnp.TwoPartyServer(write, bootstrap=Poly())

# instantiate a client with bootstrapping
#
# > capabilities are intrinsically dynamic, and they hold no run time type information, so we need to pick what interface to interpret them as.

client = capnp.TwoPartyClient(read)
# polynomial implementation lives on the server
polynomial = client.bootstrap().cast_as(kamodo_capnp.Kamodo.Function)

b.to_dict()

poly_promise = polynomial.call(params=[b])

# +
# evaluate ...

response = poly_promise.wait()
# -

param_to_array(response.result) # (sum(b))**2 - sum(b) - 1


class FunctionRPC:
    def __init__(self):
        self.func = client.bootstrap().cast_as(kamodo_capnp.Kamodo.Function)
    
    def __call__(self, params):
        params_ = [array_to_param(_) for _ in params]
        func_promise = self.func.call(params_)
        # evaluate
        response = func_promise.wait().result
        return param_to_array(response)


serverside_function = FunctionRPC()

a

serverside_function(params=[a, a, a])

# ## Function groups

import capnp
# capnp.remove_import_hook()
kamodo_capnp = capnp.load('kamodo.capnp')

kamodo_capnp.Kamodo.Function

a = np.linspace(-1,1,10)
b = array_to_param(a)
b.to_dict()

arg_units = kamodo_capnp.Kamodo.Map.new_message(entries=[dict(key='a', value='b')])

arg_units.to_dict()

# +
#     units @0 :Text;
#     argUnits @1 :Map(Text, Text);
#     citation @2 :Text;
#     equation @3 :Text; # latex expression
#     hiddenArgs @4 :List(Text);

# +
meta_ = kamodo_capnp.Kamodo.Meta(
    units= 'nPa',
    argUnits = dict(entries=[dict(key='x', value='cm')]),
    citation = '',
    equation ='x^2-x-1',
    hiddenArgs = [])

field = kamodo_capnp.Kamodo.Field.new_message(
            func=Poly(),
            defaults=dict(entries=[dict(key='x', value=b)]),
            meta=meta_,
            data=b,
        )
field.to_dict()

# +
import forge

from kamodo.util import construct_signature

import socket
read, write = socket.socketpair()

from kamodo import Kamodo, kamodofy

b = array_to_param(a)

class KamodoServer(kamodo_capnp.Kamodo.Server):
    def __init__(self):
        field = kamodo_capnp.Kamodo.Field.new_message(
                    func=Poly(),
                    defaults=dict(entries=[dict(key='x', value=b)]),
                    meta=meta_,
                    data=b,
                )  
        self.fields = dict(entries=[dict(key='P_n', value=field)])

    def getFields(self, **kwargs):
        return self.fields
    
server = capnp.TwoPartyServer(write, bootstrap=KamodoServer())

def rpc_map_to_dict(rpc_map, callback = None):
    if callback is None:
        return {_.key: _.value for _ in rpc_map.entries}
    else:
        return {_.key: callback(_.value) for _ in rpc_map.entries}

def rpc_dict_to_map(d):
    return dict(entries=[dict(key=k, value=v) for k,v in d.items()])

class KamodoClient(Kamodo):
    def __init__(self, client, **kwargs):
        self._client = client.bootstrap().cast_as(kamodo_capnp.Kamodo)
        self._rpc_fields = self._client.getFields().wait().fields
        
        super(KamodoClient, self).__init__(**kwargs)
        
        for entry in self._rpc_fields.entries:
            self.register_rpc(entry)
            
    def register_rpc(self, entry):
        symbol = entry.key
        field = entry.value
        
        meta = field.meta
        arg_units = rpc_map_to_dict(meta.argUnits)
        
        defaults = rpc_map_to_dict(field.defaults, param_to_array)
        
#         print(arg_units)
#         print(field.to_dict())
#         print(defaults)
        
        @kamodofy(units=meta.units,
                  arg_units=arg_units,
                  citation=meta.citation,
                  equation=meta.equation,
                  hidden_args=meta.hiddenArgs)
        @forge.sign(*construct_signature(**defaults))
        def remote_func(**kwargs):
            # params must be List(Variable) for now
            params = [array_to_param(v) for k,v in kwargs.items()]
            response = field.func.call(params=params).wait().result
            return param_to_array(response)

        self[symbol] = remote_func
        
client = capnp.TwoPartyClient(read)
        
kclient = KamodoClient(client)
kclient
# -

x = np.linspace(-5,5,33)
kclient.P_n(x)

# default forwarding works when input units are unchanged
kclient['p(x[m])[Pa]'] = 'P_n'

kclient['P_2[nPa]'] = lambda x=b: x**2-x-1

kclient

kclient.detail()

try:
    kclient.plot('p')
except TypeError as m:
    print(m)

kclient['p_2(x[m])[nPa]'] = 'P_n'
kclient

# +
from kamodo.util import construct_signature

import socket
read, write = socket.socketpair()

from kamodo import Kamodo, kamodofy

b = array_to_param(a)

def rpc_map_to_dict(rpc_map, callback = None):
    if callback is None:
        return {_.key: _.value for _ in rpc_map.entries}
    else:
        return {_.key: callback(_.value) for _ in rpc_map.entries}

def rpc_dict_to_map(d):
    return dict(entries=[dict(key=k, value=v) for k,v in d.items()])

class KamodoServer(kamodo_capnp.Kamodo.Server):
    def __init__(self):
        field = kamodo_capnp.Kamodo.Field.new_message(
                    func=Poly(),
                    defaults=dict(entries=[dict(key='x', value=b)]),
                    meta=meta_,
                    data=b,
                )  
        self.fields = dict(entries=[dict(key='P_n', value=field)])

    def getFields(self, **kwargs):
        return self.fields
    
server = capnp.TwoPartyServer(write, bootstrap=KamodoServer())

class KamodoClient(Kamodo):
    def __init__(self, client, **kwargs):
        self._client = client.bootstrap().cast_as(kamodo_capnp.Kamodo)
        self._rpc_fields = self._client.getFields().wait().fields
        
        super(KamodoClient, self).__init__(**kwargs)
        
        for entry in self._rpc_fields.entries:
            self.register_rpc(entry)
            
    def register_rpc(self, entry):
        symbol = entry.key
        field = entry.value
        
        meta = field.meta
        arg_units = rpc_map_to_dict(meta.argUnits)
        
        defaults = rpc_map_to_dict(field.defaults, param_to_array)
        
        @kamodofy(units=meta.units,
                  arg_units=arg_units,
                  citation=meta.citation,
                  equation=meta.equation,
                  hidden_args=meta.hiddenArgs)
        @forge.sign(*construct_signature(**defaults))
        def remote_func(**kwargs):
            # params must be List(Variable) for now
            params = [array_to_param(v) for k,v in kwargs.items()]
            response = field.func.callMap(params=params).wait().result
            return param_to_array(response)

        self[symbol] = remote_func
        
client = capnp.TwoPartyClient(read)
        
kclient = KamodoClient(client)
kclient
# -

kclient.P_n

kclient.to_latex()

# default forwarding breaks when input units change
kclient['p_2(x[m])[nPa]'] = 'P_n' # P_n(x[cm])[nPa]
try:
    kclient.plot('p_2')
except TypeError as m:
    print(m)

kclient.P_n(np.linspace(-5,5,10))

kclient.plot(P_n=dict(x=x))

# ## RPC decorator

import capnp
capnp.remove_import_hook()
kamodo_capnp = capnp.load('kamodo.capnp')

from kamodo import get_defaults
import numpy as np


def myfunc(x=np.linspace(-5,5,33)):
    return x


# +
def rpc_map_to_dict(rpc_map, callback = None):
    if callback is not None:
        return {_.key: callback(_.value) for _ in rpc_map.entries}
    else:
        return {_.key: _.value for _ in rpc_map.entries}
        

def rpc_dict_to_map(d, callback = None):
    if callback is not None:
        entries=[dict(key=k, value=callback(v)) for k,v in d.items()]
    else:
        entries=[dict(key=k, value=v) for k, v in d.items()]
    return dict(entries=entries)
    
def class_name(obj):
    """get fully qualified class name of object"""
    return ".".join([obj.__class__.__module__, obj.__class__.__name__])

def param_to_array(param):
    """convert from parameter to numpy array
    assume input is numpy binary
    """
    if len(param.data) > 0:
        return np.frombuffer(param.data, dtype=param.dtype).reshape(param.shape)
    else:
        return np.array([], dtype=param.dtype)

def array_to_param(arr):
    """convert an array to an rpc parameter"""
    param = kamodo_capnp.Kamodo.Variable.new_message()
    arr_ = np.array(arr)
    if len(arr) > 0:
        param.data = arr_.tobytes()
        param.shape = arr_.shape
        param.dtype = str(arr_.dtype)
    return param


# -

array_to_param([3,4,5,1.]).to_dict()

get_defaults(myfunc)

from kamodo import Kamodo

# +
from kamodo.util import get_args

import socket

class KamodoRPC(kamodo_capnp.Kamodo.Server):
    def __init__(self, **fields):
        self.fields = fields

    def getFields(self, **kwargs):
        return rpc_dict_to_map(self.fields)

    def __getitem__(self, key):
        return self.fields[key]
    
    def __setitem__(self, key, field):
        self.fields[key] = field

class FunctionRPC(kamodo_capnp.Kamodo.Function.Server):
    def __init__(self, func, verbose=False):
        """Converts a function to RPC callable"""
        self._func = func
        self.verbose = verbose
        self.args = get_args(self._func)
        self.kwargs = get_defaults(self._func)
    
    def getArgs(self, **rpc_kwargs):
        return list(self.args)
        
    def getKwargs(self, **rpc_kwargs):
        if self.verbose:
            print('retrieving kwargs')
        return [dict(name=k, value=array_to_param(v)) for k,v in self.kwargs.items()]
        
    def call(self, args, kwargs, **rpc_kwargs):
        """mimic a pythonic function
        
        Should raise TypeError when multiple values for argument"""
        
        param_dict = self.kwargs
        
        # insert args
        arg_dict = {}
        for i, value in enumerate(args):
            arg_dict.update({self.args[i]: param_to_array(value)})
        param_dict.update(arg_dict)
        
        # insert kwargs
        for kwarg in kwargs:
            if kwarg.name in arg_dict:
                raise TypeError('multiple values for argument {}, len(args)={}'.format(kwarg.name, len(args)))
            param_dict.update({kwarg.name: param_to_array(kwarg.value)})
        if self.verbose:
            print('serverside function called with {} params'.format(len(param_dict)))
        result = self._func(**param_dict)
        result_param = array_to_param(result)
        return result_param


# test FunctionRPC
read, write = socket.socketpair()

server = capnp.TwoPartyServer(write,
                              bootstrap=FunctionRPC(
                                  lambda x=np.linspace(-5,5,30): x**2-x-1,
                                  verbose=True))
client = capnp.TwoPartyClient(read)
polynomial = client.bootstrap().cast_as(kamodo_capnp.Kamodo.Function)
defaults = polynomial.getKwargs().wait().kwargs

result = polynomial.call(kwargs=[dict(name='x', value=array_to_param(np.linspace(-5,5,11)))]).wait().result
# -

param_to_array(result)

for _ in polynomial.getArgs().wait().args:
    print(_)

# FunctionRPC converts a function into an RPC object, so any of KamodoServer's functions will be callable.

kserver.verbose=False

kserver['f(x[km])[cm]'] = 'x**2-x-1'

kserver.detail()

kserver.f.meta

# +
# kamodofy?
# -



# +
from kamodo import Kamodo, latex


# class KamodoServer(Kamodo):
#     def __init__(self, **kwargs):
#         """A Kamodo server capable of serving its functions over RPC"""
#         super(KamodoServer, self).__init__(**kwargs)


#     def to_rpc_meta(self, key):
#         meta = self[key].meta
#         equation = meta.get('equation', self.to_latex(key, mode='inline')).strip('$')
#         equation = meta.get('equation', latex(self.signatures[key]['rhs']))
#         arg_unit_entries = []
#         for k,v in meta.get('arg_units', {}):
#             arg_unit_entries.append({'key': k, 'value': v})
            
#         return kamodo_capnp.Kamodo.Meta(
#             units=meta.get('units', ''),
#             argUnits=dict(entries=arg_unit_entries),
#             citation=meta.get('citation', ''),
#             equation=equation,
#             hiddenArgs=meta.get('hidden_args', []),
#         )
    
#     def register_rpc_field(self, key):
#         func = self[key]
#         signature = self.signatures[key]
#         field = kamodo_capnp.Kamodo.Field.new_message(
#             func=FunctionRPC(func),
#             meta=self.to_rpc_meta(key),
#         )
#         self._server[key] = field

#     def serve(self, write):
#         self._server = KamodoRPC()
        
#         for key in self.signatures:
#             print('serving {}'.format(key))
#             self.register_rpc_field(key)
        
#         server = capnp.TwoPartyServer(write, bootstrap = self._server)
#         return server
    




# def poly_impl(x = np.linspace(-5,5,33)):
#     print('polynomial called with x={}'.format(x.shape))
#     return x**2 - x - 1
    

kserver = Kamodo(f='x**2-x-1')

read, write = socket.socketpair()

server = kserver.serve(write)

from kamodo import kamodofy
import forge
from kamodo.util import construct_signature

class KamodoClient(Kamodo):
    def __init__(self, client, **kwargs):
        self._client = client.bootstrap().cast_as(kamodo_capnp.Kamodo)
        self._rpc_fields = self._client.getFields().wait().fields
        
        super(KamodoClient, self).__init__(**kwargs)
        
        for entry in self._rpc_fields.entries:
            self.register_rpc(entry)
            
    def register_rpc(self, entry):
        """resolve the remote signature
        f(*args, **kwargs) -> f(x,y,z=value)
        """
        symbol = entry.key
        field = entry.value
        
        meta = field.meta
        arg_units = rpc_map_to_dict(meta.argUnits)
        
        defaults_ = field.func.getKwargs().wait().kwargs
        func_defaults = {_.name: param_to_array(_.value) for _ in defaults_}
        func_args_ = [str(_) for _ in field.func.getArgs().wait().args]
        func_args = [_ for _ in func_args_ if _ not in func_defaults]
        
        if len(meta.equation) > 0:
            equation = meta.equation
        else:
            equation = None
        
        @kamodofy(units=meta.units,
                  arg_units=arg_units,
                  citation=meta.citation,
                  equation=equation,
                  hidden_args=meta.hiddenArgs)
        @forge.sign(*construct_signature(*func_args, **func_defaults))
        def remote_func(*args, **kwargs):
            # params must be List(Variable) for now
#             print(args)
#             print(kwargs)
            args_ = [array_to_param(arg) for arg in args]
            kwargs_ = [dict(name=k, value= array_to_param(v)) for k, v in kwargs.items()]
            result = field.func.call(args=args_, kwargs=kwargs_).wait().result
            return param_to_array(result)

        self[symbol] = remote_func
        
client = capnp.TwoPartyClient(read)
        
kclient = KamodoClient(client)
kclient
# -

kserver

kclient.f([3, 4, 5])

kclient.plot(f=dict(x=np.linspace(-1, 5, 303)))

kclient.f(np.linspace(-5,5,11))

param_to_array(array_to_param([3,4,5]))

kclient.f(x=[3, 4, 5, 9])

kserver = KamodoServer(f='x**2-x-1')
kserver

# # Test

import socket
read, write = socket.socketpair()

from kamodo import Kamodo

kserver = Kamodo(f='x**2-x-1')

import numpy as np

kserver.plot(f=dict(x=np.linspace(-5,5,55)))

server = kserver.server(write)

from proto import KamodoClient



# ## String Sanitizing

from asteval import Interpreter

aeval = Interpreter()

aeval('x=3')
aeval('1+x')

aeval.symtable['sum']

# ## RPC expressions
#
# Wrap sympy expressions with placeholder alegebraic calls (to be executed on server)

import numpy as np

import sys

sys.path.append('..')

from proto import FunctionRPC


def add_impl(*args):
    print('computing {}'.format('+'.join((str(_) for _ in args))))
    return np.add(*args)


add_ = lambda *params: np.add(*params)

add_(np.array([3,24]), 4)

np.multiply(np.array([3,4]), 10)

np.power(2,3.)

FunctionRPC(add_)

add_impl(np.array([2,4,5]), 4)

# +
from sympy import Function, sympify
from sympy import Add, Mul, Pow
from functools import reduce
from operator import mul, add, pow

AddRPC = Function('AddRPC')
MulRPC = Function('MulRPC')
PowRPC = Function('PowRPC')


def rpc_expr(expr):
    if len(expr.args) > 0:
        gather = [rpc_expr(arg) for arg in expr.args]
        if expr.func == Add:
            return AddRPC(*gather)
        if expr.func == Mul:
            return MulRPC(*gather)
        if expr.func == Pow:
            return PowRPC(*gather)
    return expr


def add_impl(*args):
    print('computing {}'.format('+'.join((str(_) for _ in args))))
    return reduce(add, args)

def mul_impl(*args):
    print('computing {}'.format('*'.join((str(_) for _ in args))))
    return reduce(mul, args)

def pow_impl(base, exp):
    print('computing {}^{}'.format(base, exp))
    return pow(base,exp)


# -

expr_ = sympify('30*a*b + c**2+sin(c)')
expr_

rpc_expr(expr_)

add_impl(3,4,5)

mul_impl(3,4,5)

pow_impl(3,4)

func_impl = dict(AddRPC=add_impl,
                 MulRPC=mul_impl,
                 PowRPC=pow_impl)

from kamodo import Kamodo
from sympy import lambdify
from kamodo.util import sign_defaults


# +
class KamodoClient(Kamodo):
    def __init__(self, server, **kwargs):
        self._server = server
        super(KamodoClient, self).__init__(**kwargs)
        
    def vectorize_function(self, symbol, rhs_expr, composition):
        """lambdify the input expression using server-side promises"""
        print('vectorizing {} = {}'.format(symbol, rhs_expr))
        print('composition keys {}'.format(list(composition.keys())))
        func = lambdify(symbol.args,
                        rpc_expr(rhs_expr),
                        modules=[func_impl, 'numpy', composition])
        signature = sign_defaults(symbol, rhs_expr, composition)
        return signature(func)
    
kamodo = KamodoClient('localhost:8050')
kamodo['f[cm]'] = 'x**2-x-1'
kamodo['g'] = 'f**2'
kamodo['h[km]'] = 'f'
kamodo
# -

kamodo.f(3)

assert kamodo.f(3) == 3**2 - 3 - 1

kamodo

kamodo.g(4)

# ## Serverside Algebra

# +
import capnp
# capnp.remove_import_hook()
kamodo_capnp = capnp.load('kamodo.capnp')

import socket
read, write = socket.socketpair()

# +
from operator import mul, add, pow
from functools import reduce

# def add_impl(*args):
#     print('computing {}'.format('+'.join((str(_) for _ in args))))
#     return reduce(add, args)

# def mul_impl(*args):
#     print('computing {}'.format('*'.join((str(_) for _ in args))))
#     return reduce(mul, args)

# def pow_impl(base, exp):
#     print('computing {}^{}'.format(base, exp))
#     return pow(base,exp)

class AddImpl(kamodo_capnp.Kamodo.Function.Server):
    def call(self, params, **kwargs):
        result = reduce(add, param_arrays)
        return array_to_param(result)


class Algebra(kamodo_capnp.Kamodo.Server):
    def __init__(self):
        self.add = AddImpl()


# -

server = capnp.TwoPartyServer(write, bootstrap=Algebra())

client = capnp.TwoPartyClient(read)


class ClientSideFunction:
    def __init__(self, client, op):
        self.kamodo = client.bootstrap().cast_as(kamodo_capnp.Kamodo)
    
    def __call__(self, *params):
        params_ = [array_to_param(_) for _ in params]
        print('client passing params to server')
        func_promise = self.kamodo.Algebra.add.call(params_)
        # evaluate
        response = func_promise.wait().result
        return param_to_array(response)


f = ClientSideFunction(client, 'add')

import numpy as np

a = np.linspace(-1,1,15)

f(a,a,a,a)

# ## Kamodo Fields
#
# Define fields available for remote call

# +
from kamodo import Kamodo
import capnp
# import kamodo_capnp

class KamodoRPCImpl(kamodo_capnp.Kamodo.Server):
    """Interface class for capnp"""
    def __init__(self):
        pass
    
    def getFields(self, **kwargs):
        """
        Need to return a list of fields
          struct Field {
            symbol @0 :Text;
            func @1 :Function;
          }
        """

        f = kamodo_capnp.Kamodo.Field.new_message(symbol='f', func=Poly())
        return [f]


# -

read, write = socket.socketpair()

write = capnp.TwoPartyServer(write, bootstrap=KamodoRPCImpl())

client = capnp.TwoPartyClient(read)
kap = client.bootstrap().cast_as(kamodo_capnp.Kamodo)

fields = kap.getFields().wait().fields

param_to_array(fields[0].func.call([
    array_to_param(a)]).wait().result)


class KamodoRPC(Kamodo):
    def __init__(self, read_url=None, write_url=None, **kwargs):
        
        if read_url is not None:
            self.client = capnp.TwoPartyClient(read_url)
        if write_url is not None:
            self.server = capnp.TwoPartyServer(write_url,
                                               bootstrap=kamodo_capnp.Kamodo())
        super(Kamodo, self).__init__(**kwargs)


from kamodo import reserved_names

from sympy.abc import _clash # gathers reserved symbols
