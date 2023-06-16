"""
Tests for rpc interface
"""

import pytest

from kamodo import Kamodo, KamodoClient, kamodofy

import capnp
import socket


def test_register_remote():
	@kamodofy(units='kg')
	def remote_f(x):
	    print('remote f called')
	    return x**2-x-1
	    
	kserver = Kamodo(f=remote_f)
	read, write = socket.socketpair()
	server = kserver.serve(write)
	kclient = KamodoClient(read, verbose=False)
	assert kclient.f(3) == 3**2-3-1
	assert kclient.f.meta['units'] == 'kg'

def test_remote_composition():
	@kamodofy(units='kg')
	def myf(x):
	    print('remote f called')
	    return x**2-x-1

	@kamodofy(units='gram')
	def myg(y):
	    print('remote g called')
	    return y - 1
	    
	kserver = Kamodo(f=myf, g=myg, verbose=True)

	read, write = socket.socketpair()
	server = kserver.serve(write)
	kclient = KamodoClient(read, verbose=False)

	kclient['H(x,y)[kg]'] = 'f+g' # results in f(x)+g(y)/1000
	assert kclient.H(3,4) == 3**2-3-1 + (4-1)/1000


# def test_relay():
# 	# this test fails with capnp.lib.capnp.KjException
# 	# !loop.running; wait() is not allowed from within event callbacks.
# 	# The wait() call occurs in remote_func

# 	@kamodofy(units='kg')
# 	def remote_f(x):
# 	    print('remote f called')
# 	    return x**2-x-1
	    
# 	kserver = Kamodo(f=remote_f)

# 	read, write = socket.socketpair()
# 	kserver.serve(write)
	
# 	relay = KamodoClient(read, verbose=False)

# 	read2, write2 = socket.socketpair()
# 	relay.serve(write2)

# 	kclient = KamodoClient(read2, verbose=False)

# 	assert kclient.f(3) == 3**2-3-1
# 	assert kclient.f.meta['units'] == 'kg'



