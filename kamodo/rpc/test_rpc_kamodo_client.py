import asyncio

import numpy as np

from kamodo import KamodoClient, kamodo, get_defaults


kclient = KamodoClient(verbose=True)

print('f defaults: (should match server defaults)')
print(get_defaults(kclient.f), '\n')
print('calling f without params')
print(kclient.f())

# print(asyncio.run(kclient.f(3)))
x = np.linspace(-1,1,12)
print('test_rpc_kamodo_client calling kclient.f({})'.format(x))

result = kclient.f(x)

print('result: {}'.format(result))
print('f defaults:')
print(get_defaults(kclient.f))

print('calling f() without params')
print(kclient.f())

