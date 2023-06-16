from kamodo import kamodofy, Kamodo, get_defaults
import numpy as np

@kamodofy(units='kg')
def remote_f(x=np.linspace(-5,5,33)):
    print('remote f called')
    print('remote f defaults: {}'.format(get_defaults(remote_f)))
    x_ = np.array(x)
    return x_**2 - x_ - 1

kserver = Kamodo(f=remote_f, verbose=True)

server = kserver.serve(certfile='hey.cert', keyfile='hey.key')
