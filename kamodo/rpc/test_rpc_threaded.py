import platform
import socket
import threading

import pytest

import capnp

from kamodo import kamodofy, Kamodo, get_defaults, KamodoClient
import numpy as np
import time


@kamodofy(units='kg')
def remote_f(x=np.linspace(-5, 5, 33)):
    print('remote f called')
    print(f'remote f defaults: {get_defaults(remote_f)}')
    x_ = np.array(x)
    return x_ ** 2 - x_ - 1


def test_using_threads():
    """
    Thread test
    """
    capnp.remove_event_loop(True)
    capnp.create_event_loop(True)

    def run_server():
        kserver = Kamodo(f=remote_f, verbose=True)
        server = kserver.serve()

    server_thread = threading.Thread(target=run_server)
    server_thread.daemon = True
    server_thread.start()

    wait = 1
    print(f'waiting {wait} second')
    time.sleep(wait)
    kclient = KamodoClient(verbose=True)

    print('client started')

    print('f defaults: (should match server defaults)')
    print(get_defaults(kclient.f), '\n')
    print('calling f without params')
    print(kclient.f())

    # server_thread.stop()

if __name__ == "__main__":
    test_using_threads()

    