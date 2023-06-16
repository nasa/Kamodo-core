
import argparse
import asyncio
import os
import socket
import ssl

import capnp
import calculator_capnp
from kamodo.rpc.proto import from_rpc_literal, to_rpc_literal



this_dir = os.path.dirname(os.path.abspath(__file__))


kamodo_capnp = capnp.load('{}/kamodo.capnp'.format(this_dir))

async def myreader(client, reader):
    while True:
        data = await reader.read(4096)
        client.write(data)


async def mywriter(client, writer):
    while True:
        data = await client.read(4096)
        writer.write(data.tobytes())
        await writer.drain()


def parse_args():
    parser = argparse.ArgumentParser(
        usage="Connects to the Calculator server \
at the given address and does some RPCs"
    )
    parser.add_argument("host", help="HOST:PORT")

    return parser.parse_args()


async def main(host):
    host = host.split(":")
    addr = host[0]
    port = host[1]

    # Setup SSL context
    ctx = ssl.create_default_context(
        ssl.Purpose.SERVER_AUTH, cafile=os.path.join(this_dir, "selfsigned.cert")
    )

    # Handle both IPv4 and IPv6 cases
    try:
        print("Try IPv4")
        reader, writer = await asyncio.open_connection(
            addr, port, ssl=ctx, family=socket.AF_INET
        )
    except Exception:
        print("Try IPv6")
        reader, writer = await asyncio.open_connection(
            addr, port, ssl=ctx, family=socket.AF_INET6
        )

    # Start TwoPartyClient using TwoWayPipe (takes no arguments in this mode)
    client = capnp.TwoPartyClient()

    # Assemble reader and writer tasks, run in the background
    coroutines = [myreader(client, reader), mywriter(client, writer)]
    asyncio.gather(*coroutines, return_exceptions=True)

    # Bootstrap the Calculator interface
    # calculator = client.bootstrap().cast_as(calculator_capnp.Calculator)
    fclient = client.bootstrap().cast_as(kamodo_capnp.Kamodo.Function)

    print(fclient)
    result = (await fclient.call([to_rpc_literal(5)]).a_wait()).result
    assert from_rpc_literal(result) == 25
    print('done')

if __name__ == "__main__":
    # Using asyncio.run hits an asyncio ssl bug
    # https://bugs.python.org/issue36709
    # asyncio.run(main(parse_args().host), loop=loop, debug=True)
    loop = asyncio.get_event_loop()
    loop.run_until_complete(main(parse_args().host))