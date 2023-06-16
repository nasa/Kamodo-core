# RPC

Kamodo includes a Remote Call Procedure (RPC) interface, based on [capnproto](https://capnproto.org/). This allows Kamodo objects to both serve and connect to other kamodo functions hosted on external servers.

We can test this functionality using [docker compose](https://docs.docker.com/compose/) (See the `docker-compose.yaml` file in the base of this repo).

Our docker-compose configuration contains two services we can use to test this functionality:

* kamodo-rpc-py37 (runs kamodo-core with python 3.7)
* kamodo-rpc-py38 (runs kamodo-core with python 3.8)

Here are the corresponding definitions in `docker-compose.yaml`:

```yaml
  kamodo-rpc-py37:
    build:
      context: .
      dockerfile: dockerfiles/kamodo-rpc-py37.Dockerfile
    volumes:
      - type: bind
        source: ${PWD}
        target: /kamodo-core
    ports:
      - "60001:60000"
  kamodo-rpc-py38:
    build:
      context: .
      dockerfile: dockerfiles/kamodo-rpc-py38.Dockerfile
    volumes:
      - type: bind
        source: ${PWD}
        target: /kamodo-core
    ports:
      - "60000:60000"
    command:
      - kamodo-rpc
      - rpc_conf=kamodo/rpc/kamodo_rpc_test.yaml
      - host=0.0.0.0
      - port='60000'
```

Either one will mount the base of the repo into the corresponding container at run time. Note that `kamodo-rpc-py37` will use port `60001`.


## Configuring a Kamodo server

To configure a Kamodo server, create a `kamodo_rpc.yaml` file. A simple example is given in the root of the kamodo-core repo:

```yaml 
{! ../kamodo_rpc.yaml !}
```
The `!Kamodo` line informs our yaml parser that the next line defines a kamodo object with the following key-value pairs.

!!! note
    The kamodo specification file is under rapid development

<!-- #region -->
## Starting an Kamodo server

To build a container compatible with python `3.7`, run

```sh
docker compose build kamodo-rpc-py37
```

This will pull the latest version of `kamodo-core` and build it inside a container ready for deployment. Feel free to build on top of the resulting image.

Next, start the container with:

```sh
docker compose up kamodo-rpc-py37
```

<details>
    <summary> Click to expand console output

    </summary>
```console
[+] Running 1/0
 ⠿ Container kamodo-core-kamodo-rpc-py37-1  Created                                                                     0.0s
Attaching to kamodo-core-kamodo-rpc-py37-1
kamodo-core-kamodo-rpc-py37-1  | {'rpc_conf': 'kamodo/rpc/kamodo_rpc_test.yaml', 'host': '0.0.0.0', 'port': '60000'}
kamodo-core-kamodo-rpc-py37-1  | serving   arg_units lhs           rhs symbol units
kamodo-core-kamodo-rpc-py37-1  | f        {}   f  x**2 - x - 1   f(x)      
kamodo-core-kamodo-rpc-py37-1  | certfile not supplied
kamodo-core-kamodo-rpc-py37-1  | keyfile not supplied
kamodo-core-kamodo-rpc-py37-1  | using default certificate
kamodo-core-kamodo-rpc-py37-1  | Using selfsigned cert from: selfsigned.cert
kamodo-core-kamodo-rpc-py37-1  | [2022-05-04 22:23:23,543][rpc.proto][DEBUG] - Try IPv4
```
</details>

At startup, the container automatically generates an SSL key and certificate (when one is not already present). 

## Starting a client

Since the root of the repo is mapped into the container, we can access the cert from the host `selfsigned.cert`.

Open a separate terminal and connect to the container from your host system (use the same directory where `selfsigned.cert` i.e. the root of the mounted kamodo-core repo):

```python
In [1]: from kamodo import KamodoClient
In [2]: k = KamodoClient(port=60001)
In [3]: k.f(5) 
Out[3]: 19 # (5)**2-(5)-1

```
<!-- #endregion -->

<!-- #region -->
# RPC Spec

Kamodo uses capnproto to communicate binary data between functions hosted on different systems. This avoids the need for json serialization and allows for server-side function pipelining while minimizing data transfers.

Kamodo's RPC specification file is located in `kamodo/rpc/kamodo.capnp`:

```sh
{! ../kamodo/rpc/kamodo.capnp !}
```

The above spec allows a Kamodo client (or server) to be implemented in many languages, including [C++](https://capnproto.org/cxx.html), C# (.NET Core), Erlang, Go, Haskell, JavaScript, OCaml, and Rust.

Further reading on capnproto may be found here: 

* [Overview](https://capnproto.org/index.html)
* [Schema language](https://capnproto.org/language.html)
* [RPC](https://capnproto.org/rpc.html)
* Python implementation - [pycapnp](http://capnproto.github.io/pycapnp/quickstart.html)
<!-- #endregion -->
