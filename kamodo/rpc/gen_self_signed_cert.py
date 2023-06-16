# implementation provided by John Schember and released under MIT License
# https://nachtimwald.com/2019/11/14/python-self-signed-cert-gen/

import socket
import sys

def _gen_openssl(valid_for):
    import random
    from OpenSSL import crypto

    pkey = crypto.PKey()
    pkey.generate_key(crypto.TYPE_RSA, 2048)

    x509 = crypto.X509()
    subject = x509.get_subject()
    subject.commonName = socket.gethostname()
    x509.set_issuer(subject)
    x509.gmtime_adj_notBefore(0)
    x509.gmtime_adj_notAfter(valid_for*24*60*60)
    x509.set_pubkey(pkey)
    x509.set_serial_number(random.randrange(100000))
    x509.set_version(2)
    x509.add_extensions([
        crypto.X509Extension(b'subjectAltName', False,
            ','.join([
                'DNS:%s' % socket.gethostname(),
                'DNS:*.%s' % socket.gethostname(),
                'DNS:localhost',
                'DNS:*.localhost']).encode()),
        crypto.X509Extension(b"basicConstraints", True, b"CA:false")])

    x509.sign(pkey, 'SHA256')

    return (crypto.dump_certificate(crypto.FILETYPE_PEM, x509),
        crypto.dump_privatekey(crypto.FILETYPE_PEM, pkey))

def _gen_cryptography(valid_for):
    import datetime
    from cryptography import x509
    from cryptography.hazmat.backends import default_backend
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.asymmetric import rsa
    from cryptography.hazmat.primitives import serialization
    from cryptography.x509.oid import NameOID

    one_day = datetime.timedelta(1, 0, 0)
    private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048,
            backend=default_backend())
    public_key = private_key.public_key()

    builder = x509.CertificateBuilder()
    builder = builder.subject_name(x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, socket.gethostname())]))
    builder = builder.issuer_name(x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, socket.gethostname())]))
    builder = builder.not_valid_before(datetime.datetime.today() - one_day)
    builder = builder.not_valid_after(datetime.datetime.today() + (one_day*valid_for))
    builder = builder.serial_number(x509.random_serial_number())
    builder = builder.public_key(public_key)
    builder = builder.add_extension(
        x509.SubjectAlternativeName([
            x509.DNSName(socket.gethostname()),
            x509.DNSName('*.%s' % socket.gethostname()),
            x509.DNSName('localhost'),
            x509.DNSName('*.localhost'),
        ]),
        critical=False)
    builder = builder.add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)

    certificate = builder.sign(
        private_key=private_key, algorithm=hashes.SHA256(),
        backend=default_backend())

    return (certificate.public_bytes(serialization.Encoding.PEM),
        private_key.private_bytes(serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption()))

def gen_self_signed_cert(valid_for):
    '''
    Returns (cert, key) as ASCII PEM strings
    '''

    try:
        return _gen_openssl(valid_for)
    except ImportError:
        try:
            return _gen_cryptography(valid_for)
        except ImportError:
            raise

def write_self_signed_cert(fname, cert, key):
    with open(fname+'.cert', 'wb') as f:
        f.write(cert)
    with open(fname+'.key', 'wb') as f:
        f.write(key)
    print(f'wrote {fname}.key and {fname}.cert')


def main():
    if len(sys.argv) > 1:
        fname = sys.argv[1]
    else:
        fname = 'selfsigned'
        print(f'cert name not specified, using {fname}')

    if len(sys.argv) > 2:
        valid_for = int(sys.argv[2])
    else:
        valid_for = 365*5 # 5 years
        print(f'valid_for not specified, using {valid_for} days')

    cert, key = gen_self_signed_cert(valid_for)
    
    write_self_signed_cert(fname, cert, key)


# entrypoint for package installer
def entry():
    main()

if __name__ == "__main__":
    main()
