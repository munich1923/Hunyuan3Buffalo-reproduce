"""Loopback-only HTTP CONNECT bridge to an SSH reverse dynamic SOCKS forward.
No TLS interception and no request URL logging. Stop this helper when downloads finish.
"""
import select,socket,socketserver,struct
from remote_runtime import ROOT
def recv_exact(s,n):
    out=b""
    while len(out)<n:
        b=s.recv(n-len(out))
        if not b:raise ConnectionError("SOCKS relay closed")
        out+=b
    return out
class Handler(socketserver.BaseRequestHandler):
    def handle(self):
        client=self.request;remote=None
        try:
            client.settimeout(60);head=b""
            while b"\r\n\r\n" not in head:
                head+=client.recv(4096)
                if len(head)>65536:raise ValueError("Header too large")
                if not head:raise ConnectionError("empty header")
            method,address,_=head.split(b"\r\n",1)[0].decode("ascii").split()
            if method!="CONNECT":raise ValueError("HTTPS CONNECT only")
            host,port=address.rsplit(":",1);encoded=host.encode("idna")
            remote=socket.create_connection(("127.0.0.1",19790),timeout=30)
            remote.sendall(b"\x05\x01\x00")
            if recv_exact(remote,2)!=b"\x05\x00":raise ConnectionError("SOCKS negotiation failed")
            remote.sendall(b"\x05\x01\x00\x03"+bytes([len(encoded)])+encoded+struct.pack(">H",int(port)))
            reply=recv_exact(remote,4)
            if reply[1]!=0:raise ConnectionError("SOCKS target connection failed")
            typ=reply[3];size=4 if typ==1 else 16 if typ==4 else recv_exact(remote,1)[0]
            recv_exact(remote,size+2)
            client.sendall(b"HTTP/1.1 200 Connection Established\r\n\r\n")
            client.settimeout(None);remote.settimeout(None)
            while True:
                readable,_,_=select.select([client,remote],[],[],300)
                if not readable:return
                for sock in readable:
                    data=sock.recv(65536)
                    if not data:return
                    (remote if sock is client else client).sendall(data)
        except Exception:
            try:client.sendall(b"HTTP/1.1 502 Bad Gateway\r\nConnection: close\r\nContent-Length: 0\r\n\r\n")
            except OSError:pass
        finally:
            if remote:remote.close()
class Server(socketserver.ThreadingTCPServer):
    allow_reuse_address=True;daemon_threads=True
if __name__=="__main__":
    with Server(("127.0.0.1",19791),Handler) as server:
        print("Loopback HTTPS relay ready on 19791 -> SSH SOCKS 19790",flush=True)
        server.serve_forever()
