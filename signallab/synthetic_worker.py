"""Lightweight owned-worker bootstrap: lifetime watchdog precedes ML imports."""
import json
import os
import socket
import struct
import sys
import threading

MAX_MESSAGE=128*1024


def _exact(channel,size):
    result=bytearray()
    while len(result)<size:
        part=channel.recv(size-len(result))
        if not part:
            raise EOFError('Supervisor channel closed.')
        result.extend(part)
    return bytes(result)


def receive(channel):
    size=struct.unpack('!I',_exact(channel,4))[0]
    if not 0<size<=MAX_MESSAGE:
        raise ValueError('Worker message exceeds bound.')
    return json.loads(_exact(channel,size))


def send(channel,value):
    data=json.dumps(value,ensure_ascii=False,allow_nan=False,separators=(',', ':')).encode()
    if not 0<len(data)<=MAX_MESSAGE:
        raise ValueError('Worker result exceeds bound.')
    channel.sendall(struct.pack('!I',len(data))+data)


def _watch(read_fd):
    try:
        while os.read(read_fd,1):
            pass
    finally:
        os._exit(70)  # Exit this worker only, even if generation is still active.


def main():
    channel_fd,life_fd,lock_fd=map(int,sys.argv[1:4])
    if lock_fd>=0:
        os.fstat(lock_fd)  # Keep inherited generation lock alive until process exit.
    threading.Thread(target=_watch,args=(life_fd,),daemon=True).start()
    channel=socket.socket(fileno=channel_fd)
    from signallab.synthetic_cli import worker_main
    try:
        worker_main(channel)
    finally:
        channel.close()


if __name__=='__main__':
    main()
