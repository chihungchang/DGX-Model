#!/usr/bin/env python3
"""Exercise TensorFold's two-port GPU gather without loading model weights.

Run the same file in the TensorFold image on each Spark with ranks 0 and 1.
It checks exact values for eager and captured gathers, including ring reuse.
"""

import argparse
from datetime import timedelta
import json

import torch
from torch.distributed import TCPStore
from tensorfold.cuda.rdma import RdmaGather


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("rank", type=int, choices=[0, 1])
    parser.add_argument("--master", default="192.168.100.10")
    parser.add_argument("--port", type=int, default=29559)
    args = parser.parse_args()
    torch.cuda.set_device(0)
    store = TCPStore(args.master, args.port, 2, args.rank == 0,
                     timeout=timedelta(seconds=90))
    gather = RdmaGather(store, args.rank, 2, max_bytes=1024 * 1024)
    sizes = [4, 6144, 5 * 6144, 17 * 6144, 32 * 6144]
    for count in sizes:
        send = torch.full((count,), args.rank + 1, device="cuda", dtype=torch.float32)
        recv = torch.empty(2, count, device="cuda", dtype=torch.float32)
        for _ in range(8):
            gather.all_gather(send, recv)
            torch.cuda.synchronize()
            assert torch.equal(recv[0], torch.ones_like(send))
            assert torch.equal(recv[1], torch.full_like(send, 2))
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            gather.all_gather(send, recv)
        for iteration in range(16):
            base = iteration + 3
            send.fill_(base + args.rank)
            graph.replay()
            torch.cuda.synchronize()
            assert torch.equal(recv[0], torch.full_like(send, base))
            assert torch.equal(recv[1], torch.full_like(send, base + 1))
        print(json.dumps({"rank": args.rank, "elements": count,
                          "eager_and_graph": "pass"}), flush=True)
    assert not gather.failure(), gather.failure()
    store.set(f"done/{args.rank}", "ok")
    store.get(f"done/{1 - args.rank}")
    print(json.dumps({"rank": args.rank, "rdma": "pass", "devices": gather.devices}), flush=True)


if __name__ == "__main__":
    main()
