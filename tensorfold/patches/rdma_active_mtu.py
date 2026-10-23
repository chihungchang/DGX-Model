"""Negotiate the active RDMA MTU on the pinned TensorFold Python engine.

Upstream hardcodes IBV_MTU_4096. These Sparks have Ethernet MTU 1500 and
verbs active_mtu 1024, so that setting causes transport retry exhaustion.
Exchange each port's active MTU and use the smaller value for each QP.
"""

from pathlib import Path


def patch(path):
    source = path.read_text()
    replacements = [
        ("    ibv_gid gid{};", "    ibv_gid gid{};\n    ibv_mtu mtu = IBV_MTU_256;"),
        (
            "    std::random_device rd;",
            "    ibv_port_attr port_attr{};\n"
            "    TORCH_CHECK(ibv_query_port(nic.ctx, g.port, &port_attr) == 0,\n"
            '                "rdma gather: ibv_query_port failed");\n'
            "    nic.mtu = port_attr.active_mtu;\n"
            "    std::random_device rd;",
        ),
        (
            "        std::vector<py::bytes> gid;",
            "        std::vector<py::bytes> gid;\n        std::vector<int> mtu;",
        ),
        (
            "            psn.push_back(nic.psns[p]);",
            "            psn.push_back(nic.psns[p]);\n"
            "            mtu.push_back(static_cast<int>(nic.mtu));",
        ),
        (
            "py::make_tuple(qpn, psn, gid, (uint64_t)(uintptr_t)g.recv_ring, rkey)",
            "py::make_tuple(qpn, psn, gid, (uint64_t)(uintptr_t)g.recv_ring, rkey, mtu)",
        ),
        (
            "        peer.rkey = remote[p][4].cast<std::vector<uint32_t>>();",
            "        peer.rkey = remote[p][4].cast<std::vector<uint32_t>>();\n"
            "        auto peer_mtus = remote[p][5].cast<std::vector<int>>();\n"
            "        TORCH_CHECK((int)peer_mtus.size() == nd,\n"
            '                    "rdma gather: peer MTU count differs");',
        ),
        (
            "            rtr.path_mtu = IBV_MTU_4096;",
            "            rtr.path_mtu = static_cast<ibv_mtu>(\n"
            "                std::min(static_cast<int>(g.nics[d].mtu), peer_mtus[d]));",
        ),
    ]
    for old, new in replacements:
        if source.count(old) != 1:
            raise RuntimeError("Pinned TensorFold RDMA source changed: " + old)
        source = source.replace(old, new)
    path.write_text(source)


if __name__ == "__main__":
    patch(Path("/opt/tensorfold/src/tensorfold/cuda/rdma_gather.cu"))
