import asyncio
import contextlib
import time

import pytest

ray = pytest.importorskip("ray")

from ray.dag import InputNode


@ray.remote
class Echo:
    def echo(self, x):
        return x


@contextlib.contextmanager
def force_remote_split():
    """Force CompositeChannel to treat readers as different-node.

    We monkeypatch split_actors_by_node_locality to put all readers into
    the different-node list so CompositeChannel uses the remote path.
    """
    import ray.experimental.channel.utils as utils

    orig = utils.split_actors_by_node_locality

    def _forced(node, actor_and_node_list):
        return [], actor_and_node_list

    utils.split_actors_by_node_locality = _forced
    try:
        yield
    finally:
        utils.split_actors_by_node_locality = orig


def test_compiled_dag_two_execs_with_remote_pool(ray_start_regular):
    from ray.dag import DAGContext

    # Enable remote pool with 2 slots
    DAGContext.get_current().num_remote_buffers = 2

    a = Echo.remote()
    b = Echo.remote()

    # Simple 2-stage chain
    with InputNode() as inp:
        dag = b.echo.bind(a.echo.bind(inp))

    compiled = dag.experimental_compile()

    # Force remote path so CompositeChannel uses RemoteChannelPool
    with force_remote_split():
        # Execute twice quickly; should not hit RayChannelTimeoutError
        r1 = compiled.execute(b"m1")
        r2 = compiled.execute(b"m2")
        out1, out2 = ray.get([r1, r2])
        assert out1 == b"m1"
        assert out2 == b"m2"

