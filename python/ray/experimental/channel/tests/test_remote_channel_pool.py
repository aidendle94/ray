import contextlib
from typing import List, Tuple

import pytest
import ray

from ray.dag import DAGContext


@ray.remote
class Dummy:
    def node(self) -> str:
        import ray

        return ray.get_runtime_context().get_node_id()


@contextlib.contextmanager
def patched_traced_channel():
    """Temporarily patch Channel to TracedChannel to observe writes."""
    import ray.experimental.channel.shared_memory_channel as shm
    from ray.experimental.channel.conftest import TracedChannel

    orig = shm.Channel
    shm.Channel = TracedChannel
    try:
        yield shm
    finally:
        shm.Channel = orig


@pytest.mark.parametrize("slots", [1, 2])
def test_composite_channel_uses_pool_when_slots_gt_one(ray_start_regular, slots):
    # Arrange
    writer = Dummy.remote()
    r1 = Dummy.remote()
    r2 = Dummy.remote()

    writer_node = ray.get(writer.node.remote())

    # Force readers to be considered on a different node by passing a different node id.
    readers: List[Tuple[ray.actor.ActorHandle, str]] = [
        (r1, "different-node-1"),
        (r2, "different-node-2"),
    ]

    from ray.experimental.channel.shared_memory_channel import (
        CompositeChannel,
        SharedMemoryType,
        RemoteChannelPool,
    )

    # Set remote slots via DAGContext
    ctx = DAGContext.get_current()
    ctx.num_remote_buffers = slots

    # Act
    # Use the factory to construct CompositeChannel the same way compiled DAG does.
    chan = CompositeChannel(writer, readers, num_shm_buffers=1)
    # Register to avoid lazy init branches
    chan.ensure_registered_as_writer()

    # Assert type selection based on slots
    # CompositeChannel._channels is a set; we expect exactly one remote channel inside.
    channels = list(chan._channels)
    assert len(channels) >= 1

    has_pool = any(isinstance(c, RemoteChannelPool) for c in channels)
    if slots > 1:
        assert has_pool, "RemoteChannelPool should be used when slots>1"
    else:
        assert not has_pool, "Single Channel should be used when slots==1"


def test_remote_channel_pool_round_robin_write(ray_start_regular):
    # Arrange
    writer = Dummy.remote()
    r1 = Dummy.remote()
    r2 = Dummy.remote()

    readers = [(r1, "remote-a"), (r2, "remote-b")]

    from ray.experimental.channel.shared_memory_channel import (
        CompositeChannel,
        RemoteChannelPool,
    )
    from ray.experimental.channel.shared_memory_channel import SharedMemoryType

    DAGContext.get_current().num_remote_buffers = 2

    with patched_traced_channel() as shm:
        # Build CompositeChannel which should internally use RemoteChannelPool
        chan = CompositeChannel(writer, readers, num_shm_buffers=1)
        chan.ensure_registered_as_writer()

        pool = next(c for c in chan._channels if isinstance(c, RemoteChannelPool))
        # Underlying buffers are TracedChannel instances
        bufs = pool._buffers
        assert len(bufs) == 2

        # Act: perform 3 writes; expect round-robin across two buffers
        chan.write(b"x1")
        chan.write(b"x2")
        chan.write(b"x3")

        # Assert: first buffer saw 2 writes, second saw 1 (or vice versa)
        counts = sorted(len(b.ops) for b in bufs)
        assert counts == [1, 2]

