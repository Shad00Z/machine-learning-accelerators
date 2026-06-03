XDNA GEMM
=========

Task 0: Setup
-------------

For this initial task we simply copied our ``matmul.s`` kernel from the previous week and also added the ``verify()`` function to the new driver.


Task 1: MLIR-AIE operations
---------------------------

a) ``aie.tile()``
^^^^^^^^^^^^^^^^^

The ``aie.tile()`` operation creates an **AIE tile** within the AI Engine array.

This operation can either specify ``row`` and ``col`` attributes, denoting the column and row of the tile or it can specify a ``allocation_scheme`` for the tile.
The ``allocation_scheme`` can either be basic-sequential or bank-aware.

A tile encompasses a CoreOp, MemOp, SwitchboxOp, BufferOp and LockOp.

.. code-block:: asm
    :caption: Example usage of ``aie.tile()``

    %tile33 = aie.tile(3, 3)


b) ``aie.core()``
^^^^^^^^^^^^^^^^^

The ``aie.core()`` operation represents an **AIEngine processor core**, which belongs to a tile. 
At the end this operation generates a binary core for each core. 

This operation takes a ``tile`` as an operand and can specify a number of **optional** attributes:

- ``stack_size``: specifies the amount of memory (bytes) reserved for the stack_size
- ``link_with``: (deprecated) value is folded into ``link_files``
- ``link_files``: links to externally-defined functions
- ``elf_file``: specifies the name of the generated binary file
- ``dynamic_objfifo_lowering``: specifies if the ``objectfifos`` of the cores are lowered using the dynamic runtime lowering

.. code-block:: asm
    :caption: Example usage of ``aie.core()`` (1)

    %tile = aie.tile(1, 1)
    %lock11_8 = aie.lock(%tile, 8)
    aie.core(%tile) {
      aie.use_lock(%lock11_8, "Acquire", 1)
      aie.use_lock(%lock11_8, "Release", 0)
      aie.end
    }

.. code-block:: asm
    :caption: Example usage of ``aie.core()`` (2)

    %tile = aie.tile(3, 3)
    aie.core(%tile) {
      aie.end
    } { stackSize = 2048 : i32, elf_file = "core_33.elf" }


c) ``aie.runtime_sequence()``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The instructions that are specified within the ``aie.runtime_sequence()`` operation allow the AI Engine array to be (**re-**) **configured** at runtime (e.g. data movement buffer descriptors). 
These instructions will then execute on the configuration co-processor of the AI Engine array.

This operation has one ``sym_name`` attribute.

.. code-block:: asm
    :caption: Example usage of ``aie.runtime_sequence()``

    aie.runtime_sequence(%arg0: memref<16xi32>) {
      aie.trace.host_config buffer_size=65536
      aie.trace.start_config @trace1
    }


d) ``aie.objectfifo()``
^^^^^^^^^^^^^^^^^^^^^^^

The ``aie.objectfifo()`` creates a **circular buffer** between two tile operations (``aie_tile``). 
This buffer is created between one producer and one or more consumer tile operations. 

The operation takes two operands ``producerTile`` and a arbitrary number of ``consumerTiles``.
There are also 15 attributes that can be specified for this operation. 

.. code-block:: asm
    :caption: Example usage of ``aie.objectfifo()`` (1 to 1)

    aie.objectfifo @of1 (%tile12, { %tile23 }, 4 : i32) : !aie.objectfifo<memref<16xi32>>

.. code-block:: asm
    :caption: Example usage of ``aie.objectfifo()`` (1 to 2 - broadcast)

    aie.objectfifo @of2 (%tile12, { %tile13, %tile23 }, 4 : i32) : !aie.objectfifo<memref<16xi32>>

.. code-block:: asm
    :caption: Example usage of ``aie.objectfifo()`` (attributes)

    aie.objectfifo @of4 (%tile12 dimensionsToStream [<16, 1>, <16, 16>, <1,1>],
                         {
                           %tile13 dimensionsFromStream [],
                           %tile23 dimensionsFromStream [<2, 1>, <128, 2>]
                           }, 2 : i32
                         ) : !aie.objectfifo<memref<256xi32>>


e) ``aie.objectfifo.link()``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The ``aie.objectfifo.link()`` operation links two ``objectFifos``. 
That means that the ``objectFifos`` form **one** dataflow movement, which is split among multiple ``objectFifos``.

The important thing for this operation is that the ``objectFifos`` should have a link point, which can be a shared AIE tile.
This can be achieved throught the attributes for this operation: 

- ``fifoIns``: references the input fifos
- ``fifoOut``: references the input fifos
- ``src_offset``: offset for the source tile
- ``dst_offset``: offset for the destination tile


.. code-block:: asm
    :caption: Example usage of ``aie.objectfifo.link()``

    aie.objectfifo @of1 (%t70, { %t72 }, 2) : !aie.objectfifo<memref<64xi16>>
    aie.objectfifo @of2 (%t72, { %t74 }, 2) : !aie.objectfifo<memref<64xi16>>
    aie.objectfifo.link [@of1] -> [@of2] ([] [])


.. _acquire:

f) ``aie.objectfifo.acquire()``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The ``aie.objectfifo.acquire()`` operation **acquires locks** for the next given number of objects in ``objectFifo``.
Thereby, the lock mode can be chosen based on the port (producer - write, consumer - read).
At the end this operation returns a subview of the acquired objects which can be accessed by them.

The attributes of this operation are:

- ``port``: is the port of an object fifo
- ``objFifo_name``: description of the fifo object
- ``size``: amount of fifo objects / locks to acquire

.. code-block:: asm
    :caption: Example usage of ``aie.objectfifo.acquire()`` 

    %subview = aie.objectfifo.acquire @of1 (Consume, 2) : !aie.objectfifosubview<memref<16xi32>>


g) ``aie.objectfifo.release()``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The ``aie.objectfifo.release()`` operation **releases the locks** that have been acquired for the next given number of objects in the ``objectFifo``.
Here again, the lock mode is chosen based on the port (producer - release for read, consumer - release for write).

This operation shares the same attributes as the :ref:`aie.objectfifo.acquire() <acquire>` operation.

.. code-block:: asm
    :caption: Example usage of ``aie.objectfifo.release()`` 

    aie.objectfifo.release @of1 (Produce, 1)


h) ``aiex.npu.dma_memcpy_nd()``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The ``aiex.npu.dma_memcpy_nd()`` operation programs a direct memory access (DMA) to **access** a memory ``memref`` with an **access pattern** specified by ``offsets``, ``sizes`` and ``strides`` (operands) or alternatively by ``static_offsets``, ``static_sizes`` and ``static_strides`` (attributes).
There are a few specific concepts worth noting:

- *Data Layout Transformation*: is described by the ``sizes`` and ``strides``
- *Automatic Linearization of Contiguous Accesses*: happens automatically through a canonicalization pattern
- *Packet Header Attribute*: specifying the ``packet`` attribute ensures that every DMA BD a packet header is being generated (useful to guide arbitration throughout)

Moreover, this operation has 4 operands and 14 attributes. 

.. code-block:: asm
    :caption: Example usage of ``aiex.npu.dma_memcpy_nd()``

    aiex.npu.dma_memcpy_nd(0, 0, %arg2[1, 1, 0, 0][1, 1, 32, 32][1, 1, 64, 1]) {id = 0 : i64, issue_token = true, metadata = @out0} : memref<32x64xi32>


i) ``aiex.npu.dma_wait()``
^^^^^^^^^^^^^^^^^^^^^^^^^^

The ``aiex.npu.dma_wait()`` operation is a **blocking operation** that **waits** for a DMA to complete its execution. 
``symbol`` specifies which DMA should be waited for. 
Waiting is ensured with the issuing of a task-complete-token (TCT).

.. code-block:: asm
    :caption: Example usage of ``aiex.npu.dma_wait()``

    aiex.npu.dma_wait { symbol = @out0 }

Task 2: Data Layouts and Loops
------------------------------
