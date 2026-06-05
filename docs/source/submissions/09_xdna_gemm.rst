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


Task 3: Implementation
----------------------

a) Data Movement
^^^^^^^^^^^^^^^^

To adjust the ``driver.py`` to the new dimension sizes we simply had to replace the ones present with the new values. 

.. code-block:: py

    data_in0 = torch.randn( 256, 1024, dtype=torch.bfloat16)
    data_in1 = torch.randn(1024,  128, dtype=torch.bfloat16)
    data_out = torch.zeros( 256,  128, dtype=torch.bfloat16)

Implementing the data movement inside ``matmul.mlir`` proved to be more challenging.
We start with the todo's inside the core unit. 

.. code-block:: asm

    %core_0_2 = aie.core(%tile_0_2) {
      %c0 = arith.constant 0 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c4294967295 step %c1 {
        %ab = arith.constant 128 : index
        scf.for %arg1 = %c0 to %ab step %c1 {
          ...
          %c = arith.constant 16 : index
          scf.for %arg2 = %c1 to %c step %c1 {
            ...
          }
          aie.objectfifo.release @out_L1L2_0_0(Produce, 1)
        }
      }
      aie.end
    }

For that we looked at how many ``16x16`` tiles fit into the ``out`` matrix. 
In this specific case we had ``256x128``, which boils down to ``16x8`` and this results in ``128`` iterations.

To get a correct result we then had to fill in the inner most loop. 
This loop accumulates the result in the currently acquired memory. 
As the contraction dimension is ``K=1024`` and we always load ``64`` elements in the ``K`` dimension, we ultimately have ``16`` iterations to perform. 

After implementing the loops for the core, we moved on to the ``runtime_sequence``.
The first step was to adjust the ``memref`` sizes accordingly.

.. code-block:: asm 
    :capation: ``memref`` sizes

    aie.runtime_sequence(%arg0: memref<256x1024xbf16>, %arg1: memref<1024x128xbf16>, %arg2: memref<256x128xbf16>)

After that we enhance the data movement operations. 
As we did not want to write an endless number of loops we decided to increase the sizes for the ``dma_memcpy_nd`` operations. 

.. code-block:: asm
    :caption: data movement operations

    // =========================================================================
    // LONG-RUNNING ASYNC OUTPUT (Reserved exclusively on ID 0)
    // =========================================================================
    aiex.npu.dma_memcpy_nd(%arg2[0, 0, 0, 0][16, 8, 16, 16][2048, 16, 128, 1]) {id = 0 : i64, metadata = @out_L2L3_0} : memref<256x128xbf16>

    // =========================================================================
    // BATCH 1: First 4 Pairs (Rows 0 to 63) -> IDs 1-8
    // =========================================================================
    aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 0][8, 16, 16, 64][0, 64, 1024, 1]) {id = 1 : i64, metadata = @in0_L3L2_0} : memref<256x1024xbf16>
    aiex.npu.dma_memcpy_nd(%arg1[0, 0, 0, 0][8, 16, 64, 16][16, 8192, 128, 1]) {id = 2 : i64, metadata = @in1_L3L2_0} : memref<1024x128xbf16>
    aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 16384][8, 16, 16, 64][0, 64, 1024, 1]) {id = 3 : i64, metadata = @in0_L3L2_0} : memref<256x1024xbf16>
    aiex.npu.dma_memcpy_nd(%arg1[0, 0, 0, 0][8, 16, 64, 16][16, 8192, 128, 1]) {id = 4 : i64, metadata = @in1_L3L2_0} : memref<1024x128xbf16>
    aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 32768][8, 16, 16, 64][0, 64, 1024, 1]) {id = 5 : i64, metadata = @in0_L3L2_0} : memref<256x1024xbf16>
    aiex.npu.dma_memcpy_nd(%arg1[0, 0, 0, 0][8, 16, 64, 16][16, 8192, 128, 1]) {id = 6 : i64, metadata = @in1_L3L2_0} : memref<1024x128xbf16>
    aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 49152][8, 16, 16, 64][0, 64, 1024, 1]) {id = 7 : i64, metadata = @in0_L3L2_0, issue_token = true} : memref<256x1024xbf16>
    aiex.npu.dma_memcpy_nd(%arg1[0, 0, 0, 0][8, 16, 64, 16][16, 8192, 128, 1]) {id = 8 : i64, metadata = @in1_L3L2_0, issue_token = true} : memref<1024x128xbf16>

    aiex.npu.dma_wait {symbol = @in0_L3L2_0}
    aiex.npu.dma_wait {symbol = @in1_L3L2_0}

    // =========================================================================
    // BATCH 2: Next 4 Pairs (Rows 64 to 127) -> Reuse IDs 1-8 safely
    // =========================================================================
    aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 65536][8, 16, 16, 64][0, 64, 1024, 1]) {id = 1 : i64, metadata = @in0_L3L2_0} : memref<256x1024xbf16>
    aiex.npu.dma_memcpy_nd(%arg1[0, 0, 0, 0][8, 16, 64, 16][16, 8192, 128, 1]) {id = 2 : i64, metadata = @in1_L3L2_0} : memref<1024x128xbf16>
    aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 81920][8, 16, 16, 64][0, 64, 1024, 1]) {id = 3 : i64, metadata = @in0_L3L2_0} : memref<256x1024xbf16>
    aiex.npu.dma_memcpy_nd(%arg1[0, 0, 0, 0][8, 16, 64, 16][16, 8192, 128, 1]) {id = 4 : i64, metadata = @in1_L3L2_0} : memref<1024x128xbf16>
    aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 98304][8, 16, 16, 64][0, 64, 1024, 1]) {id = 5 : i64, metadata = @in0_L3L2_0} : memref<256x1024xbf16>
    aiex.npu.dma_memcpy_nd(%arg1[0, 0, 0, 0][8, 16, 64, 16][16, 8192, 128, 1]) {id = 6 : i64, metadata = @in1_L3L2_0} : memref<1024x128xbf16>
    aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 114688][8, 16, 16, 64][0, 64, 1024, 1]) {id = 7 : i64, metadata = @in0_L3L2_0, issue_token = true} : memref<256x1024xbf16>
    aiex.npu.dma_memcpy_nd(%arg1[0, 0, 0, 0][8, 16, 64, 16][16, 8192, 128, 1]) {id = 8 : i64, metadata = @in1_L3L2_0, issue_token = true} : memref<1024x128xbf16>

    aiex.npu.dma_wait {symbol = @in0_L3L2_0}
    aiex.npu.dma_wait {symbol = @in1_L3L2_0}

    // =========================================================================
    // BATCH 3: Next 4 Pairs (Rows 128 to 191) -> Reuse IDs 1-8 safely
    // =========================================================================
    aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 131072][8, 16, 16, 64][0, 64, 1024, 1]) {id = 1 : i64, metadata = @in0_L3L2_0} : memref<256x1024xbf16>
    aiex.npu.dma_memcpy_nd(%arg1[0, 0, 0, 0][8, 16, 64, 16][16, 8192, 128, 1]) {id = 2 : i64, metadata = @in1_L3L2_0} : memref<1024x128xbf16>
    aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 147456][8, 16, 16, 64][0, 64, 1024, 1]) {id = 3 : i64, metadata = @in0_L3L2_0} : memref<256x1024xbf16>
    aiex.npu.dma_memcpy_nd(%arg1[0, 0, 0, 0][8, 16, 64, 16][16, 8192, 128, 1]) {id = 4 : i64, metadata = @in1_L3L2_0} : memref<1024x128xbf16>
    aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 163840][8, 16, 16, 64][0, 64, 1024, 1]) {id = 5 : i64, metadata = @in0_L3L2_0} : memref<256x1024xbf16>
    aiex.npu.dma_memcpy_nd(%arg1[0, 0, 0, 0][8, 16, 64, 16][16, 8192, 128, 1]) {id = 6 : i64, metadata = @in1_L3L2_0} : memref<1024x128xbf16>
    aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 180224][8, 16, 16, 64][0, 64, 1024, 1]) {id = 7 : i64, metadata = @in0_L3L2_0, issue_token = true} : memref<256x1024xbf16>
    aiex.npu.dma_memcpy_nd(%arg1[0, 0, 0, 0][8, 16, 64, 16][16, 8192, 128, 1]) {id = 8 : i64, metadata = @in1_L3L2_0, issue_token = true} : memref<1024x128xbf16>

    aiex.npu.dma_wait {symbol = @in0_L3L2_0}
    aiex.npu.dma_wait {symbol = @in1_L3L2_0}

    // =========================================================================
    // BATCH 4: Final 4 Pairs (Rows 192 to 255) -> Reuse IDs 1-8 safely
    // =========================================================================
    aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 196608][8, 16, 16, 64][0, 64, 1024, 1]) {id = 1 : i64, metadata = @in0_L3L2_0} : memref<256x1024xbf16>
    aiex.npu.dma_memcpy_nd(%arg1[0, 0, 0, 0][8, 16, 64, 16][16, 8192, 128, 1]) {id = 2 : i64, metadata = @in1_L3L2_0} : memref<1024x128xbf16>
    aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 212992][8, 16, 16, 64][0, 64, 1024, 1]) {id = 3 : i64, metadata = @in0_L3L2_0} : memref<256x1024xbf16>
    aiex.npu.dma_memcpy_nd(%arg1[0, 0, 0, 0][8, 16, 64, 16][16, 8192, 128, 1]) {id = 4 : i64, metadata = @in1_L3L2_0} : memref<1024x128xbf16>
    aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 229376][8, 16, 16, 64][0, 64, 1024, 1]) {id = 5 : i64, metadata = @in0_L3L2_0} : memref<256x1024xbf16>
    aiex.npu.dma_memcpy_nd(%arg1[0, 0, 0, 0][8, 16, 64, 16][16, 8192, 128, 1]) {id = 6 : i64, metadata = @in1_L3L2_0} : memref<1024x128xbf16>
    aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 245760][8, 16, 16, 64][0, 64, 1024, 1]) {id = 7 : i64, metadata = @in0_L3L2_0, issue_token = true} : memref<256x1024xbf16>
    aiex.npu.dma_memcpy_nd(%arg1[0, 0, 0, 0][8, 16, 64, 16][16, 8192, 128, 1]) {id = 8 : i64, metadata = @in1_L3L2_0, issue_token = true} : memref<1024x128xbf16>

    // =========================================================================
    // FINAL CLEANUP AND VERIFICATION
    // =========================================================================
    aiex.npu.dma_wait {symbol = @in0_L3L2_0}
    aiex.npu.dma_wait {symbol = @in1_L3L2_0}
    aiex.npu.dma_wait {symbol = @out_L2L3_0}

The important things to notice are:

- We are using only the buffer descriptor IDs ``0-9``, as this allows us to evenly distribute the data movements between the ``aiex.npu.dma_wait {symbol = ...}`` instructions.
- It is necessary that the last two ``aiex.npu.dma_memcpy_nd`` operations in a block are enhanced to create a token (``issue_token``), which can then be used again for the synchronization of the data movement.

.. code-block:: asm
    :caption: data movement synchronization

    // =========================================================================
      // BATCH 1: First 4 Pairs (Rows 0 to 63) -> IDs 1-8
      // =========================================================================
      aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 0][8, 16, 16, 64][0, 64, 1024, 1]) {id = 1 : i64, metadata = @in0_L3L2_0} : memref<256x1024xbf16>
      ...
      aiex.npu.dma_memcpy_nd(%arg1[0, 0, 0, 0][8, 16, 64, 16][16, 8192, 128, 1]) {id = 6 : i64, metadata = @in1_L3L2_0} : memref<1024x128xbf16>
      aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 49152][8, 16, 16, 64][0, 64, 1024, 1]) {id = 7 : i64, metadata = @in0_L3L2_0, issue_token = true} : memref<256x1024xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 0, 0, 0][8, 16, 64, 16][16, 8192, 128, 1]) {id = 8 : i64, metadata = @in1_L3L2_0, issue_token = true} : memref<1024x128xbf16>

      aiex.npu.dma_wait {symbol = @in0_L3L2_0}
      aiex.npu.dma_wait {symbol = @in1_L3L2_0}

b) Verifying Correctness
^^^^^^^^^^^^^^^^^^^^^^^^

Based on our understanding of the architecture (XDNA2), the ``matmul`` kernel from the previous assignment and the row-major data layout we expected everything to work like this. 

However, as this was not the case, we enhanced our ``driver.py`` slightly to help us debug our current ``matmul.mlir``.

.. code-block:: py
    :caption: debug code

    print("\n" + "="*40)
    print("       NPU DEBUG DIAGNOSTICS        ")
    print("="*40)
    
    # 1. Check for complete silence (All Zeros)
    is_all_zero = torch.all(out == 0).item()
    print(f"Is output completely zeros?: {is_all_zero}")
    
    # 2. Check fill percentage
    nonzero_count = torch.count_nonzero(out).item()
    total_elements = out.numel()
    print(f"Active elements: {nonzero_count} / {total_elements} ({nonzero_count/total_elements*100:.2f}%)")
    
    # 3. Value Range Comparison
    print(f"NPU Output Range: Min = {out.min().item():.4f}, Max = {out.max().item():.4f}")
    print(f"CPU Reference Range: Min = {ref.min().item():.4f}, Max = {ref.max().item():.4f}")
    
    # 4. Error Metrics
    abs_diff = torch.abs(out - ref)
    print(f"Max Absolute Error: {abs_diff.max().item():.4f}")
    print(f"Mean Absolute Error: {abs_diff.mean().item():.4f}")
    
    # 5. Row-by-Row Spatial Analysis
    failing_rows = []
    for r in range(out.shape[0]):
        if not torch.allclose(out[r], ref[r], rtol=0.5, atol=2):
            failing_rows.append(r)
            
    print(f"Failing Rows: {len(failing_rows)} / {out.shape[0]}")
    if len(failing_rows) > 0:
        print(f"First 10 failing row indices: {failing_rows[:10]}")
    print("="*40 + "\n")

Based on these debug messages we could verify that we are working on our whole memory, but since the matmul still failed, we knew something was wrong.

.. code-block:: txt
    :caption: initial debug response

    ========================================
    Is output completely zeros?: False
    Active elements: 32768 / 32768 (100.00%)
    NPU Output Range: Min = -1560.0000, Max = 1064.0000
    CPU Reference Range: Min = -131.0000, Max = 134.0000
    Max Absolute Error: 1576.0000
    Mean Absolute Error: 199.0000
    Failing Rows: 256 / 256
    First 10 failing row indices: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    ========================================

After performing some experiments, we realized that there is an initialization problem for our accumulation registers (``dm0-dm4``).
Therefore, we added an additional clearing of the ``dm1-dm4`` registers with the ``vlcr`` instruction.

.. code-block:: asm
    :caption: accumulation register reset

    matmul_init:
        vclr	dm1
        vclr	dm2
        vclr	dm3
        vclr	dm4

However, we can't just apply these changes every time, but everytime before the 16 accumulation loops.
That is why we also enhanced ``matmul.mlir`` file accordingly:

.. code-block:: asm
    :caption: memory reset

    module {
        memref.global @in0_L3L2_ping : memref<256x1024xbf16>
        memref.global @in0_L3L2_pong : memref<256x1024xbf16>
        aie.device(npu2) {
            ...

            %core_0_2 = aie.core(%tile_0_2) {
                ...
                scf.for %arg0 = %c0 to %c4294967295 step %c1 {
                    ...
                    scf.for %arg1 = %c0 to %ab step %c1 {
                        ...
                        // first K-iteration: clears accumulators
                        %b0i0 = aie.objectfifo.acquire @in0_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<2x8x8x8xbf16>>
                        %i0_0 = aie.objectfifo.subview.access %b0i0[0] : !aie.objectfifosubview<memref<2x8x8x8xbf16>> -> memref<2x8x8x8xbf16>
                        %b0i1 = aie.objectfifo.acquire @in1_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<8x2x8x8xbf16>>
                        %i1_0 = aie.objectfifo.subview.access %b0i1[0] : !aie.objectfifosubview<memref<8x2x8x8xbf16>> -> memref<8x2x8x8xbf16>
                        func.call @matmul_init(%i0_0, %i1_0, %out) : (memref<2x8x8x8xbf16>, memref<8x2x8x8xbf16>, memref<2x2x8x8xbf16>) -> ()
                        aie.objectfifo.release @in0_L2L1_0(Consume, 1)
                        aie.objectfifo.release @in1_L2L1_0(Consume, 1)

                        // remaining 15 K-iterations (1024 / 64): accumulate
                        %c = arith.constant 16 : index
                        scf.for %arg2 = %c1 to %c step %c1 {
                            ...
                        }
                        aie.objectfifo.release @out_L1L2_0_0(Produce, 1)
                    }
                }
                aie.end
                } {stack_size = 1024 : i32}

                aie.runtime_sequence(%arg0: memref<256x1024xbf16>, %arg1: memref<1024x128xbf16>, %arg2: memref<256x128xbf16>) {
                    ...
                }
            }
        }

After integrating both of these changes we could run ``make run_matmul`` and everything matches.

