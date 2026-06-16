Using the whole NPU
====================

Task 1: Setup of the Whole NPU
--------------------------------

The ``npu2`` device has three types of tiles arranged in 8 columns and 6 rows,
as defined in the `MLIR-AIE device model <https://github.com/Xilinx/mlir-aie/blob/5a61144e13c99c9d430fbd09740c98fcf71d1936/docs/Devices.md?plain=1#L92-L103>`_:

.. code-block:: none
    :caption: Tile layout of the npu2 device

    5 CCCCCCCC
    4 CCCCCCCC
    3 CCCCCCCC
    2 CCCCCCCC
    1 MMMMMMMM
    0 DDDDDDDD
      01234567

These are:

- **D: Shim tiles** (row 0): interface to host memory via DMA (columns 0–7)
- **M: Memory tiles** (row 1): intermediate L2 buffers between the shim and compute tiles (columns 0–7)
- **C: Compute tiles** (rows 2–5): the actual processing elements (8 columns * 4 rows)

Since the task is to use the whole NPU, we first had to create variables for all tiles:

.. code-block:: none
    :caption: Tile variables for the whole NPU
    :linenos:

    // Shim tiles (row 0)
    %shim_noc_tile_0_0 = aie.tile(0, 0)
    %shim_noc_tile_1_0 = aie.tile(1, 0)
    ...
    %shim_noc_tile_7_0 = aie.tile(7, 0)

    // Memory tiles (row 1)
    %mem_tile_0_1 = aie.tile(0, 1)
    %mem_tile_1_1 = aie.tile(1, 1)
    ...
    %mem_tile_7_1 = aie.tile(7, 1)

    // Compute tiles (x=8 columns, y=4 rows -> rows 2–5)
    %tile_0_2 = aie.tile(0, 2)
    %tile_0_3 = aie.tile(0, 3)
    %tile_0_4 = aie.tile(0, 4)
    %tile_0_5 = aie.tile(0, 5)
    %tile_1_2 = aie.tile(1, 2)
    %tile_1_3 = aie.tile(1, 3)
    %tile_1_4 = aie.tile(1, 4)
    %tile_1_5 = aie.tile(1, 5)
    ...
    %tile_7_2 = aie.tile(7, 2)
    %tile_7_3 = aie.tile(7, 3)
    %tile_7_4 = aie.tile(7, 4)
    %tile_7_5 = aie.tile(7, 5)

Next, we needed to implement the core function for each of the 32 compute tiles. 
Since every compute tile performs the same operation, we could simply duplicate the same core function.
We only had to change the loop to use ``%c4 = arith.constant 4 : index`` instead of ``%c128 = arith.constant 128 : index`` 
in order to process 4 ab-iterations per tile instead of 128 (``x`` and ``y`` are now handled spatially, leaving only ``a*b = 4`` sequential iterations).
We also had to correctly connect each compute tile to its corresponding input and output FIFOs:

- ``@in0_L2L1_<col>``: the A-matrix FIFO for its column.
- ``@in1_L2L1_<row>``: the B-matrix FIFO for its row.
- ``@out_L1L2_<col>_<row>``: its own dedicated output FIFO.

For example, the core at column 4, row 0 (tile (4,2)) uses ``@in0_L2L1_4``, ``@in1_L2L1_0``, and ``@out_L1L2_4_0``:

.. code-block:: none
    :caption: Core function for tile (4,2) at column 4, row 0
    :linenos:

    %core_4_2 = aie.core(%tile_4_2) {
      %c0 = arith.constant 0 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c4294967295 step %c1 {
        %c0_1 = arith.constant 0 : index
        %c4 = arith.constant 4 : index
        %c1_1 = arith.constant 1 : index
        scf.for %i_ab = %c0_1 to %c4 step %c1_1 {
          %buffer_out = aie.objectfifo.acquire @out_L1L2_4_0(Produce, 1) : !aie.objectfifosubview<memref<2x2x8x8xbf16>>
          %out = aie.objectfifo.subview.access %buffer_out[0] : !aie.objectfifosubview<memref<2x2x8x8xbf16>> -> memref<2x2x8x8xbf16>
          func.call @zero(%out) : (memref<2x2x8x8xbf16>) -> ()
          %c0_2 = arith.constant 0 : index
          %c16 = arith.constant 16 : index
          %c1_2 = arith.constant 1 : index
          scf.for %i_c = %c0_2 to %c16 step %c1_2 {
            %buffer_in0 = aie.objectfifo.acquire @in0_L2L1_4(Consume, 1) : !aie.objectfifosubview<memref<2x8x8x8xbf16>>
            %in0 = aie.objectfifo.subview.access %buffer_in0[0] : !aie.objectfifosubview<memref<2x8x8x8xbf16>> -> memref<2x8x8x8xbf16>
            %buffer_in1 = aie.objectfifo.acquire @in1_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<8x2x8x8xbf16>>
            %in1 = aie.objectfifo.subview.access %buffer_in1[0] : !aie.objectfifosubview<memref<8x2x8x8xbf16>> -> memref<8x2x8x8xbf16>
            func.call @matmul(%in0, %in1, %out) : (memref<2x8x8x8xbf16>, memref<8x2x8x8xbf16>, memref<2x2x8x8xbf16>) -> ()
            aie.objectfifo.release @in0_L2L1_4(Consume, 1)
            aie.objectfifo.release @in1_L2L1_0(Consume, 1)
          }
          aie.objectfifo.release @out_L1L2_4_0(Produce, 1)
        }
      }
      aie.end
    } {stack_size = 1024 : i32}

Task 2: Broadcasting the Inputs
--------------------------------

In this task, we had to extend the existing FIFOs to support the broadcast pattern of the inputs across the whole NPU.

.. code-block:: none
    :caption: Original FIFOs for the single-tile version
    :linenos:

    aie.objectfifo @in0_L3L2_0(%shim_noc_tile_0_0, {%mem_tile_0_1}, 2 : i32) 
        : !aie.objectfifo<memref<16x64xbf16>>
    aie.objectfifo @in0_L2L1_0(%mem_tile_0_1
        dimensionsToStream [<size = 2, stride = 512>, <size = 8, stride = 8>,
                            <size = 8, stride = 64>, <size = 8, stride = 1>],
        {%tile_0_2}, 2 : i32) : !aie.objectfifo<memref<2x8x8x8xbf16>>
    aie.objectfifo.link [@in0_L3L2_0] -> [@in0_L2L1_0]([] [])
    aie.objectfifo @in1_L3L2_0(%shim_noc_tile_0_0, {%mem_tile_0_1}, 2 : i32) 
        : !aie.objectfifo<memref<64x16xbf16>>
    aie.objectfifo @in1_L2L1_0(%mem_tile_0_1
        dimensionsToStream [<size = 8, stride = 128>, <size = 2, stride = 8>,
                            <size = 8, stride = 16>, <size = 8, stride = 1>],
        {%tile_0_2}, 2 : i32) : !aie.objectfifo<memref<8x2x8x8xbf16>>
    aie.objectfifo.link [@in1_L3L2_0] -> [@in1_L2L1_0]([] [])

To scale this up, we had to duplicate and extend these FIFOs according to the broadcast pattern of each input:

- ``in0`` (A): broadcast per column, so all 4 row tiles in a column share the same A tile
    - Firstly we duplicated ``in0_L3L2`` and ``in0_L2L1`` for each of the 8 columns
    - Secondly we added all 4 row compute tiles to the ``in0_L2L1`` consumer list
- ``in1`` (B): broadcast per row, so all 8 column tiles in a row share the same B tile
    - We created 4 ``in1_L3L2`` / ``in1_L2L1`` FIFO pairs (one per row), each with all 8 column compute tiles as consumers

For ``in0`` column 4, the consumer list expands from ``{%tile_4_2}`` to all 4 tiles in that column:

.. code-block:: none
    :caption: Broadcast FIFOs for ``in0`` column 4
    :linenos:

    // col 4: shim tile 4 -> mem tile 4 -> compute tiles in column 4
    aie.objectfifo @in0_L3L2_4(%shim_noc_tile_4_0, {%mem_tile_4_1}, 2 : i32)
        : !aie.objectfifo<memref<16x64xbf16>>
    aie.objectfifo @in0_L2L1_4(%mem_tile_4_1
        dimensionsToStream [<size = 2, stride = 512>, <size = 8, stride = 8>,
                            <size = 8, stride = 64>, <size = 8, stride = 1>],
        {%tile_4_2, %tile_4_3, %tile_4_4, %tile_4_5}, 2 : i32)
        : !aie.objectfifo<memref<2x8x8x8xbf16>>
    aie.objectfifo.link [@in0_L3L2_4] -> [@in0_L2L1_4]([] [])

For ``in1`` row 0, the consumer list expands to all 8 column tiles in that row:

.. code-block:: none
    :caption: Broadcast FIFOs for ``in1`` row 0
    :linenos:

    // row 0: shim tile 0 -> mem tile 0 -> compute tiles in row 0 (all 8 columns)
    aie.objectfifo @in1_L3L2_0(%shim_noc_tile_0_0, {%mem_tile_0_1}, 2 : i32)
        : !aie.objectfifo<memref<64x16xbf16>>
    aie.objectfifo @in1_L2L1_0(%mem_tile_0_1
        dimensionsToStream [<size = 8, stride = 128>, <size = 2, stride = 8>,
                            <size = 8, stride = 16>, <size = 8, stride = 1>],
        {%tile_0_2, %tile_1_2, %tile_2_2, %tile_3_2,
         %tile_4_2, %tile_5_2, %tile_6_2, %tile_7_2}, 2 : i32)
        : !aie.objectfifo<memref<8x2x8x8xbf16>>
    aie.objectfifo.link [@in1_L3L2_0] -> [@in1_L2L1_0]([] [])

Task 3: Writing the Output
--------------------------------

Before we could test the whole NPU, we had to extend the output FIFO pattern as well.
The original single-tile version had one ``out_L1L2`` FIFO from the compute tile to the memory tile,
and one ``out_L2L3`` FIFO from the memory tile to the shim:

.. code-block:: none
    :caption: Original output FIFOs for the single-tile version
    :linenos:

    aie.objectfifo @out_L1L2_0_0(%tile_0_2, {%mem_tile_0_1}, 2 : i32) 
        : !aie.objectfifo<memref<2x2x8x8xbf16>>
    aie.objectfifo @out_L2L3_0(%mem_tile_0_1
        dimensionsToStream [<size = 2, stride = 128>, <size = 8, stride = 8>,
                            <size = 2, stride = 64>, <size = 8, stride = 1>],
        {%shim_noc_tile_0_0}, 2 : i32) : !aie.objectfifo<memref<16x16xbf16>>
    aie.objectfifo.link [@out_L1L2_0_0] -> [@out_L2L3_0]([] [])

For the full NPU, each of the 32 compute tiles needs its own ``out_L1L2`` FIFO, 
and the 4 row tiles in each column need to be joined into a single ``out_L2L3`` FIFO for that column.
We used ``aie.objectfifo.link`` with write offsets to place each tile's 256-element result 
into a different slot of the shared L2 buffer:

.. code-block:: none
    :caption: Output FIFOs for column 0, joining 4 row tiles
    :linenos:

    // col 0: one output FIFO per row tile
    aie.objectfifo @out_L1L2_0_0(%tile_0_2, {%mem_tile_0_1}, 2 : i32) 
        : !aie.objectfifo<memref<2x2x8x8xbf16>>
    aie.objectfifo @out_L1L2_0_1(%tile_0_3, {%mem_tile_0_1}, 2 : i32) 
        : !aie.objectfifo<memref<2x2x8x8xbf16>>
    aie.objectfifo @out_L1L2_0_2(%tile_0_4, {%mem_tile_0_1}, 2 : i32) 
        : !aie.objectfifo<memref<2x2x8x8xbf16>>
    aie.objectfifo @out_L1L2_0_3(%tile_0_5, {%mem_tile_0_1}, 2 : i32) 
        : !aie.objectfifo<memref<2x2x8x8xbf16>>
    // join all 4 row outputs, write offsets [0, 256, 512, 768]
    aie.objectfifo @out_L2L3_0(%mem_tile_0_1
        dimensionsToStream [<size = 2, stride = 128>, <size = 8, stride = 8>,
                            <size = 2, stride = 64>, <size = 8, stride = 1>],
        {%shim_noc_tile_0_0}, 2 : i32) : !aie.objectfifo<memref<4x16x16xbf16>>
    aie.objectfifo.link [@out_L1L2_0_0, @out_L1L2_0_1, @out_L1L2_0_2, @out_L1L2_0_3]
        -> [@out_L2L3_0]([0, 256, 512, 768] [])

The write offsets place each tile's 256-element ``pqmn`` block into a separate y-slot in the
1024-element L2 buffer, so the combined buffer has layout ``ypqmn``.

The ``dimensionsToStream`` on ``out_L2L3`` reorders the data from ``pqmn`` to ``pmqn`` within each y-slot
as it streams out to the shim, matching the layout the host DMA expects:

.. code-block:: none
    :caption: Dimension order for the output FIFO from L2 to L3
    :linenos:

    dimensionsToStream [<size = 2, stride = 128>,   // p: outer
                        <size = 8, stride = 8>,     // m
                        <size = 2, stride = 64>,    // q
                        <size = 8, stride = 1>]     // n: inner

The ``out_L2L3`` memref grew from ``memref<16x16xbf16>`` (single tile, 256 elements) to
``memref<4x16x16xbf16>`` (4 tiles joined, 1024 elements). The shape describes
the memory layout as ``y * (pm) * (qn)`` where the first dimension indexes the 4 y-slots
(one per row tile), the second covers the 16 pm-rows (p=2, m=8), and the third covers the
16 qn-columns (q=2, n=8).

We repeated this for all 8 columns, resulting in 8 ``out_L2L3`` FIFOs in total.

Task 4: Testing
--------------------------------

With all the tiles, FIFOs, and connections in place, we could finally test the whole NPU:

.. code-block:: bash
    :caption: Testing the whole NPU
    :linenos:

    $ iron-fhs
    Activated virtualenv: /home/mla01/.cache/nix-amd-npu/venv-iron

    $ make run_matmul
    python3 src/driver.py
    [PASS] matmul verification passed.

Group Specific Component
--------------------------------

Pitch 1: Fused cuTile Kernels for Local Diffusion
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- Runnable local text-to-image generator on the NVIDIA Spark (optionally behind a small CLI/Gradio front end)
- Use fused cuTile kernels inside diffusion denoising loop that collapse chained operations into a single pass.
- **Problem**: repeated data round trips to memory for intermediate activations in the denoising step -> memory-bound
- **Solution**: Profile one denoising step to locate the memory-bound chains, then replace them with hand-written fused cuTile kernels:
    - **Kernel 1**: fused GroupNorm -> SiLU (halves the activation round-trips)
    - **Kernel 2 (optional)**: fused LayerNorm -> GEGLU FFN
- Each kernel is correctness-gated against the PyTorch reference before any timing.
- **Deliverable**: a working generator with a baseline <-> fused toggle, plus a short reproducible report

Pitch 2: Unary & Binary Primitives in cuTile
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- Extension of the config object & optimizer pipeline to support first/last touch unary primitives
    - Unary primitives: e.g. ReLU, Sigmoid, Tanh
- Extension of the config object & optimizer pipeline to support binary primitives as main op
    - Binary primitives: e.g. Add, Mul, Min, Max
- Operator fusion: support for fusing first/last touch unary primitive into main op

Pitch 3: XDNA JIT Compiler
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- Implementation of a JIT compiler for XDNA
- Generates assembly code for XDNA for a given config object (from einsum string)
- TODO: restrictions on supported operations, data types, and number of dimensions