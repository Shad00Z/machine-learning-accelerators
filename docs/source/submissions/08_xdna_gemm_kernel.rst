XDNA GEMM Kernel
==========================================

Task 1: Verify Function
------------------------------------------

Task 2: Instructions and Latencies
------------------------------------------

To gather the latencies, we first looked at `Hello XDNA <https://tnzr.org/xdna/isa.html>`__. 
Here, we were able to find many values in ``Table 7`` and inferred the rest from ``Listing 4``
by counting the number of cycles between dependent instructions.

.. list-table:: Instructions
   :widths: 60 20 20
   :header-rows: 1

   * - Instruction
     - Slot
     - Latency
   * - ``vlda <Xd>, [<Pn>, #<imm>]``
     - A
     - 7
   * - ``vldb <Xd>, [<Pn>, #<imm>]``
     - B
     - 7
   * - ``vlda.conv.fp32.bf16 <CMLd>, [<Pn>, #<imm>]``
     - A
     - 7
   * - ``movxm <Rd>, #<imm32>``
     - XM
     - 1
   * - ``mova <Rd>, #<imm>``
     - A
     - 1
   * - ``vbcst.16 <Xd>, <Rd>``
     - V
     - 1
   * - ``vmov <Xd>, <Xm>``
     - M
     - 2
   * - ``vshuffle <Xd>, <Xr>, <Xs>, <Rn>``
     - M
     - 2
   * - ``vmul.f <DMd>, <Yr>, <Ys>, <Rn>``
     - V
     - 6
   * - ``vconv.bfp16ebs8.fp32 <EXd>, <DMm>``
     - M
     - 4
   * - ``vmac.f <DMd>, <DMm>, <EXr>, <EXs>, <Rn>``
     - V
     - 6
   * - ``vst.conv.bf16.fp32 <CMLd>, [<Pn>, #<imm>]``
     - S
     - 2
   * - ``ret lr``
     - X
     - 6

Task 3: Register Blocking
------------------------------------------

We keep all four ``out`` tiles (``p=2``, ``q=2``) in the accumulator registers for
the whole ``r``-loop, to avoid repeated loads and stores of the output.
For this we can use ``dm0`` to ``dm3``, which leaves us ``dm4`` for converting both ``in0`` and ``in1`` tiles.
Since the conversion is not needed until the end of the ``r``-loop, we can reuse the same register for both inputs.

``in0`` is loaded directly as FP32 via ``vlda.conv.fp32.bf16`` into ``dm4``
(two half-registers ``cml4`` / ``cmh4``) and then converted to BFP16 with ``vconv.bfp16ebs8.fp32``.
The two ``p`` tiles produce ``ex0`` (``p=0``) and ``ex1`` (``p=1``).

``in1`` is in ``kn`` layout and must first be transposed to ``nk`` before conversion.
It is loaded into vector half-registers ``x2`` / ``x3`` (q=0) and ``x4`` / ``x5`` (q=1) via ``vldb``,
transposed with four ``vshuffle`` calls (modes 53 and 52 per tile),
element-wise multiplied by a pre-loaded BF16 ones vector (``y0``) using ``vmul.f`` into ``dm4``,
then converted to BFP16 via ``vconv``.
The two ``q`` tiles produce ``ex2`` (``q=0``) and ``ex3`` (``q=1``).

.. list-table:: Register Blocking
   :widths: 20 80
   :header-rows: 1

   * - Tensor
     - Registers
   * - ``out``
     - | ``dm0`` (p=0, q=0), ``dm1`` (p=0, q=1)
       | ``dm2`` (p=1, q=0), ``dm3`` (p=1, q=1)
   * - ``in0``
     - | ``dm4`` (FP32 temporary, shared with ``in1``)
       | ``ex0`` (p=0, BFP16), ``ex1`` (p=1, BFP16)
   * - ``in1``
     - | ``y0`` (BF16 ones constant, full 1024-bit Y-register)
       | ``x2``, ``x3`` (q=0 BF16 half-loads, reused each iteration)
       | ``x4``, ``x5`` (q=1 BF16 half-loads, reused each iteration)
       | ``dm4`` (FP32 temporary, shared with ``in0``)
       | ``ex2`` (q=0, BFP16), ``ex3`` (q=1, BFP16)

Task 4: Data Layouts and Pointer Updates
------------------------------------------

The kernel receives three pointer registers: ``p0`` pointing to ``in0``,
``p1`` pointing ``in1`` and ``p2`` pointing to ``out``.
Since the ``r``-loop is fully unrolled (``r=8``, no control flow), all tile accesses use
fixed immediate offsets from these base pointers and no pointer register updates are required.

Each ``8 x 8`` BF16 tile is 128 bytes. Each half-load (``vlda.conv.fp32.bf16`` or ``vldb``) transfers
64 bytes, so two half-loads are needed per tile.

**in0 layout (prmk, BF16)**, accessed via ``p0``

.. list-table::
   :widths: 20 20 60
   :header-rows: 1

   * - Tile (p, r)
     - Base offset (bytes)
     - Half-load offsets
   * - p=0, r=0
     - 0
     - ``[p0, #0]``, ``[p0, #64]``
   * - p=0, r=1
     - 128
     - ``[p0, #128]``, ``[p0, #192]``
   * - ...
     - ...
     - ...
   * - p=0, r=7
     - 896
     - ``[p0, #896]``, ``[p0, #960]``
   * - p=1, r=0
     - 1024
     - ``[p0, #1024]``, ``[p0, #1088]``
   * - ...
     - ...
     - ...
   * - p=1, r=7
     - 1920
     - ``[p0, #1920]``, ``[p0, #1984]``

General formula: tile (p, r) starts at ``p * 1024 + r * 128``.

**in1 layout (rqkn, BF16)**, accessed via ``p1``

.. list-table::
   :widths: 20 20 60
   :header-rows: 1

   * - Tile (r, q)
     - Base offset (bytes)
     - Half-load offsets
   * - r=0, q=0
     - 0
     - ``[p1, #0]``, ``[p1, #64]``
   * - r=0, q=1
     - 128
     - ``[p1, #128]``, ``[p1, #192]``
   * - r=1, q=0
     - 256
     - ``[p1, #256]``, ``[p1, #320]``
   * - r=1, q=1
     - 384
     - ``[p1, #384]``, ``[p1, #448]``
   * - ...
     - ...
     - ...
   * - r=7, q=1
     - 1920
     - ``[p1, #1920]``, ``[p1, #1984]``

General formula: tile (r, q) starts at ``r * 256 + q * 128``.

**out layout (pqmn, BF16)**, loaded and stored via ``p2``:

.. list-table::
   :widths: 20 20 60
   :header-rows: 1

   * - Tile (p, q)
     - Base offset (bytes)
     - Half-load / half-store offsets
   * - p=0, q=0 -> ``dm0``
     - 0
     - ``[p2, #0]``, ``[p2, #64]``
   * - p=0, q=1 -> ``dm1``
     - 128
     - ``[p2, #128]``, ``[p2, #192]``
   * - p=1, q=0 -> ``dm2``
     - 256
     - ``[p2, #256]``, ``[p2, #320]``
   * - p=1, q=1 -> ``dm3``
     - 384
     - ``[p2, #384]``, ``[p2, #448]``

General formula: tile (p, q) starts at ``p * 256 + q * 128``.

Task 5: Implementation
------------------------------------------

The kernel is implemented in ``src/matmul.s``.
It fully unrolls the ``r``-loop (``r=8``) and uses only a final ``ret lr`` as the sole control-flow instruction.

For each ``r``-step the kernel:

1. Loads the two ``in0`` half-tiles (p=0, p=1) via ``vlda.conv.fp32.bf16`` into ``dm4`` and converts to BFP16 (``ex0``, ``ex1``).
2. Loads the two ``in1`` half-tiles (q=0, q=1) via ``vldb`` into half-registers ``x2``/``x3`` (q=0) and ``x4``/``x5`` (q=1), transposes each from ``kn`` to ``nk`` layout with four ``vshuffle`` calls (modes 53 and 52 per tile), multiplies by a pre-loaded BF16 ones vector (``y0``) using ``vmul.f`` into ``dm4``, then converts to BFP16 via ``vconv`` (``ex2``, ``ex3``).
3. Issues four ``vmac.f`` instructions to accumulate into ``dm0``–``dm3``.

After all eight r-steps the four accumulators are written back to the output buffer as BF16 via ``vst.conv.bf16.fp32``.

Task 6: Performance
------------------------------------------