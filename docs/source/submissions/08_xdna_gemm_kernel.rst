XDNA GEMM Kernel
==========================================

Task 1: Verify Function
------------------------------------------

To verify the correctness of our kernel, we filled the missing code in the ``verify`` function like so:

.. code-block:: python

  ref = in0 @ in1

  if not torch.allclose(out, ref, rtol=0.2, atol=0.5):
      raise ValueError(f"[FAIL] verification did not pass.")

   
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
   * - ``padda [<Pn>], #<imm>``
     - A
     - 1
   * - ``paddb [<Pn>], #<imm>``
     - B
     - 1
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
For this we can use ``dm1`` to ``dm4``, which leaves us ``dm0`` for converting both ``in0`` and ``in1`` tiles.
Since the conversion is not needed until the end of the ``r``-loop, we can reuse the same register for both inputs.

``in0`` is loaded directly as FP32 via ``vlda.conv.fp32.bf16`` into ``dm0``
(two half-registers ``cml0`` / ``cmh0``) and then converted to BFP16 with ``vconv.bfp16ebs8.fp32``.
The two ``p`` tiles produce ``ex10`` (``p=0``) and ``ex11`` (``p=1``).

``in1`` is in ``kn`` layout and must first be transposed to ``nk`` before conversion.
It is loaded into vector half-registers ``x0`` / ``x1`` (q=0) and ``x2`` / ``x3`` (q=1) via ``vldb``,
transposed with four ``vshuffle`` calls (modes 52 and 53 per tile),
element-wise multiplied by a pre-loaded BF16 ones vector (``y4``) using ``vmul.f`` into ``dm0``,
then converted to BFP16 via ``vconv``.
The two ``q`` tiles produce ``ex0`` (``q=0``) and ``ex1`` (``q=1``).

.. list-table:: Register Blocking
   :widths: 20 80
   :header-rows: 1

   * - Tensor
     - Registers
   * - ``out``
     - | ``dm1`` (p=0, q=0), ``dm2`` (p=0, q=1)
       | ``dm3`` (p=1, q=0), ``dm4`` (p=1, q=1)
   * - ``in0``
     - | ``dm0`` (FP32 temporary, shared with ``in1``)
       | ``ex10`` (p=0, BFP16), ``ex11`` (p=1, BFP16)
   * - ``in1``
     - | ``y4`` (= ``x8`` + ``x9``, BF16 ones constant, full 1024-bit Y-register)
       | ``x0``, ``x1`` (q=0 BF16 half-loads, reused each iteration)
       | ``x2``, ``x3`` (q=1 BF16 half-loads, reused each iteration)
       | ``dm0`` (FP32 temporary, shared with ``in0``)
       | ``ex0`` (q=0, BFP16), ``ex1`` (q=1, BFP16)

Task 4: Data Layouts and Pointer Updates
------------------------------------------

The kernel receives three pointer registers: ``p0`` pointing to ``in0``,
``p1`` pointing to ``in1`` and ``p2`` pointing to ``out``.
During setup, ``p3`` is initialised to ``p0 + 1024`` (four ``padda [p3], #256`` instructions)
to serve as a dedicated pointer for the ``in0`` ``p=1`` row.
After the four ``vmac.f`` instructions in each r-iteration, the pointers are advanced:
``p0`` and ``p3`` each increment by 128 bytes (one ``8×8`` BF16 tile) via ``padda``,
and ``p1`` increments by 256 bytes (two ``q`` tiles) via ``paddb``.
The output is written at the end with sequential post-increment stores (``[p2], #64``).

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
     - ``[p0, #0]``, ``[p0, #64]`` (after 1 advance of p0)
   * - ...
     - ...
     - ...
   * - p=0, r=7
     - 896
     - ``[p0, #0]``, ``[p0, #64]`` (after 7 advances of p0)
   * - p=1, r=0
     - 1024
     - ``[p3, #0]``, ``[p3, #64]`` (p3 = p0\ :sub:`initial` + 1024)
   * - ...
     - ...
     - ...
   * - p=1, r=7
     - 1920
     - ``[p3, #0]``, ``[p3, #64]`` (after 7 advances of p3)

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
     - ``[p1, #0]``, ``[p1, #64]`` (after 1 advance of p1)
   * - r=1, q=1
     - 384
     - ``[p1, #128]``, ``[p1, #192]`` (after 1 advance of p1)
   * - ...
     - ...
     - ...
   * - r=7, q=1
     - 1920
     - ``[p1, #128]``, ``[p1, #192]`` (after 7 advances of p1)

General formula: tile (r, q) starts at ``r * 256 + q * 128``.

**out layout (pqmn, BF16)**, loaded and stored via ``p2``:

.. list-table::
   :widths: 20 20 60
   :header-rows: 1

   * - Tile (p, q)
     - Base offset (bytes)
     - Half-load / half-store offsets
   * - p=0, q=0 -> ``dm1``
     - 0
     - ``[p2], #64`` (post-increment store x2)
   * - p=0, q=1 -> ``dm2``
     - 128‚
     - ``[p2], #64`` (post-increment store x2)
   * - p=1, q=0 -> ``dm3``
     - 256
     - ``[p2], #64`` (post-increment store x2)
   * - p=1, q=1 -> ``dm4``
     - 384
     - ``[p2], #64`` (post-increment store x2)

General formula: tile (p, q) starts at ``p * 256 + q * 128``.

Task 5: Implementation
------------------------------------------

The kernel is implemented in ``src/matmul.s``.
It fully unrolls the ``r``-loop (``r=8``) and uses only a final ``ret lr`` as the only control-flow instr‚uction.

For each r-iteration the kernel:

1. Loads the four ``in1`` quarter-tiles via ``vldb`` into ``x0`` / ``x1`` (q=0) and ``x2`` / ``x3`` (q=1), transposes each from ``kn`` to ``nk`` layout with four ``vshuffle`` calls (modes 52 and 53 per tile), multiplies by the pre-loaded BF16 ones vector (``y4``) using ``vmul.f`` into ``dm0``, and converts to BFP16 via ``vconv`` (``ex0`` for q=0, ``ex1`` for q=1).
2. Loads the ``in0`` (p=0) half-tile via ``vlda.conv.fp32.bf16`` into ``dm0`` (``cml0``/``cmh0``) and converts to BFP16 (``ex10``); repeats for (p=1) via ``p3`` into ``ex11``.
3. Issues four ``vmac.f`` instructions to accumulate into ``dm1`` to ``dm4``.
4. Advances ``p0`` and ``p3`` each by 128 bytes (``padda``) and ``p1`` by 256 bytes (``paddb``).

After all eight r-iterations the four accumulators are written back to the output buffer as BF16 via ``vst.conv.bf16.fp32``.

Task 6: Performance
------------------------------------------