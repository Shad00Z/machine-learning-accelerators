Inferring the VLIW ISA of XDNA2
===============================

With this assignment, the second half of the lecture starts.
From now on, we are working with the VLIW instruction set architecture of the XDNA2 compute tile.

Task 1: Vector-Add Kernel
-------------------------

a) Element-wise Vector Addition
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To start off the implementation on the XDNA2 architecture, we are implementing a simple vector addition kernel.
Most of the code is already given, so we only have to make ourselves familiar with the `aie::add <https://download.amd.com/docnav/aiengine/xilinx2025_2/aiengine_api/aie_api/doc/group__group__arithmetic.html#gafd9f3351ca16d4398e29251c6b903663>`_ instruction.

We fill in the implementation gap by adding the ``aie::add`` instruction to the code:

.. code-block:: cpp

    // load data
    v_in0 = aie::load_v<r>(ptr_in0);
    v_in1 = aie::load_v<r>(ptr_in1);

    // element-wise addition using the AIE-API
    v_out = aie::add(v_in0, v_in1);

    // store data
    aie::store_v(ptr_out, v_out);

b) Compiling to Assembly
^^^^^^^^^^^^^^^^^^^^^^^^

Following the implementation, we compile the vector add function to assembly using: 

.. code-block:: bash

    make asm_vadd

Running this command generates an assembly file in the ``build`` directory. 
Within that file we can see that the mnemonic for the BF16 element-wise addition is actually ``vadd.f``.

.. code-block:: asm

    vadd.f	dm0, dm0, dm1, r0

c) Result Verification
^^^^^^^^^^^^^^^^^^^^^^

The last step is to implement the ``verify()`` function in the ``driver.py`` file.

.. _vadd_verify:

.. code-block:: python 

    ref = 0

    if kernel == "vadd":
        ref = in0 + in1
    else:
        raise NotImplementedError("verify() not yet implemented")

    print(f"  in0[:8]  = {in0[:8]}")
    print(f"  in1[:8]  = {in1[:8]}")
    print(f"  ref[:8]  = {ref[:8]}")
    print(f"  out[:8]  = {out[:8]}")
    
    if not torch.allclose(out, ref, rtol=1e-2, atol=1e-2):
        raise ValueError(f"[FAIL] {kernel} verification did not pass.")

Task 2: Identify VLIW Slots
---------------------------

For this task we are looking deeper into the XDNA2 VLIW and its functional slots. 

The key insight is that the mnemonic name of the instruction (``vlda``) gives us hints to which functional unit it belongs (occupied).

.. _vliw_slots:

.. list-table:: VLIW Slots
   :widths: 20 10 20 30
   :header-rows: 1

   * - Functional Unit
     - Slot
     - NOP mnemonic
     - Occupied in the second instruction?
   * - Vector Unit
     - V
     - ``nopv``
     - No
   * - Load Unit A
     - A
     - ``nopa``
     - Yes (``vlda.conv.fp32.bf16``)
   * - Load Unit B
     - B
     - ``nopb``
     - No
   * - Store Unit
     - S
     - ``nops``
     - No
   * - Scalar / Control Unit
     - X (XM)
     - ``nopxm`` / ``nopx``
     - No
   * - Movement Unit
     - M (XM)
     - ``nopxm``
     - No


Task 3: Identify Instructions and Register Classes per Slot
-----------------------------------------------------------

a) Instruction Table
^^^^^^^^^^^^^^^^^^^^

As described in the previous task, to find out the slot that is occupied, we have to look at the instruction itself.
For instructions where there is no direct indication, we thought about the most logical Slot by looking at the table with the :ref:`VLIW Slots <vliw_slots>`.

.. list-table:: Instructions
   :widths: 50 10 40
   :header-rows: 1

   * - Instruction
     - Slot
     - Short description
   * - ``vlda.conv.fp32.bf16 cml0 [p0, #0]``
     - A
     - | ``vlda`` vector load (load unit A)
       | ``.conv.fp32.bf16`` convert from FP32 to BF16
       | ``cml0`` destination acc register
       | ``[p0, #0]`` pointer register and byte offset
   * - ``movx r6, #1``
     - X
     - | ``movx`` move immediate (copy)
       | ``r6`` destination scalar register
       | ``#1`` immediate value to load
   * - ``vldb x1, [p1, #0]``
     - B
     - | ``vldb`` vector load (load unit B)
       | ``x1`` destination vector register
       | ``[p1, #0]`` pointer register and byte offset
   * - ``vmov bmhl2, bmhh4``
     - V
     - | ``vmov`` vector move
       | ``bmhl2`` destination acc register (lower half)
       | ``bmhh4`` source acc register (upper half)
   * - ``mova r0, #60``
     - A
     - | ``mova`` move immediate (A-Slot)
       | ``r0`` destination scalar register
       | ``#60`` immediate value
   * - ``vadd.f dm0, dm0, dm1, r0``
     - V
     - | ``vadd.f`` vector add (floating point)
       | ``dm0`` destination acc register
       | ``dm0`` 1st source acc register
       | ``dm1`` 2nd source acc register
       | ``r0`` scalar operand
   * - ``ret lr``
     - X
     - | ``ret`` return from subroutine
       | ``lr`` link register
   * - ``mov p1, p4``
     - M
     - | ``mov`` move scalar
       | ``p1`` destination pointer register
       | ``p4`` source pointer register
   * - ``vst.conv.bf16.fp32 cml0, [p2, #0]``
     - S
     - | ``vst`` vector store
       | ``.conv.bf16.fp32`` convert from BF16 to FP32
       | ``cml0`` source acc register
       | ``[p2, #0]`` pointer register and byte offset

b) Register-Class Table
^^^^^^^^^^^^^^^^^^^^^^^

For the register table we refer to the hints given at the start of task 3.

Hints:

- Mnemonic prefix/suffix indicates the slot
- Register name prefix indicates class: 
  - ``p`` → pointer register; 
  - ``r`` → scalar register; 
  - ``x`` / ``y`` → vector register; 
  - ``dm`` / ``cm`` / ``bm`` → accumulator register

.. list-table:: Instructions
   :widths: 10 40 40
   :header-rows: 1

   * - Slot
     - Register classes (dst / src)
     - Example registers
   * - V
     - vector / vector
     - ``x``, ``y``
   * - A
     - accumulator / pointer
     - ``cml0``, ``dm0``, ``dm1``, ``bmhl2``, ``bmhh4``, ``p0``
   * - B
     - accumulator / pointer
     - ``cml0``, ``dm0``, ``dm1``, ``bmhl2``, ``bmhh4``, ``p0``
   * - S
     - pointer / accumulator
     - ``cml0``, ``dm0``, ``dm1``, ``bmhl2``, ``bmhh4``, ``p0``
   * - X
     - scalar / scalar
     - ``r0``, ``r6``
   * - M
     - pointer / pointer
     - ``p0``, ``p6``
   * - XM
     - scalar + pointer / scalar + pointer
     - ``r0``, ``p0``
  
Task 4: Infer Operation Latencies
---------------------------------

Based on the information given in ``build/vadd.s`` file, we can derive information regarding the latency of the ``mova`` and ``vadd.f`` instruction.

.. list-table:: Operation Latencies
   :widths: 10 10 40 10 10
   :header-rows: 1

   * - Instruction
     - Output register
     - First dependent instruction
     - Cycles apart
     - Latency
   * - ``mova``
     - ``r0``
     - ``vadd.f dm0, dm0, dm1, r0``
     - 1
     - 1
   * - ``vadd.f``
     - ``dm0``
     - ``vst.conv.bf16.fp32 cml0, [p2, #0]``
     - 6
     - 6

As ``dm0 = cml0, cmh0`` we said that ``vst`` is the next instruction depending on ``vadd.f``.

Task 5: BF16 Hand-Scheduled Vector-Add Assembly Kernel
------------------------------------------------------

a) Add Instructions
^^^^^^^^^^^^^^^^^^^

Based on the insights gained from the previous tasks, we are now implementing a hand-scheduled vector-add assembly kernel.

For that we follow a simple schema:

- load the data into registers,
- compute the element-wise vector add,
- store the values from the registers back into memory.

And to follow the conventions, we also have to add a number of NOP cycles.

.. code-block:: asm

    custom_vadd:
    // Computes C = A + B + B
    // Calling convention: p0 = ptr_in0, p1 = ptr_in1, p2 = ptr_out
        vlda.conv.fp32.bf16	 cml0, [p0, #0]
        vlda.conv.fp32.bf16	 cmh0, [p0, #64]
        vlda.conv.fp32.bf16	 cml1, [p1, #0]
        vlda.conv.fp32.bf16	 cmh1, [p1, #64]
        nop
        nop
        mova	r0, #60
        vadd.f	dm0, dm0, dm1, r0
        nop
        nop
        vadd.f	dm2, dm0, dm1, r0
        nop
        nop
        ret lr
        nop                                  // Delay Slot 5
        nop                                  // Delay Slot 4
        vst.conv.bf16.fp32	 cml2, [p2, #0]  // Delay Slot 3
        vst.conv.bf16.fp32	 cmh2, [p2, #64] // Delay Slot 2
        nop                                  // Delay Slot 1

Our first experiments employed the latency information from task 4, leading to 6 NOP cycles after each ``vadd`` instruction.
However, after making some experiments we found out that there is no need for 6 NOP cycles **between** the two ``vadd`` instructions.
We assume that the hardware does not need to fully write back the result, but rather only needs to make the result readable for another vector unit instruction.

By using this information we reduce the VLIW cycles from 22 to 19.
Based on the knowledge that we have in regards to the XDNA instructions, we assume that this is the fewest possible number of cycles. 

b) Assemble Kernel
^^^^^^^^^^^^^^^^^^

After the implementation we assemble our kernel by running:

.. code-block:: bash

    make obj_custom_vadd

Running this command generates a ``custom_vadd.o`` in the build ``directory``.

c) Result Verification
^^^^^^^^^^^^^^^^^^^^^^

As with the vector-add kernel from task 1, the last step was to verify the correctness of our kernel. 
For that we enhanced the :ref:`condition <vadd_verify>` in the ``driver.py``.

.. code-block:: python

    ref = 0

    if ...
    elif kernel == "custom_vadd":
        ref = in0 + in1 + in1
    else:
        raise NotImplementedError("verify() not yet implemented")

Task 6: MAC Kernel
------------------

The optional task let us compile two given matmul kernels, where one simply adds the ``-DAIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16`` flag to the compilation path.

a) Instruction Count
^^^^^^^^^^^^^^^^^^^^

Counting VLIW cycles in the function bodies of ``build/matmul_normal.s`` and ``build/matmul_bfp16.s``:

.. list-table::
   :widths: 40 30 30
   :header-rows: 1

   * - Mode
     - Total VLIW cycles
     - Total non-NOP slot operations
   * - Normal (``matmul_normal.s``)
     - 43
     - 62
   * - BFP16 (``matmul_bfp16.s``)
     - 30
     - 22

b) What the Flag Changes
^^^^^^^^^^^^^^^^^^^^^^^^

Without the flag, the compiler has no native 8x8x8 BF16 matrix-multiply instruction and emulates it by manually decomposing the outer product:

- ``vextbcstshfl.64`` shuffles and extracts sub-rows of matrix A into vector registers
- ``vextbcst.128`` broadcasts columns of matrix B
- 16 times ``vmac.f`` which executes one multiply-accumulate per outer-product contribution
- Two output accumulators (``dm0``, ``dm1``) to process two output rows in parallel

With ``-DAIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16``, the compiler activates a hardware path based on BFP16, replacing the entire shuffle + 16-mac sequence with:

- ``vmul.f`` (**one instruction!**) performs the full 8x8x8 multiply using BFP16 format
- ``vconv.bfp16ebs8.fp32`` converts the BFP16 result back to FP32 accumulators
- ``vmac.f`` which is the single final accumulate into the output

Additionally, the BFP16 path reuses only ``cml0``/``cmh0`` (= ``dm0``) for the output, whereas the normal path uses ``cml0``/``cml1`` (= ``dm0`` and ``dm1``) to tile two rows simultaneously.

c) Performance Implications
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The reduction from 43 to 30 VLIW cycles gives a theoretical speedup of approximately **1.4×** for a single tile invocation.
The non-NOP operation count (62 -> 22, ~2.8x) provides an upper bound on speedup if all NOPs could be hidden through software pipelining across iterations.

The trade-off is numerical precision, because BFP16 stores a block of values with a shared exponent, which is slightly less precise than standard BF16.
