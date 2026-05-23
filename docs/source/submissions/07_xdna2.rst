Inferring the VLIW ISA of XDNA2
===============================

This assignment starts of the second half of the lectures. 
From now on, we are working with the VLIW instruction set architecture of the XDNA2 compute tile.

Task 1: Vector-Add Kernel
-------------------------

a) Element-wise Vector Addition
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To start of the implementation on the XDNA2 architecture, we are implementing a simple vector addition kernel.
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

.. code-block:: python 

    if kernel == "vadd":
        ref = in0 + in1
        
        if not torch.allclose(out, ref, rtol=1e-2, atol=1e-2):
            max_err = (out - ref).abs().max().item()
            raise ValueError(f"[FAIL] {kernel} verification passed.")
    else:
        raise NotImplementedError("verify() not yet implemented")

Task 2: Identify VLIW Slots
---------------------------

For this task we are looking deeper into the XDNA2 VLIW instruciton word and its functional slots. 

The key insight is that the mnemonic name of the instruction (``vlda``) tells us which functional unit belongs to it (is occupied).

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
