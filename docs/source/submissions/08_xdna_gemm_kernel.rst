XDNA GEMM Kernel
==========================================

Task 1: Verify Function
------------------------------------------

Task 2: Instructions and Latencies
------------------------------------------

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

Task 4: Data Layouts and Pointer Updates
------------------------------------------

Task 5: Implementation
------------------------------------------

Task 6: Performance
------------------------------------------