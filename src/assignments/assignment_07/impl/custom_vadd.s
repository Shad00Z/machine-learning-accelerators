  .file "custom_vadd.s"
  .section .text.custom_vadd,"ax",@progbits
  .globl custom_vadd
  .p2align 4
  .type custom_vadd,@function
custom_vadd:
// Computes C = A + B + B
// Calling convention: p0 = ptr_in0, p1 = ptr_in1, p2 = ptr_out
// Load A: 64 elements as 2x 32-element BF16 vectors
  vlda.conv.fp32.bf16	 cml0, [p0, #0]   // lower 32 elements of A
  vlda.conv.fp32.bf16	 cmh0, [p0, #64]  // upper 32 elements of A -> together as dm0
// Load B: 64 elements as 2x 32-element BF16 vectors
  vlda.conv.fp32.bf16	 cml1, [p1, #0]   // lower 32 elements of B
  vlda.conv.fp32.bf16	 cmh1, [p1, #64]  // upper 32 elements of B -> together as dm1
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
.Lfunc_end0:
  .size custom_vadd, .Lfunc_end0-custom_vadd