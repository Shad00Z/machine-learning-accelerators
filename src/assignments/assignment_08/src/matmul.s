  .file "matmul.s"
  .section .text.matmul,"ax",@progbits
  .globl matmul
  .p2align 4
  .type matmul,@function
matmul:
// Computes out += in0 * in1
// L1 tensor views:
//   p=2, q=2, r=8, m=8, n=8, k=8
//   in0: prmk  (BF16, p0)   in1: rqkn  (BF16, p1)   out: pqmn  (BF16, p2, zero-init)
//
//   dm0..dm3 : output accumulators (FP32); dm0=(p0,q0), dm1=(p0,q1), dm2=(p1,q0), dm3=(p1,q1)
//   dm4      : FP32 scratch for conversions
//   ex0/ex1  : BFP16 in0 tiles p=0/p=1
//   ex2/ex3  : BFP16 in1 tiles q=0/q=1
//   y0       : BF16 ones constant (64 elements, full 1024-bit Y-register)
//   x2,x3    : halves of y1 — in1 q=0 BF16 scratch (loaded as pair)
//   x4,x5    : halves of y2 — in1 q=1 BF16 scratch (loaded as pair)
//
//   in0 tile (p,r): p*1024 + r*128  (halves: +0, +64)
//   in1 tile (r,q): r*256  + q*128  (halves: +0, +64)
//   out tile (p,q): p*256  + q*128  (halves: +0, +64)
//
//   vlda/vldb/vlda.conv.fp32.bf16 : 7 cycles
//   vshuffle                       : 2 cycles
//   vmul.f                         : 6 cycles
//   vconv.bfp16ebs8.fp32           : 4 cycles
//   vmac.f                         : 6 cycles  (DMm read at cycle 4)
//   vbcst.16 / mova / movxm        : 1 cycle
//   vst.conv.bf16.fp32             : 2 cycles
//   ret lr                         : 6 cycles  (5 delay slots)
//
// VLIW instruction format: V-slot ; A-slot ; B-slot ; S-slot ; XM-slot ; M-slot

  // setup
  movxm r0, #16256                             // r0 = 0x3F80 = bf16(1.0)
  movxm r4, #780                               // r4 = vmac.f config (bfp16ebs8 8x8x8)
  mova  r5, #60                                // r5 = vmul.f config (bf16, 64 elements)
  mova  r6, #53                                // r6 = vshuffle mode 53 (kn->nk low)
  vbcst.16 y0, r0 ; mova r7, #52              // y0=[1.0..]; r7=vshuffle mode 52 (kn->nk high)

// r=0: A0=0, A1=1024, B_q0=0, B_q1=128
  vlda.conv.fp32.bf16 cml4, [p0, #0]    ; vldb x2, [p1, #0]   // load in0 (p=0,r=0) BF16->FP32 into cml4/cmh4 (lower/upper 512 bits of dm4)
  vlda.conv.fp32.bf16 cmh4, [p0, #64]   ; vldb x3, [p1, #64]  // load in1 (r=0,q=0) lower/upper 512-bit halves into x2/x3
  vldb x4, [p1, #128] // load in1 (r=0,q=1) lower/upper 512-bit halves into x4/x5
  vldb x5, [p1, #192]
  nop
  nop
  nop
  nop‚
  vlda.conv.fp32.bf16 cml4, [p0, #1024] ; vshuffle x2, x2, x3, r6   // already schedule load of new in0 (p=1,r=0) for next iteration; shuffle x2+x3 as the low half of the nk-transposed q=0 tile 
  vlda.conv.fp32.bf16 cmh4, [p0, #1088] ; vconv.bfp16ebs8.fp32 ex0, dm4   // schedule load of new in0 (p=1,r=0) for next iteration, while converting current in0 to bfp16
  vshuffle x3, x2, x3, r7   // shuffle x2+x3 as the high half of the nk-transposed q=0 tile
  vshuffle x4, x4, x5, r6   // shuffle x4+x5 as the low half of the nk-transposed q=1 tile
  nop
  vmul.f dm4, y0, y1, r5 ; vshuffle x5, x4, x5, r7 // widen y1=x2+x3 to FP32 and shuffle x4+x5 as the high half of the nk-transposed q=1 tile
  nop
  vmul.f dm4, y0, y2, r5 // widen y2=x4+x5 to FP32
  vconv.bfp16ebs8.fp32 ex1, dm4 // convert in0 (p=1) to bfp16 and store in ex1
  nop
  nop
  vconv.bfp16ebs8.fp32 ex2, dm4 // convert widened in1 q=0 tile to bfp16 and store in ex2
  nop
  vconv.bfp16ebs8.fp32 ex3, dm4 // convert widened in1 q=1 tile to bfp16 and store in ex3
  nop
  nop
  nop
  vmac.f dm0, dm0, ex0, ex2, r4 // dm0 += ex0 (in0,p=0) * ex2 (in1,q=0) -> out(p=0,q=0)
  vmac.f dm1, dm1, ex0, ex3, r4 // dm1 += ex0           * ex3 (in1,q=1) -> out(p=0,q=1)
  vmac.f dm2, dm2, ex1, ex2, r4 // dm2 += ex1 (in0,p=1) * ex2           -> out(p=1,q=0)
  vmac.f dm3, dm3, ex1, ex3, r4 // dm3 += ex1           * ex3           -> out(p=1,q=1)

  // r=1: A0=128, A1=1152, B_q0=256, B_q1=384
  vlda.conv.fp32.bf16 cml4, [p0, #128]  ; vldb x2, [p1, #256]
  vlda.conv.fp32.bf16 cmh4, [p0, #192]  ; vldb x3, [p1, #320]
  vldb x4, [p1, #384]
  vldb x5, [p1, #448]
  nop
  nop
  nop
  nop
  vlda.conv.fp32.bf16 cml4, [p0, #1152] ; vshuffle x2, x2, x3, r6
  vlda.conv.fp32.bf16 cmh4, [p0, #1216] ; vconv.bfp16ebs8.fp32 ex0, dm4
  vshuffle x3, x2, x3, r7
  vshuffle x4, x4, x5, r6
  nop
  vmul.f dm4, y0, y1, r5 ; vshuffle x5, x4, x5, r7
  nop
  vmul.f dm4, y0, y2, r5
  vconv.bfp16ebs8.fp32 ex1, dm4
  nop
  nop
  vconv.bfp16ebs8.fp32 ex2, dm4
  nop
  vconv.bfp16ebs8.fp32 ex3, dm4
  nop
  nop
  nop
  vmac.f dm0, dm0, ex0, ex2, r4
  vmac.f dm1, dm1, ex0, ex3, r4
  vmac.f dm2, dm2, ex1, ex2, r4
  vmac.f dm3, dm3, ex1, ex3, r4

  // r=2: A0=256, A1=1280, B_q0=512, B_q1=640
  vlda.conv.fp32.bf16 cml4, [p0, #256]  ; vldb x2, [p1, #512]
  vlda.conv.fp32.bf16 cmh4, [p0, #320]  ; vldb x3, [p1, #576]
  vldb x4, [p1, #640]
  vldb x5, [p1, #704]
  nop
  nop
  nop
  nop
  vlda.conv.fp32.bf16 cml4, [p0, #1280] ; vshuffle x2, x2, x3, r6
  vlda.conv.fp32.bf16 cmh4, [p0, #1344] ; vconv.bfp16ebs8.fp32 ex0, dm4
  vshuffle x3, x2, x3, r7
  vshuffle x4, x4, x5, r6
  nop
  vmul.f dm4, y0, y1, r5 ; vshuffle x5, x4, x5, r7
  nop
  vmul.f dm4, y0, y2, r5
  vconv.bfp16ebs8.fp32 ex1, dm4
  nop
  nop
  vconv.bfp16ebs8.fp32 ex2, dm4
  nop
  vconv.bfp16ebs8.fp32 ex3, dm4
  nop
  nop
  nop
  vmac.f dm0, dm0, ex0, ex2, r4
  vmac.f dm1, dm1, ex0, ex3, r4
  vmac.f dm2, dm2, ex1, ex2, r4
  vmac.f dm3, dm3, ex1, ex3, r4

  // r=3: A0=384, A1=1408, B_q0=768, B_q1=896
  vlda.conv.fp32.bf16 cml4, [p0, #384]  ; vldb x2, [p1, #768]
  vlda.conv.fp32.bf16 cmh4, [p0, #448]  ; vldb x3, [p1, #832]
  vldb x4, [p1, #896]
  vldb x5, [p1, #960]
  nop
  nop
  nop
  nop
  vlda.conv.fp32.bf16 cml4, [p0, #1408] ; vshuffle x2, x2, x3, r6
  vlda.conv.fp32.bf16 cmh4, [p0, #1472] ; vconv.bfp16ebs8.fp32 ex0, dm4
  vshuffle x3, x2, x3, r7
  vshuffle x4, x4, x5, r6
  nop
  vmul.f dm4, y0, y1, r5 ; vshuffle x5, x4, x5, r7
  nop
  vmul.f dm4, y0, y2, r5
  vconv.bfp16ebs8.fp32 ex1, dm4
  nop
  nop
  vconv.bfp16ebs8.fp32 ex2, dm4
  nop
  vconv.bfp16ebs8.fp32 ex3, dm4
  nop
  nop
  nop
  vmac.f dm0, dm0, ex0, ex2, r4
  vmac.f dm1, dm1, ex0, ex3, r4
  vmac.f dm2, dm2, ex1, ex2, r4
  vmac.f dm3, dm3, ex1, ex3, r4

  // r=4: A0=512, A1=1536, B_q0=1024, B_q1=1152
  vlda.conv.fp32.bf16 cml4, [p0, #512]  ; vldb x2, [p1, #1024]
  vlda.conv.fp32.bf16 cmh4, [p0, #576]  ; vldb x3, [p1, #1088]
  vldb x4, [p1, #1152]
  vldb x5, [p1, #1216]
  nop
  nop
  nop
  nop
  vlda.conv.fp32.bf16 cml4, [p0, #1536] ; vshuffle x2, x2, x3, r6
  vlda.conv.fp32.bf16 cmh4, [p0, #1600] ; vconv.bfp16ebs8.fp32 ex0, dm4
  vshuffle x3, x2, x3, r7
  vshuffle x4, x4, x5, r6
  nop
  vmul.f dm4, y0, y1, r5 ; vshuffle x5, x4, x5, r7
  nop
  vmul.f dm4, y0, y2, r5
  vconv.bfp16ebs8.fp32 ex1, dm4
  nop
  nop
  vconv.bfp16ebs8.fp32 ex2, dm4
  nop
  vconv.bfp16ebs8.fp32 ex3, dm4
  nop
  nop
  nop
  vmac.f dm0, dm0, ex0, ex2, r4
  vmac.f dm1, dm1, ex0, ex3, r4
  vmac.f dm2, dm2, ex1, ex2, r4
  vmac.f dm3, dm3, ex1, ex3, r4

  // r=5: A0=640, A1=1664, B_q0=1280, B_q1=1408
  vlda.conv.fp32.bf16 cml4, [p0, #640]  ; vldb x2, [p1, #1280]
  vlda.conv.fp32.bf16 cmh4, [p0, #704]  ; vldb x3, [p1, #1344]
  vldb x4, [p1, #1408]
  vldb x5, [p1, #1472]
  nop
  nop
  nop
  nop
  vlda.conv.fp32.bf16 cml4, [p0, #1664] ; vshuffle x2, x2, x3, r6
  vlda.conv.fp32.bf16 cmh4, [p0, #1728] ; vconv.bfp16ebs8.fp32 ex0, dm4
  vshuffle x3, x2, x3, r7
  vshuffle x4, x4, x5, r6
  nop
  vmul.f dm4, y0, y1, r5 ; vshuffle x5, x4, x5, r7
  nop
  vmul.f dm4, y0, y2, r5
  vconv.bfp16ebs8.fp32 ex1, dm4
  nop
  nop
  vconv.bfp16ebs8.fp32 ex2, dm4
  nop
  vconv.bfp16ebs8.fp32 ex3, dm4
  nop
  nop
  nop
  vmac.f dm0, dm0, ex0, ex2, r4
  vmac.f dm1, dm1, ex0, ex3, r4
  vmac.f dm2, dm2, ex1, ex2, r4
  vmac.f dm3, dm3, ex1, ex3, r4

  // r=6: A0=768, A1=1792, B_q0=1536, B_q1=1664
  vlda.conv.fp32.bf16 cml4, [p0, #768]  ; vldb x2, [p1, #1536]
  vlda.conv.fp32.bf16 cmh4, [p0, #832]  ; vldb x3, [p1, #1600]
  vldb x4, [p1, #1664]
  vldb x5, [p1, #1728]
  nop
  nop
  nop
  nop
  vlda.conv.fp32.bf16 cml4, [p0, #1792] ; vshuffle x2, x2, x3, r6
  vlda.conv.fp32.bf16 cmh4, [p0, #1856] ; vconv.bfp16ebs8.fp32 ex0, dm4
  vshuffle x3, x2, x3, r7
  vshuffle x4, x4, x5, r6
  nop
  vmul.f dm4, y0, y1, r5 ; vshuffle x5, x4, x5, r7
  nop
  vmul.f dm4, y0, y2, r5
  vconv.bfp16ebs8.fp32 ex1, dm4
  nop
  nop
  vconv.bfp16ebs8.fp32 ex2, dm4
  nop
  vconv.bfp16ebs8.fp32 ex3, dm4
  nop
  nop
  nop
  vmac.f dm0, dm0, ex0, ex2, r4
  vmac.f dm1, dm1, ex0, ex3, r4
  vmac.f dm2, dm2, ex1, ex2, r4
  vmac.f dm3, dm3, ex1, ex3, r4

  // r=7: A0=896, A1=1920, B_q0=1792, B_q1=1920
  vlda.conv.fp32.bf16 cml4, [p0, #896]  ; vldb x2, [p1, #1792]
  vlda.conv.fp32.bf16 cmh4, [p0, #960]  ; vldb x3, [p1, #1856]
  vldb x4, [p1, #1920]
  vldb x5, [p1, #1984]
  nop
  nop
  nop
  nop
  vlda.conv.fp32.bf16 cml4, [p0, #1920] ; vshuffle x2, x2, x3, r6
  vlda.conv.fp32.bf16 cmh4, [p0, #1984] ; vconv.bfp16ebs8.fp32 ex0, dm4
  vshuffle x3, x2, x3, r7
  vshuffle x4, x4, x5, r6
  nop
  vmul.f dm4, y0, y1, r5 ; vshuffle x5, x4, x5, r7
  nop
  vmul.f dm4, y0, y2, r5
  vconv.bfp16ebs8.fp32 ex1, dm4
  nop
  nop
  vconv.bfp16ebs8.fp32 ex2, dm4
  nop
  vconv.bfp16ebs8.fp32 ex3, dm4
  nop
  nop
  nop
  vmac.f dm0, dm0, ex0, ex2, r4
  vmac.f dm1, dm1, ex0, ex3, r4
  vmac.f dm2, dm2, ex1, ex2, r4
  vmac.f dm3, dm3, ex1, ex3, r4
  nop
  nop
  nop
  nop
  nop
  nop
  // dm0 -> out (p=0,q=0), base offset 0
  vst.conv.bf16.fp32 cml0, [p2, #0]
  vst.conv.bf16.fp32 cmh0, [p2, #64]
  // dm1 -> out (p=0,q=1), base offset 128
  vst.conv.bf16.fp32 cml1, [p2, #128]
  vst.conv.bf16.fp32 cmh1, [p2, #192]
  // dm2 -> out (p=1,q=0), base offset 256
  vst.conv.bf16.fp32 cml2, [p2, #256]
  vst.conv.bf16.fp32 cmh2, [p2, #320]
  // dm3 -> out (p=1,q=1), base offset 384
  vst.conv.bf16.fp32 cml3, [p2, #384]
  vst.conv.bf16.fp32 cmh3, [p2, #448]

  ret lr
  nop  // Delay Slot 5
  nop  // Delay Slot 4
  nop  // Delay Slot 3
  nop  // Delay Slot 2
  nop  // Delay Slot 1
.Lfunc_end0:
  .size matmul, .Lfunc_end0-matmul
