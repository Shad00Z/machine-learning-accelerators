.file "matmul.s"
  .section .text.matmul,"ax",@progbits
  .globl	matmul_init
  .globl matmul
  .p2align 4
  .type	matmul_init,@function
  .type matmul,@function
matmul_init:
	vclr	dm1
	vclr	dm2
	vclr	dm3
	vclr	dm4
matmul:
// Computes out += in0 * in1
// L1 tensor views:
//   p=2, q=2, r=8, m=8, n=8, k=8
//   in0: prmk
//   in1: rqkn
//   out: pqmn

  // setup
  movxm r0, #16256      // r0 = 0x3F80 = bf16(1.0)
  mova  r1, #780      // r1 = vmac.f config (bfp16ebs8 8x8x8)
  mova  r2, #52         // vshuffle mode 52
  mova  r3, #53         // vshuffle mode 53
  mova  r4, #60         // vmul.f mode
  vbcst.16 x8, r0       // y4 (x8, x9) = 64x bf16(1.0)
  vmov x9, x8
  mov p3, p0            // p3 = in0 p=1 base (= p0 + 1024)
  padda [p3], #256    // max 511 immediate offset, so 4 times 256
  padda [p3], #256
  padda [p3], #256
  padda [p3], #256
  mov p4, p2

// Load output tensor into dm1..dm4
  vlda.conv.fp32.bf16 cml1, [p4], #64
  vlda.conv.fp32.bf16 cmh1, [p4], #64
  vlda.conv.fp32.bf16 cml2, [p4], #64
  vlda.conv.fp32.bf16 cmh2, [p4], #64
  vlda.conv.fp32.bf16 cml3, [p4], #64
  vlda.conv.fp32.bf16 cmh3, [p4], #64
  vlda.conv.fp32.bf16 cml4, [p4], #64
  vlda.conv.fp32.bf16 cmh4, [p4], #64

// r=0
  // Step 1: load in1 into x0..x3
  vldb x0, [p1, #0]
  vldb x1, [p1, #64]
  vldb x2, [p1, #128]
  vldb x3, [p1, #192]
  nop
  nop
  nop
  nop
  nop
  // Step 2: transpose in1 kn->nk via vshuffle
  vshuffle x4, x0, x1, r2   // mode 52: q=0, low half
  vshuffle x5, x0, x1, r3   // mode 53: q=0, high half -> y2=x4+x5 is full q=0 nk tile
  vshuffle x6, x2, x3, r2   // mode 52: q=1, low half
  vshuffle x7, x2, x3, r3   // mode 53: q=1, high half -> y3=x6+x7 is full q=1 nk tile
  nop
  
  // Step 3: vmul in1 tiles by ones to get FP32, then convert to BFP16
  vmul.f dm0, y2, y4, r4    // dm0 = in1(q=0) FP32
  nop
  nop
  nop
  nop
  nop
  nop
  vconv.bfp16ebs8.fp32 ex0, dm0   // ex0 = in1(q=0) BFP16
  nop
  nop
  nop
  nop
  vmul.f dm0, y3, y4, r4    // dm0 = in1(q=1) FP32
  nop
  nop
  nop
  nop
  nop
  nop
  vconv.bfp16ebs8.fp32 ex1, dm0   // ex1 = in1(q=1) BFP16
  nop
  nop
  nop
  nop

  // Step 4: load in0(p=0) into dm0, then convert
  vlda.conv.fp32.bf16 cml0, [p0, #0]
  vlda.conv.fp32.bf16 cmh0, [p0, #64]
  nop
  nop
  nop
  nop
  nop
  nop
  vconv.bfp16ebs8.fp32 ex10, dm0   // ex10 = in0(p=0) BFP16
  nop
  nop
  nop
  nop

  // Step 5: load in0(p=1) into dm0, then convert
  vlda.conv.fp32.bf16 cml0, [p3, #0]
  vlda.conv.fp32.bf16 cmh0, [p3, #64]
  nop
  nop
  nop
  nop
  nop
  nop
  vconv.bfp16ebs8.fp32 ex11, dm0   // ex11 = in0(p=1) BFP16
  nop
  nop
  nop
  nop

  // Step 7: vmac — all four output tiles
  vmac.f dm1, dm1, ex10, ex0, r1   // out(p=0,q=0) += in0(p=0) * in1(q=0)
  vmac.f dm2, dm2, ex10, ex1, r1   // out(p=0,q=1) += in0(p=0) * in1(q=1)
  vmac.f dm3, dm3, ex11, ex0, r1   // out(p=1,q=0) += in0(p=1) * in1(q=0)
  vmac.f dm4, dm4, ex11, ex1, r1   // out(p=1,q=1) += in0(p=1) * in1(q=1)
  padda [p0], #128               // advance in0 p=0 pointer to r+1
  padda [p3], #128               // advance in0 p=1 pointer to r+1
  paddb [p1], #256               // advance in1 pointer to r+1
  nop
  nop
  nop
  nop
  nop
  nop
// r=1
  // Step 1: load in1 into x0..x3
  vldb x0, [p1, #0]
  vldb x1, [p1, #64]
  vldb x2, [p1, #128]
  vldb x3, [p1, #192]
  nop
  nop
  nop
  nop
  nop
  // Step 2: transpose in1 kn->nk via vshuffle
  vshuffle x4, x0, x1, r2   // mode 52: q=0, low half
  vshuffle x5, x0, x1, r3   // mode 53: q=0, high half -> y2=x4+x5 is full q=0 nk tile
  vshuffle x6, x2, x3, r2   // mode 52: q=1, low half
  vshuffle x7, x2, x3, r3   // mode 53: q=1, high half -> y3=x6+x7 is full q=1 nk tile
  nop
  
  // Step 3: vmul in1 tiles by ones to get FP32, then convert to BFP16
  vmul.f dm0, y2, y4, r4    // dm0 = in1(q=0) FP32
  nop
  nop
  nop
  nop
  nop
  nop
  vconv.bfp16ebs8.fp32 ex0, dm0   // ex0 = in1(q=0) BFP16
  nop
  nop
  nop
  nop
  vmul.f dm0, y3, y4, r4    // dm0 = in1(q=1) FP32
  nop
  nop
  nop
  nop
  nop
  nop
  vconv.bfp16ebs8.fp32 ex1, dm0   // ex1 = in1(q=1) BFP16
  nop
  nop
  nop
  nop

  // Step 4: load in0(p=0) into dm0, then convert
  vlda.conv.fp32.bf16 cml0, [p0, #0]
  vlda.conv.fp32.bf16 cmh0, [p0, #64]
  nop
  nop
  nop
  nop
  nop
  nop
  vconv.bfp16ebs8.fp32 ex10, dm0   // ex10 = in0(p=0) BFP16
  nop
  nop
  nop
  nop

  // Step 5: load in0(p=1) into dm0, then convert
  vlda.conv.fp32.bf16 cml0, [p3, #0]
  vlda.conv.fp32.bf16 cmh0, [p3, #64]
  nop
  nop
  nop
  nop
  nop
  nop
  vconv.bfp16ebs8.fp32 ex11, dm0   // ex11 = in0(p=1) BFP16
  nop
  nop
  nop
  nop

  // Step 7: vmac — all four output tiles
  vmac.f dm1, dm1, ex10, ex0, r1   // out(p=0,q=0) += in0(p=0) * in1(q=0)
  vmac.f dm2, dm2, ex10, ex1, r1   // out(p=0,q=1) += in0(p=0) * in1(q=1)
  vmac.f dm3, dm3, ex11, ex0, r1   // out(p=1,q=0) += in0(p=1) * in1(q=0)
  vmac.f dm4, dm4, ex11, ex1, r1   // out(p=1,q=1) += in0(p=1) * in1(q=1)
  padda [p0], #128               // advance in0 p=0 pointer to r+1
  padda [p3], #128               // advance in0 p=1 pointer to r+1
  paddb [p1], #256               // advance in1 pointer to r+1
  nop
  nop
  nop
  nop
  nop
  nop
// r=2
  // Step 1: load in1 into x0..x3
  vldb x0, [p1, #0]
  vldb x1, [p1, #64]
  vldb x2, [p1, #128]
  vldb x3, [p1, #192]
  nop
  nop
  nop
  nop
  nop
  // Step 2: transpose in1 kn->nk via vshuffle
  vshuffle x4, x0, x1, r2   // mode 52: q=0, low half
  vshuffle x5, x0, x1, r3   // mode 53: q=0, high half -> y2=x4+x5 is full q=0 nk tile
  vshuffle x6, x2, x3, r2   // mode 52: q=1, low half
  vshuffle x7, x2, x3, r3   // mode 53: q=1, high half -> y3=x6+x7 is full q=1 nk tile
  nop
  
  // Step 3: vmul in1 tiles by ones to get FP32, then convert to BFP16
  vmul.f dm0, y2, y4, r4    // dm0 = in1(q=0) FP32
  nop
  nop
  nop
  nop
  nop
  nop
  vconv.bfp16ebs8.fp32 ex0, dm0   // ex0 = in1(q=0) BFP16
  nop
  nop
  nop
  nop
  vmul.f dm0, y3, y4, r4    // dm0 = in1(q=1) FP32
  nop
  nop
  nop
  nop
  nop
  nop
  vconv.bfp16ebs8.fp32 ex1, dm0   // ex1 = in1(q=1) BFP16
  nop
  nop
  nop
  nop

  // Step 4: load in0(p=0) into dm0, then convert
  vlda.conv.fp32.bf16 cml0, [p0, #0]
  vlda.conv.fp32.bf16 cmh0, [p0, #64]
  nop
  nop
  nop
  nop
  nop
  nop
  vconv.bfp16ebs8.fp32 ex10, dm0   // ex10 = in0(p=0) BFP16
  nop
  nop
  nop
  nop

  // Step 5: load in0(p=1) into dm0, then convert
  vlda.conv.fp32.bf16 cml0, [p3, #0]
  vlda.conv.fp32.bf16 cmh0, [p3, #64]
  nop
  nop
  nop
  nop
  nop
  nop
  vconv.bfp16ebs8.fp32 ex11, dm0   // ex11 = in0(p=1) BFP16
  nop
  nop
  nop
  nop

  // Step 7: vmac — all four output tiles
  vmac.f dm1, dm1, ex10, ex0, r1   // out(p=0,q=0) += in0(p=0) * in1(q=0)
  vmac.f dm2, dm2, ex10, ex1, r1   // out(p=0,q=1) += in0(p=0) * in1(q=1)
  vmac.f dm3, dm3, ex11, ex0, r1   // out(p=1,q=0) += in0(p=1) * in1(q=0)
  vmac.f dm4, dm4, ex11, ex1, r1   // out(p=1,q=1) += in0(p=1) * in1(q=1)
  padda [p0], #128               // advance in0 p=0 pointer to r+1
  padda [p3], #128               // advance in0 p=1 pointer to r+1
  paddb [p1], #256               // advance in1 pointer to r+1
  nop
  nop
  nop
  nop
  nop
  nop
// r=3
  // Step 1: load in1 into x0..x3
  vldb x0, [p1, #0]
  vldb x1, [p1, #64]
  vldb x2, [p1, #128]
  vldb x3, [p1, #192]
  nop
  nop
  nop
  nop
  nop
  // Step 2: transpose in1 kn->nk via vshuffle
  vshuffle x4, x0, x1, r2   // mode 52: q=0, low half
  vshuffle x5, x0, x1, r3   // mode 53: q=0, high half -> y2=x4+x5 is full q=0 nk tile
  vshuffle x6, x2, x3, r2   // mode 52: q=1, low half
  vshuffle x7, x2, x3, r3   // mode 53: q=1, high half -> y3=x6+x7 is full q=1 nk tile
  nop
  
  // Step 3: vmul in1 tiles by ones to get FP32, then convert to BFP16
  vmul.f dm0, y2, y4, r4    // dm0 = in1(q=0) FP32
  nop
  nop
  nop
  nop
  nop
  nop
  vconv.bfp16ebs8.fp32 ex0, dm0   // ex0 = in1(q=0) BFP16
  nop
  nop
  nop
  nop
  vmul.f dm0, y3, y4, r4    // dm0 = in1(q=1) FP32
  nop
  nop
  nop
  nop
  nop
  nop
  vconv.bfp16ebs8.fp32 ex1, dm0   // ex1 = in1(q=1) BFP16
  nop
  nop
  nop
  nop

  // Step 4: load in0(p=0) into dm0, then convert
  vlda.conv.fp32.bf16 cml0, [p0, #0]
  vlda.conv.fp32.bf16 cmh0, [p0, #64]
  nop
  nop
  nop
  nop
  nop
  nop
  vconv.bfp16ebs8.fp32 ex10, dm0   // ex10 = in0(p=0) BFP16
  nop
  nop
  nop
  nop

  // Step 5: load in0(p=1) into dm0, then convert
  vlda.conv.fp32.bf16 cml0, [p3, #0]
  vlda.conv.fp32.bf16 cmh0, [p3, #64]
  nop
  nop
  nop
  nop
  nop
  nop
  vconv.bfp16ebs8.fp32 ex11, dm0   // ex11 = in0(p=1) BFP16
  nop
  nop
  nop
  nop

  // Step 7: vmac — all four output tiles
  vmac.f dm1, dm1, ex10, ex0, r1   // out(p=0,q=0) += in0(p=0) * in1(q=0)
  vmac.f dm2, dm2, ex10, ex1, r1   // out(p=0,q=1) += in0(p=0) * in1(q=1)
  vmac.f dm3, dm3, ex11, ex0, r1   // out(p=1,q=0) += in0(p=1) * in1(q=0)
  vmac.f dm4, dm4, ex11, ex1, r1   // out(p=1,q=1) += in0(p=1) * in1(q=1)
  padda [p0], #128               // advance in0 p=0 pointer to r+1
  padda [p3], #128               // advance in0 p=1 pointer to r+1
  paddb [p1], #256               // advance in1 pointer to r+1
  nop
  nop
  nop
  nop
  nop
  nop
// r=4
  // Step 1: load in1 into x0..x3
  vldb x0, [p1, #0]
  vldb x1, [p1, #64]
  vldb x2, [p1, #128]
  vldb x3, [p1, #192]
  nop
  nop
  nop
  nop
  nop
  // Step 2: transpose in1 kn->nk via vshuffle
  vshuffle x4, x0, x1, r2   // mode 52: q=0, low half
  vshuffle x5, x0, x1, r3   // mode 53: q=0, high half -> y2=x4+x5 is full q=0 nk tile
  vshuffle x6, x2, x3, r2   // mode 52: q=1, low half
  vshuffle x7, x2, x3, r3   // mode 53: q=1, high half -> y3=x6+x7 is full q=1 nk tile
  nop
  
  // Step 3: vmul in1 tiles by ones to get FP32, then convert to BFP16
  vmul.f dm0, y2, y4, r4    // dm0 = in1(q=0) FP32
  nop
  nop
  nop
  nop
  nop
  nop
  vconv.bfp16ebs8.fp32 ex0, dm0   // ex0 = in1(q=0) BFP16
  nop
  nop
  nop
  nop
  vmul.f dm0, y3, y4, r4    // dm0 = in1(q=1) FP32
  nop
  nop
  nop
  nop
  nop
  nop
  vconv.bfp16ebs8.fp32 ex1, dm0   // ex1 = in1(q=1) BFP16
  nop
  nop
  nop
  nop

  // Step 4: load in0(p=0) into dm0, then convert
  vlda.conv.fp32.bf16 cml0, [p0, #0]
  vlda.conv.fp32.bf16 cmh0, [p0, #64]
  nop
  nop
  nop
  nop
  nop
  nop
  vconv.bfp16ebs8.fp32 ex10, dm0   // ex10 = in0(p=0) BFP16
  nop
  nop
  nop
  nop

  // Step 5: load in0(p=1) into dm0, then convert
  vlda.conv.fp32.bf16 cml0, [p3, #0]
  vlda.conv.fp32.bf16 cmh0, [p3, #64]
  nop
  nop
  nop
  nop
  nop
  nop
  vconv.bfp16ebs8.fp32 ex11, dm0   // ex11 = in0(p=1) BFP16
  nop
  nop
  nop
  nop

  // Step 7: vmac — all four output tiles
  vmac.f dm1, dm1, ex10, ex0, r1   // out(p=0,q=0) += in0(p=0) * in1(q=0)
  vmac.f dm2, dm2, ex10, ex1, r1   // out(p=0,q=1) += in0(p=0) * in1(q=1)
  vmac.f dm3, dm3, ex11, ex0, r1   // out(p=1,q=0) += in0(p=1) * in1(q=0)
  vmac.f dm4, dm4, ex11, ex1, r1   // out(p=1,q=1) += in0(p=1) * in1(q=1)
  padda [p0], #128               // advance in0 p=0 pointer to r+1
  padda [p3], #128               // advance in0 p=1 pointer to r+1
  paddb [p1], #256               // advance in1 pointer to r+1
  nop
  nop
  nop
  nop
  nop
  nop
// r=5
  // Step 1: load in1 into x0..x3
  vldb x0, [p1, #0]
  vldb x1, [p1, #64]
  vldb x2, [p1, #128]
  vldb x3, [p1, #192]
  nop
  nop
  nop
  nop
  nop
  // Step 2: transpose in1 kn->nk via vshuffle
  vshuffle x4, x0, x1, r2   // mode 52: q=0, low half
  vshuffle x5, x0, x1, r3   // mode 53: q=0, high half -> y2=x4+x5 is full q=0 nk tile
  vshuffle x6, x2, x3, r2   // mode 52: q=1, low half
  vshuffle x7, x2, x3, r3   // mode 53: q=1, high half -> y3=x6+x7 is full q=1 nk tile
  nop
  
  // Step 3: vmul in1 tiles by ones to get FP32, then convert to BFP16
  vmul.f dm0, y2, y4, r4    // dm0 = in1(q=0) FP32
  nop
  nop
  nop
  nop
  nop
  nop
  vconv.bfp16ebs8.fp32 ex0, dm0   // ex0 = in1(q=0) BFP16
  nop
  nop
  nop
  nop
  vmul.f dm0, y3, y4, r4    // dm0 = in1(q=1) FP32
  nop
  nop
  nop
  nop
  nop
  nop
  vconv.bfp16ebs8.fp32 ex1, dm0   // ex1 = in1(q=1) BFP16
  nop
  nop
  nop
  nop

  // Step 4: load in0(p=0) into dm0, then convert
  vlda.conv.fp32.bf16 cml0, [p0, #0]
  vlda.conv.fp32.bf16 cmh0, [p0, #64]
  nop
  nop
  nop
  nop
  nop
  nop
  vconv.bfp16ebs8.fp32 ex10, dm0   // ex10 = in0(p=0) BFP16
  nop
  nop
  nop
  nop

  // Step 5: load in0(p=1) into dm0, then convert
  vlda.conv.fp32.bf16 cml0, [p3, #0]
  vlda.conv.fp32.bf16 cmh0, [p3, #64]
  nop
  nop
  nop
  nop
  nop
  nop
  vconv.bfp16ebs8.fp32 ex11, dm0   // ex11 = in0(p=1) BFP16
  nop
  nop
  nop
  nop

  // Step 7: vmac — all four output tiles
  vmac.f dm1, dm1, ex10, ex0, r1   // out(p=0,q=0) += in0(p=0) * in1(q=0)
  vmac.f dm2, dm2, ex10, ex1, r1   // out(p=0,q=1) += in0(p=0) * in1(q=1)
  vmac.f dm3, dm3, ex11, ex0, r1   // out(p=1,q=0) += in0(p=1) * in1(q=0)
  vmac.f dm4, dm4, ex11, ex1, r1   // out(p=1,q=1) += in0(p=1) * in1(q=1)
  padda [p0], #128               // advance in0 p=0 pointer to r+1
  padda [p3], #128               // advance in0 p=1 pointer to r+1
  paddb [p1], #256               // advance in1 pointer to r+1
  nop
  nop
  nop
  nop
  nop
  nop
// r=6
  // Step 1: load in1 into x0..x3
  vldb x0, [p1, #0]
  vldb x1, [p1, #64]
  vldb x2, [p1, #128]
  vldb x3, [p1, #192]
  nop
  nop
  nop
  nop
  nop
  // Step 2: transpose in1 kn->nk via vshuffle
  vshuffle x4, x0, x1, r2   // mode 52: q=0, low half
  vshuffle x5, x0, x1, r3   // mode 53: q=0, high half -> y2=x4+x5 is full q=0 nk tile
  vshuffle x6, x2, x3, r2   // mode 52: q=1, low half
  vshuffle x7, x2, x3, r3   // mode 53: q=1, high half -> y3=x6+x7 is full q=1 nk tile
  nop
  
  // Step 3: vmul in1 tiles by ones to get FP32, then convert to BFP16
  vmul.f dm0, y2, y4, r4    // dm0 = in1(q=0) FP32
  nop
  nop
  nop
  nop
  nop
  nop
  vconv.bfp16ebs8.fp32 ex0, dm0   // ex0 = in1(q=0) BFP16
  nop
  nop
  nop
  nop
  vmul.f dm0, y3, y4, r4    // dm0 = in1(q=1) FP32
  nop
  nop
  nop
  nop
  nop
  nop
  vconv.bfp16ebs8.fp32 ex1, dm0   // ex1 = in1(q=1) BFP16
  nop
  nop
  nop
  nop

  // Step 4: load in0(p=0) into dm0, then convert
  vlda.conv.fp32.bf16 cml0, [p0, #0]
  vlda.conv.fp32.bf16 cmh0, [p0, #64]
  nop
  nop
  nop
  nop
  nop
  nop
  vconv.bfp16ebs8.fp32 ex10, dm0   // ex10 = in0(p=0) BFP16
  nop
  nop
  nop
  nop

  // Step 5: load in0(p=1) into dm0, then convert
  vlda.conv.fp32.bf16 cml0, [p3, #0]
  vlda.conv.fp32.bf16 cmh0, [p3, #64]
  nop
  nop
  nop
  nop
  nop
  nop
  vconv.bfp16ebs8.fp32 ex11, dm0   // ex11 = in0(p=1) BFP16
  nop
  nop
  nop
  nop

  // Step 7: vmac — all four output tiles
  vmac.f dm1, dm1, ex10, ex0, r1   // out(p=0,q=0) += in0(p=0) * in1(q=0)
  vmac.f dm2, dm2, ex10, ex1, r1   // out(p=0,q=1) += in0(p=0) * in1(q=1)
  vmac.f dm3, dm3, ex11, ex0, r1   // out(p=1,q=0) += in0(p=1) * in1(q=0)
  vmac.f dm4, dm4, ex11, ex1, r1   // out(p=1,q=1) += in0(p=1) * in1(q=1)
  padda [p0], #128               // advance in0 p=0 pointer to r+1
  padda [p3], #128               // advance in0 p=1 pointer to r+1
  paddb [p1], #256               // advance in1 pointer to r+1
  nop
  nop
  nop
  nop
  nop
  nop
// r=7
  // Step 1: load in1 into x0..x3
  vldb x0, [p1, #0]
  vldb x1, [p1, #64]
  vldb x2, [p1, #128]
  vldb x3, [p1, #192]
  nop
  nop
  nop
  nop
  nop
  // Step 2: transpose in1 kn->nk via vshuffle
  vshuffle x4, x0, x1, r2   // mode 52: q=0, low half
  vshuffle x5, x0, x1, r3   // mode 53: q=0, high half -> y2=x4+x5 is full q=0 nk tile
  vshuffle x6, x2, x3, r2   // mode 52: q=1, low half
  vshuffle x7, x2, x3, r3   // mode 53: q=1, high half -> y3=x6+x7 is full q=1 nk tile
  nop
  
  // Step 3: vmul in1 tiles by ones to get FP32, then convert to BFP16
  vmul.f dm0, y2, y4, r4    // dm0 = in1(q=0) FP32
  nop
  nop
  nop
  nop
  nop
  nop
  vconv.bfp16ebs8.fp32 ex0, dm0   // ex0 = in1(q=0) BFP16
  nop
  nop
  nop
  nop
  vmul.f dm0, y3, y4, r4    // dm0 = in1(q=1) FP32
  nop
  nop
  nop
  nop
  nop
  nop
  vconv.bfp16ebs8.fp32 ex1, dm0   // ex1 = in1(q=1) BFP16
  nop
  nop
  nop
  nop

  // Step 4: load in0(p=0) into dm0, then convert
  vlda.conv.fp32.bf16 cml0, [p0, #0]
  vlda.conv.fp32.bf16 cmh0, [p0, #64]
  nop
  nop
  nop
  nop
  nop
  nop
  vconv.bfp16ebs8.fp32 ex10, dm0   // ex10 = in0(p=0) BFP16
  nop
  nop
  nop
  nop

  // Step 5: load in0(p=1) into dm0, then convert
  vlda.conv.fp32.bf16 cml0, [p3, #0]
  vlda.conv.fp32.bf16 cmh0, [p3, #64]
  nop
  nop
  nop
  nop
  nop
  nop
  vconv.bfp16ebs8.fp32 ex11, dm0   // ex11 = in0(p=1) BFP16
  nop
  nop
  nop
  nop

  // Step 7: vmac — all four output tiles
  vmac.f dm1, dm1, ex10, ex0, r1   // out(p=0,q=0) += in0(p=0) * in1(q=0)
  vmac.f dm2, dm2, ex10, ex1, r1   // out(p=0,q=1) += in0(p=0) * in1(q=1)
  vmac.f dm3, dm3, ex11, ex0, r1   // out(p=1,q=0) += in0(p=1) * in1(q=0)
  vmac.f dm4, dm4, ex11, ex1, r1   // out(p=1,q=1) += in0(p=1) * in1(q=1)
// store results
  nop
  nop
  nop
  nop
  nop
  nop
  nop
  // dm1 -> out (p=0,q=0)
  vst.conv.bf16.fp32 cml1, [p2], #64
  vst.conv.bf16.fp32 cmh1, [p2], #64
  // dm2 -> out (p=0,q=1)
  vst.conv.bf16.fp32 cml2, [p2], #64
  vst.conv.bf16.fp32 cmh2, [p2], #64
  // dm3 -> out (p=1,q=0)
  vst.conv.bf16.fp32 cml3, [p2], #64
  vst.conv.bf16.fp32 cmh3, [p2], #64
  // dm4 -> out (p=1,q=1)
  vst.conv.bf16.fp32 cml4, [p2], #64
  vst.conv.bf16.fp32 cmh4, [p2], #64

  ret lr
  nop  // Delay Slot 5
  nop  // Delay Slot 4
  nop  // Delay Slot 3
  nop  // Delay Slot 2
  nop  // Delay Slot 1
.Lfunc_end0:
  .size matmul, .Lfunc_end0-matmul
  .size	matmul_init, .Lfunc_end0-matmul_init
