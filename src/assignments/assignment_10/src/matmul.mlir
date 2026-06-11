module {
  aie.device(npu2) {
    func.func private @matmul(memref<2x8x8x8xbf16>, memref<8x2x8x8xbf16>, memref<2x2x8x8xbf16>) attributes {link_with = "matmul.o"}
    func.func private @zero(memref<2x2x8x8xbf16>) attributes {link_with = "zero.o"}
    // Shim tiles (row 0)
    %shim_noc_tile_0_0 = aie.tile(0, 0)
    %shim_noc_tile_1_0 = aie.tile(1, 0)
    %shim_noc_tile_2_0 = aie.tile(2, 0)
    %shim_noc_tile_3_0 = aie.tile(3, 0)
    %shim_noc_tile_4_0 = aie.tile(4, 0)
    %shim_noc_tile_5_0 = aie.tile(5, 0)
    %shim_noc_tile_6_0 = aie.tile(6, 0)
    %shim_noc_tile_7_0 = aie.tile(7, 0)
    // Memory tiles (row 1)
    %mem_tile_0_1 = aie.tile(0, 1)
    %mem_tile_1_1 = aie.tile(1, 1)
    %mem_tile_2_1 = aie.tile(2, 1)
    %mem_tile_3_1 = aie.tile(3, 1)
    %mem_tile_4_1 = aie.tile(4, 1)
    %mem_tile_5_1 = aie.tile(5, 1)
    %mem_tile_6_1 = aie.tile(6, 1)
    %mem_tile_7_1 = aie.tile(7, 1)
    // Compute tiles (x=8 columns, y=4 rows → rows 2–5)
    %tile_0_2 = aie.tile(0, 2)
    %tile_0_3 = aie.tile(0, 3)
    %tile_0_4 = aie.tile(0, 4)
    %tile_0_5 = aie.tile(0, 5)
    %tile_1_2 = aie.tile(1, 2)
    %tile_1_3 = aie.tile(1, 3)
    %tile_1_4 = aie.tile(1, 4)
    %tile_1_5 = aie.tile(1, 5)
    %tile_2_2 = aie.tile(2, 2)
    %tile_2_3 = aie.tile(2, 3)
    %tile_2_4 = aie.tile(2, 4)
    %tile_2_5 = aie.tile(2, 5)
    %tile_3_2 = aie.tile(3, 2)
    %tile_3_3 = aie.tile(3, 3)
    %tile_3_4 = aie.tile(3, 4)
    %tile_3_5 = aie.tile(3, 5)
    %tile_4_2 = aie.tile(4, 2)
    %tile_4_3 = aie.tile(4, 3)
    %tile_4_4 = aie.tile(4, 4)
    %tile_4_5 = aie.tile(4, 5)
    %tile_5_2 = aie.tile(5, 2)
    %tile_5_3 = aie.tile(5, 3)
    %tile_5_4 = aie.tile(5, 4)
    %tile_5_5 = aie.tile(5, 5)
    %tile_6_2 = aie.tile(6, 2)
    %tile_6_3 = aie.tile(6, 3)
    %tile_6_4 = aie.tile(6, 4)
    %tile_6_5 = aie.tile(6, 5)
    %tile_7_2 = aie.tile(7, 2)
    %tile_7_3 = aie.tile(7, 3)
    %tile_7_4 = aie.tile(7, 4)
    %tile_7_5 = aie.tile(7, 5)
    // in0: broadcast per column (all 4 row tiles in a column share the same in0) 
    // col 0
    aie.objectfifo @in0_L3L2_0(%shim_noc_tile_0_0, {%mem_tile_0_1}, 2 : i32) : !aie.objectfifo<memref<16x64xbf16>>
    aie.objectfifo @in0_L2L1_0(%mem_tile_0_1 dimensionsToStream [<size = 2, stride = 512>, <size = 8, stride = 8>, <size = 8, stride = 64>, <size = 8, stride = 1>], {%tile_0_2, %tile_0_3, %tile_0_4, %tile_0_5}, 2 : i32) : !aie.objectfifo<memref<2x8x8x8xbf16>>
    aie.objectfifo.link [@in0_L3L2_0] -> [@in0_L2L1_0]([] [])
    // col 1
    aie.objectfifo @in0_L3L2_1(%shim_noc_tile_1_0, {%mem_tile_1_1}, 2 : i32) : !aie.objectfifo<memref<16x64xbf16>>
    aie.objectfifo @in0_L2L1_1(%mem_tile_1_1 dimensionsToStream [<size = 2, stride = 512>, <size = 8, stride = 8>, <size = 8, stride = 64>, <size = 8, stride = 1>], {%tile_1_2, %tile_1_3, %tile_1_4, %tile_1_5}, 2 : i32) : !aie.objectfifo<memref<2x8x8x8xbf16>>
    aie.objectfifo.link [@in0_L3L2_1] -> [@in0_L2L1_1]([] [])
    // col 2
    aie.objectfifo @in0_L3L2_2(%shim_noc_tile_2_0, {%mem_tile_2_1}, 2 : i32) : !aie.objectfifo<memref<16x64xbf16>>
    aie.objectfifo @in0_L2L1_2(%mem_tile_2_1 dimensionsToStream [<size = 2, stride = 512>, <size = 8, stride = 8>, <size = 8, stride = 64>, <size = 8, stride = 1>], {%tile_2_2, %tile_2_3, %tile_2_4, %tile_2_5}, 2 : i32) : !aie.objectfifo<memref<2x8x8x8xbf16>>
    aie.objectfifo.link [@in0_L3L2_2] -> [@in0_L2L1_2]([] [])
    // col 3
    aie.objectfifo @in0_L3L2_3(%shim_noc_tile_3_0, {%mem_tile_3_1}, 2 : i32) : !aie.objectfifo<memref<16x64xbf16>>
    aie.objectfifo @in0_L2L1_3(%mem_tile_3_1 dimensionsToStream [<size = 2, stride = 512>, <size = 8, stride = 8>, <size = 8, stride = 64>, <size = 8, stride = 1>], {%tile_3_2, %tile_3_3, %tile_3_4, %tile_3_5}, 2 : i32) : !aie.objectfifo<memref<2x8x8x8xbf16>>
    aie.objectfifo.link [@in0_L3L2_3] -> [@in0_L2L1_3]([] [])
    // col 4
    aie.objectfifo @in0_L3L2_4(%shim_noc_tile_4_0, {%mem_tile_4_1}, 2 : i32) : !aie.objectfifo<memref<16x64xbf16>>
    aie.objectfifo @in0_L2L1_4(%mem_tile_4_1 dimensionsToStream [<size = 2, stride = 512>, <size = 8, stride = 8>, <size = 8, stride = 64>, <size = 8, stride = 1>], {%tile_4_2, %tile_4_3, %tile_4_4, %tile_4_5}, 2 : i32) : !aie.objectfifo<memref<2x8x8x8xbf16>>
    aie.objectfifo.link [@in0_L3L2_4] -> [@in0_L2L1_4]([] [])
    // col 5
    aie.objectfifo @in0_L3L2_5(%shim_noc_tile_5_0, {%mem_tile_5_1}, 2 : i32) : !aie.objectfifo<memref<16x64xbf16>>
    aie.objectfifo @in0_L2L1_5(%mem_tile_5_1 dimensionsToStream [<size = 2, stride = 512>, <size = 8, stride = 8>, <size = 8, stride = 64>, <size = 8, stride = 1>], {%tile_5_2, %tile_5_3, %tile_5_4, %tile_5_5}, 2 : i32) : !aie.objectfifo<memref<2x8x8x8xbf16>>
    aie.objectfifo.link [@in0_L3L2_5] -> [@in0_L2L1_5]([] [])
    // col 6
    aie.objectfifo @in0_L3L2_6(%shim_noc_tile_6_0, {%mem_tile_6_1}, 2 : i32) : !aie.objectfifo<memref<16x64xbf16>>
    aie.objectfifo @in0_L2L1_6(%mem_tile_6_1 dimensionsToStream [<size = 2, stride = 512>, <size = 8, stride = 8>, <size = 8, stride = 64>, <size = 8, stride = 1>], {%tile_6_2, %tile_6_3, %tile_6_4, %tile_6_5}, 2 : i32) : !aie.objectfifo<memref<2x8x8x8xbf16>>
    aie.objectfifo.link [@in0_L3L2_6] -> [@in0_L2L1_6]([] [])
    // col 7
    aie.objectfifo @in0_L3L2_7(%shim_noc_tile_7_0, {%mem_tile_7_1}, 2 : i32) : !aie.objectfifo<memref<16x64xbf16>>
    aie.objectfifo @in0_L2L1_7(%mem_tile_7_1 dimensionsToStream [<size = 2, stride = 512>, <size = 8, stride = 8>, <size = 8, stride = 64>, <size = 8, stride = 1>], {%tile_7_2, %tile_7_3, %tile_7_4, %tile_7_5}, 2 : i32) : !aie.objectfifo<memref<2x8x8x8xbf16>>
    aie.objectfifo.link [@in0_L3L2_7] -> [@in0_L2L1_7]([] [])
    // in1: broadcast per row (all 8 column tiles in a row share the same in1) 
    // Only 4 in1_L3L2 FIFOs needed (one per y-row). Placed on shim tiles 0-3.
    // row_idx 0 (tile rows 2): mem tile 0 fans out to compute tiles col 0..7 row 2
    aie.objectfifo @in1_L3L2_0(%shim_noc_tile_0_0, {%mem_tile_0_1}, 2 : i32) : !aie.objectfifo<memref<64x16xbf16>>
    aie.objectfifo @in1_L2L1_0(%mem_tile_0_1 dimensionsToStream [<size = 8, stride = 128>, <size = 2, stride = 8>, <size = 8, stride = 16>, <size = 8, stride = 1>], {%tile_0_2, %tile_1_2, %tile_2_2, %tile_3_2, %tile_4_2, %tile_5_2, %tile_6_2, %tile_7_2}, 2 : i32) : !aie.objectfifo<memref<8x2x8x8xbf16>>
    aie.objectfifo.link [@in1_L3L2_0] -> [@in1_L2L1_0]([] [])
    // row_idx 1 (tile rows 3): mem tile 1 fans out to compute tiles col 0..7 row 3
    aie.objectfifo @in1_L3L2_1(%shim_noc_tile_1_0, {%mem_tile_1_1}, 2 : i32) : !aie.objectfifo<memref<64x16xbf16>>
    aie.objectfifo @in1_L2L1_1(%mem_tile_1_1 dimensionsToStream [<size = 8, stride = 128>, <size = 2, stride = 8>, <size = 8, stride = 16>, <size = 8, stride = 1>], {%tile_0_3, %tile_1_3, %tile_2_3, %tile_3_3, %tile_4_3, %tile_5_3, %tile_6_3, %tile_7_3}, 2 : i32) : !aie.objectfifo<memref<8x2x8x8xbf16>>
    aie.objectfifo.link [@in1_L3L2_1] -> [@in1_L2L1_1]([] [])
    // row_idx 2 (tile rows 4): mem tile 2 fans out to compute tiles col 0..7 row 4
    aie.objectfifo @in1_L3L2_2(%shim_noc_tile_2_0, {%mem_tile_2_1}, 2 : i32) : !aie.objectfifo<memref<64x16xbf16>>
    aie.objectfifo @in1_L2L1_2(%mem_tile_2_1 dimensionsToStream [<size = 8, stride = 128>, <size = 2, stride = 8>, <size = 8, stride = 16>, <size = 8, stride = 1>], {%tile_0_4, %tile_1_4, %tile_2_4, %tile_3_4, %tile_4_4, %tile_5_4, %tile_6_4, %tile_7_4}, 2 : i32) : !aie.objectfifo<memref<8x2x8x8xbf16>>
    aie.objectfifo.link [@in1_L3L2_2] -> [@in1_L2L1_2]([] [])
    // row_idx 3 (tile rows 5): mem tile 3 fans out to compute tiles col 0..7 row 5
    aie.objectfifo @in1_L3L2_3(%shim_noc_tile_3_0, {%mem_tile_3_1}, 2 : i32) : !aie.objectfifo<memref<64x16xbf16>>
    aie.objectfifo @in1_L2L1_3(%mem_tile_3_1 dimensionsToStream [<size = 8, stride = 128>, <size = 2, stride = 8>, <size = 8, stride = 16>, <size = 8, stride = 1>], {%tile_0_5, %tile_1_5, %tile_2_5, %tile_3_5, %tile_4_5, %tile_5_5, %tile_6_5, %tile_7_5}, 2 : i32) : !aie.objectfifo<memref<8x2x8x8xbf16>>
    aie.objectfifo.link [@in1_L3L2_3] -> [@in1_L2L1_3]([] [])
    // Output FIFOs: each column joins 4 row outputs into one L2L3 stream 
    // out_L1L2_<col>_<row_idx>: compute tile → mem tile, shape pmqn = 2×2×8×8
    // out_L2L3_<col>: mem tile → shim tile, shape ypqmn (joined 4 rows)
    //  dimensionsToStream maps [y, p, q, m, n] in buffer to row-major [y*p*m, q*n]
    //  => sizes:  [4, 2, 8, 2, 8]  strides: [256, 128, 8, 64, 1]  on a 4*256=1024-elem L2 buf
    //  Outer L3L2 memref<64x16xbf16> covers y*p*m=64 rows and q*n=16 cols
    // col 0
    aie.objectfifo @out_L1L2_0_0(%tile_0_2, {%mem_tile_0_1}, 2 : i32) : !aie.objectfifo<memref<2x2x8x8xbf16>>
    aie.objectfifo @out_L1L2_0_1(%tile_0_3, {%mem_tile_0_1}, 2 : i32) : !aie.objectfifo<memref<2x2x8x8xbf16>>
    aie.objectfifo @out_L1L2_0_2(%tile_0_4, {%mem_tile_0_1}, 2 : i32) : !aie.objectfifo<memref<2x2x8x8xbf16>>
    aie.objectfifo @out_L1L2_0_3(%tile_0_5, {%mem_tile_0_1}, 2 : i32) : !aie.objectfifo<memref<2x2x8x8xbf16>>
    aie.objectfifo @out_L2L3_0(%mem_tile_0_1 dimensionsToStream [<size = 2, stride = 128>, <size = 8, stride = 8>, <size = 2, stride = 64>, <size = 8, stride = 1>], {%shim_noc_tile_0_0}, 2 : i32) : !aie.objectfifo<memref<64x16xbf16>>
    aie.objectfifo.link [@out_L1L2_0_0, @out_L1L2_0_1, @out_L1L2_0_2, @out_L1L2_0_3] -> [@out_L2L3_0]([0, 256, 512, 768] [])
    // col 1
    aie.objectfifo @out_L1L2_1_0(%tile_1_2, {%mem_tile_1_1}, 2 : i32) : !aie.objectfifo<memref<2x2x8x8xbf16>>
    aie.objectfifo @out_L1L2_1_1(%tile_1_3, {%mem_tile_1_1}, 2 : i32) : !aie.objectfifo<memref<2x2x8x8xbf16>>
    aie.objectfifo @out_L1L2_1_2(%tile_1_4, {%mem_tile_1_1}, 2 : i32) : !aie.objectfifo<memref<2x2x8x8xbf16>>
    aie.objectfifo @out_L1L2_1_3(%tile_1_5, {%mem_tile_1_1}, 2 : i32) : !aie.objectfifo<memref<2x2x8x8xbf16>>
    aie.objectfifo @out_L2L3_1(%mem_tile_1_1 dimensionsToStream [<size = 2, stride = 128>, <size = 8, stride = 8>, <size = 2, stride = 64>, <size = 8, stride = 1>], {%shim_noc_tile_1_0}, 2 : i32) : !aie.objectfifo<memref<64x16xbf16>>
    aie.objectfifo.link [@out_L1L2_1_0, @out_L1L2_1_1, @out_L1L2_1_2, @out_L1L2_1_3] -> [@out_L2L3_1]([0, 256, 512, 768] [])
    // col 2
    aie.objectfifo @out_L1L2_2_0(%tile_2_2, {%mem_tile_2_1}, 2 : i32) : !aie.objectfifo<memref<2x2x8x8xbf16>>
    aie.objectfifo @out_L1L2_2_1(%tile_2_3, {%mem_tile_2_1}, 2 : i32) : !aie.objectfifo<memref<2x2x8x8xbf16>>
    aie.objectfifo @out_L1L2_2_2(%tile_2_4, {%mem_tile_2_1}, 2 : i32) : !aie.objectfifo<memref<2x2x8x8xbf16>>
    aie.objectfifo @out_L1L2_2_3(%tile_2_5, {%mem_tile_2_1}, 2 : i32) : !aie.objectfifo<memref<2x2x8x8xbf16>>
    aie.objectfifo @out_L2L3_2(%mem_tile_2_1 dimensionsToStream [<size = 2, stride = 128>, <size = 8, stride = 8>, <size = 2, stride = 64>, <size = 8, stride = 1>], {%shim_noc_tile_2_0}, 2 : i32) : !aie.objectfifo<memref<64x16xbf16>>
    aie.objectfifo.link [@out_L1L2_2_0, @out_L1L2_2_1, @out_L1L2_2_2, @out_L1L2_2_3] -> [@out_L2L3_2]([0, 256, 512, 768] [])
    // col 3
    aie.objectfifo @out_L1L2_3_0(%tile_3_2, {%mem_tile_3_1}, 2 : i32) : !aie.objectfifo<memref<2x2x8x8xbf16>>
    aie.objectfifo @out_L1L2_3_1(%tile_3_3, {%mem_tile_3_1}, 2 : i32) : !aie.objectfifo<memref<2x2x8x8xbf16>>
    aie.objectfifo @out_L1L2_3_2(%tile_3_4, {%mem_tile_3_1}, 2 : i32) : !aie.objectfifo<memref<2x2x8x8xbf16>>
    aie.objectfifo @out_L1L2_3_3(%tile_3_5, {%mem_tile_3_1}, 2 : i32) : !aie.objectfifo<memref<2x2x8x8xbf16>>
    aie.objectfifo @out_L2L3_3(%mem_tile_3_1 dimensionsToStream [<size = 2, stride = 128>, <size = 8, stride = 8>, <size = 2, stride = 64>, <size = 8, stride = 1>], {%shim_noc_tile_3_0}, 2 : i32) : !aie.objectfifo<memref<64x16xbf16>>
    aie.objectfifo.link [@out_L1L2_3_0, @out_L1L2_3_1, @out_L1L2_3_2, @out_L1L2_3_3] -> [@out_L2L3_3]([0, 256, 512, 768] [])
    // col 4
    aie.objectfifo @out_L1L2_4_0(%tile_4_2, {%mem_tile_4_1}, 2 : i32) : !aie.objectfifo<memref<2x2x8x8xbf16>>
    aie.objectfifo @out_L1L2_4_1(%tile_4_3, {%mem_tile_4_1}, 2 : i32) : !aie.objectfifo<memref<2x2x8x8xbf16>>
    aie.objectfifo @out_L1L2_4_2(%tile_4_4, {%mem_tile_4_1}, 2 : i32) : !aie.objectfifo<memref<2x2x8x8xbf16>>
    aie.objectfifo @out_L1L2_4_3(%tile_4_5, {%mem_tile_4_1}, 2 : i32) : !aie.objectfifo<memref<2x2x8x8xbf16>>
    aie.objectfifo @out_L2L3_4(%mem_tile_4_1 dimensionsToStream [<size = 2, stride = 128>, <size = 8, stride = 8>, <size = 2, stride = 64>, <size = 8, stride = 1>], {%shim_noc_tile_4_0}, 2 : i32) : !aie.objectfifo<memref<64x16xbf16>>
    aie.objectfifo.link [@out_L1L2_4_0, @out_L1L2_4_1, @out_L1L2_4_2, @out_L1L2_4_3] -> [@out_L2L3_4]([0, 256, 512, 768] [])
    // col 5
    aie.objectfifo @out_L1L2_5_0(%tile_5_2, {%mem_tile_5_1}, 2 : i32) : !aie.objectfifo<memref<2x2x8x8xbf16>>
    aie.objectfifo @out_L1L2_5_1(%tile_5_3, {%mem_tile_5_1}, 2 : i32) : !aie.objectfifo<memref<2x2x8x8xbf16>>
    aie.objectfifo @out_L1L2_5_2(%tile_5_4, {%mem_tile_5_1}, 2 : i32) : !aie.objectfifo<memref<2x2x8x8xbf16>>
    aie.objectfifo @out_L1L2_5_3(%tile_5_5, {%mem_tile_5_1}, 2 : i32) : !aie.objectfifo<memref<2x2x8x8xbf16>>
    aie.objectfifo @out_L2L3_5(%mem_tile_5_1 dimensionsToStream [<size = 2, stride = 128>, <size = 8, stride = 8>, <size = 2, stride = 64>, <size = 8, stride = 1>], {%shim_noc_tile_5_0}, 2 : i32) : !aie.objectfifo<memref<64x16xbf16>>
    aie.objectfifo.link [@out_L1L2_5_0, @out_L1L2_5_1, @out_L1L2_5_2, @out_L1L2_5_3] -> [@out_L2L3_5]([0, 256, 512, 768] [])
    // col 6
    aie.objectfifo @out_L1L2_6_0(%tile_6_2, {%mem_tile_6_1}, 2 : i32) : !aie.objectfifo<memref<2x2x8x8xbf16>>
    aie.objectfifo @out_L1L2_6_1(%tile_6_3, {%mem_tile_6_1}, 2 : i32) : !aie.objectfifo<memref<2x2x8x8xbf16>>
    aie.objectfifo @out_L1L2_6_2(%tile_6_4, {%mem_tile_6_1}, 2 : i32) : !aie.objectfifo<memref<2x2x8x8xbf16>>
    aie.objectfifo @out_L1L2_6_3(%tile_6_5, {%mem_tile_6_1}, 2 : i32) : !aie.objectfifo<memref<2x2x8x8xbf16>>
    aie.objectfifo @out_L2L3_6(%mem_tile_6_1 dimensionsToStream [<size = 2, stride = 128>, <size = 8, stride = 8>, <size = 2, stride = 64>, <size = 8, stride = 1>], {%shim_noc_tile_6_0}, 2 : i32) : !aie.objectfifo<memref<64x16xbf16>>
    aie.objectfifo.link [@out_L1L2_6_0, @out_L1L2_6_1, @out_L1L2_6_2, @out_L1L2_6_3] -> [@out_L2L3_6]([0, 256, 512, 768] [])
    // col 7
    aie.objectfifo @out_L1L2_7_0(%tile_7_2, {%mem_tile_7_1}, 2 : i32) : !aie.objectfifo<memref<2x2x8x8xbf16>>
    aie.objectfifo @out_L1L2_7_1(%tile_7_3, {%mem_tile_7_1}, 2 : i32) : !aie.objectfifo<memref<2x2x8x8xbf16>>
    aie.objectfifo @out_L1L2_7_2(%tile_7_4, {%mem_tile_7_1}, 2 : i32) : !aie.objectfifo<memref<2x2x8x8xbf16>>
    aie.objectfifo @out_L1L2_7_3(%tile_7_5, {%mem_tile_7_1}, 2 : i32) : !aie.objectfifo<memref<2x2x8x8xbf16>>
    aie.objectfifo @out_L2L3_7(%mem_tile_7_1 dimensionsToStream [<size = 2, stride = 128>, <size = 8, stride = 8>, <size = 2, stride = 64>, <size = 8, stride = 1>], {%shim_noc_tile_7_0}, 2 : i32) : !aie.objectfifo<memref<64x16xbf16>>
    aie.objectfifo.link [@out_L1L2_7_0, @out_L1L2_7_1, @out_L1L2_7_2, @out_L1L2_7_3] -> [@out_L2L3_7]([0, 256, 512, 768] [])
    %core_0_2 = aie.core(%tile_0_2) {
      %c0 = arith.constant 0 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c4294967295 step %c1 {
        %c0_1 = arith.constant 0 : index
        %c4 = arith.constant 4 : index
        %c1_1 = arith.constant 1 : index
        scf.for %i_ab = %c0_1 to %c4 step %c1_1 {
          %buffer_out = aie.objectfifo.acquire @out_L1L2_0_0(Produce, 1) : !aie.objectfifosubview<memref<2x2x8x8xbf16>>
          %out = aie.objectfifo.subview.access %buffer_out[0] : !aie.objectfifosubview<memref<2x2x8x8xbf16>> -> memref<2x2x8x8xbf16>
          func.call @zero(%out) : (memref<2x2x8x8xbf16>) -> ()
          %c0_2 = arith.constant 0 : index
          %c16 = arith.constant 16 : index
          %c1_2 = arith.constant 1 : index
          scf.for %i_c = %c0_2 to %c16 step %c1_2 {
            %buffer_in0 = aie.objectfifo.acquire @in0_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<2x8x8x8xbf16>>
            %in0 = aie.objectfifo.subview.access %buffer_in0[0] : !aie.objectfifosubview<memref<2x8x8x8xbf16>> -> memref<2x8x8x8xbf16>
            %buffer_in1 = aie.objectfifo.acquire @in1_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<8x2x8x8xbf16>>
            %in1 = aie.objectfifo.subview.access %buffer_in1[0] : !aie.objectfifosubview<memref<8x2x8x8xbf16>> -> memref<8x2x8x8xbf16>
            func.call @matmul(%in0, %in1, %out) : (memref<2x8x8x8xbf16>, memref<8x2x8x8xbf16>, memref<2x2x8x8xbf16>) -> ()
            aie.objectfifo.release @in0_L2L1_0(Consume, 1)
            aie.objectfifo.release @in1_L2L1_0(Consume, 1)
          }
          aie.objectfifo.release @out_L1L2_0_0(Produce, 1)
        }
      }
      aie.end
    } {stack_size = 1024 : i32}
    // col=0, row_idx=1  (tile_0_3)
    %core_0_3 = aie.core(%tile_0_3) {
      %c0 = arith.constant 0 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c4294967295 step %c1 {
        %c0_1 = arith.constant 0 : index
        %c4 = arith.constant 4 : index
        %c1_1 = arith.constant 1 : index
        scf.for %i_ab = %c0_1 to %c4 step %c1_1 {
          %buffer_out = aie.objectfifo.acquire @out_L1L2_0_1(Produce, 1) : !aie.objectfifosubview<memref<2x2x8x8xbf16>>
          %out = aie.objectfifo.subview.access %buffer_out[0] : !aie.objectfifosubview<memref<2x2x8x8xbf16>> -> memref<2x2x8x8xbf16>
          func.call @zero(%out) : (memref<2x2x8x8xbf16>) -> ()
          %c0_2 = arith.constant 0 : index
          %c16 = arith.constant 16 : index
          %c1_2 = arith.constant 1 : index
          scf.for %i_c = %c0_2 to %c16 step %c1_2 {
            %buffer_in0 = aie.objectfifo.acquire @in0_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<2x8x8x8xbf16>>
            %in0 = aie.objectfifo.subview.access %buffer_in0[0] : !aie.objectfifosubview<memref<2x8x8x8xbf16>> -> memref<2x8x8x8xbf16>
            %buffer_in1 = aie.objectfifo.acquire @in1_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<8x2x8x8xbf16>>
            %in1 = aie.objectfifo.subview.access %buffer_in1[0] : !aie.objectfifosubview<memref<8x2x8x8xbf16>> -> memref<8x2x8x8xbf16>
            func.call @matmul(%in0, %in1, %out) : (memref<2x8x8x8xbf16>, memref<8x2x8x8xbf16>, memref<2x2x8x8xbf16>) -> ()
            aie.objectfifo.release @in0_L2L1_0(Consume, 1)
            aie.objectfifo.release @in1_L2L1_1(Consume, 1)
          }
          aie.objectfifo.release @out_L1L2_0_1(Produce, 1)
        }
      }
      aie.end
    } {stack_size = 1024 : i32}
    // col=0, row_idx=2  (tile_0_4)
    %core_0_4 = aie.core(%tile_0_4) {
      %c0 = arith.constant 0 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c4294967295 step %c1 {
        %c0_1 = arith.constant 0 : index
        %c4 = arith.constant 4 : index
        %c1_1 = arith.constant 1 : index
        scf.for %i_ab = %c0_1 to %c4 step %c1_1 {
          %buffer_out = aie.objectfifo.acquire @out_L1L2_0_2(Produce, 1) : !aie.objectfifosubview<memref<2x2x8x8xbf16>>
          %out = aie.objectfifo.subview.access %buffer_out[0] : !aie.objectfifosubview<memref<2x2x8x8xbf16>> -> memref<2x2x8x8xbf16>
          func.call @zero(%out) : (memref<2x2x8x8xbf16>) -> ()
          %c0_2 = arith.constant 0 : index
          %c16 = arith.constant 16 : index
          %c1_2 = arith.constant 1 : index
          scf.for %i_c = %c0_2 to %c16 step %c1_2 {
            %buffer_in0 = aie.objectfifo.acquire @in0_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<2x8x8x8xbf16>>
            %in0 = aie.objectfifo.subview.access %buffer_in0[0] : !aie.objectfifosubview<memref<2x8x8x8xbf16>> -> memref<2x8x8x8xbf16>
            %buffer_in1 = aie.objectfifo.acquire @in1_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<8x2x8x8xbf16>>
            %in1 = aie.objectfifo.subview.access %buffer_in1[0] : !aie.objectfifosubview<memref<8x2x8x8xbf16>> -> memref<8x2x8x8xbf16>
            func.call @matmul(%in0, %in1, %out) : (memref<2x8x8x8xbf16>, memref<8x2x8x8xbf16>, memref<2x2x8x8xbf16>) -> ()
            aie.objectfifo.release @in0_L2L1_0(Consume, 1)
            aie.objectfifo.release @in1_L2L1_2(Consume, 1)
          }
          aie.objectfifo.release @out_L1L2_0_2(Produce, 1)
        }
      }
      aie.end
    } {stack_size = 1024 : i32}
    // col=0, row_idx=3  (tile_0_5)
    %core_0_5 = aie.core(%tile_0_5) {
      %c0 = arith.constant 0 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c4294967295 step %c1 {
        %c0_1 = arith.constant 0 : index
        %c4 = arith.constant 4 : index
        %c1_1 = arith.constant 1 : index
        scf.for %i_ab = %c0_1 to %c4 step %c1_1 {
          %buffer_out = aie.objectfifo.acquire @out_L1L2_0_3(Produce, 1) : !aie.objectfifosubview<memref<2x2x8x8xbf16>>
          %out = aie.objectfifo.subview.access %buffer_out[0] : !aie.objectfifosubview<memref<2x2x8x8xbf16>> -> memref<2x2x8x8xbf16>
          func.call @zero(%out) : (memref<2x2x8x8xbf16>) -> ()
          %c0_2 = arith.constant 0 : index
          %c16 = arith.constant 16 : index
          %c1_2 = arith.constant 1 : index
          scf.for %i_c = %c0_2 to %c16 step %c1_2 {
            %buffer_in0 = aie.objectfifo.acquire @in0_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<2x8x8x8xbf16>>
            %in0 = aie.objectfifo.subview.access %buffer_in0[0] : !aie.objectfifosubview<memref<2x8x8x8xbf16>> -> memref<2x8x8x8xbf16>
            %buffer_in1 = aie.objectfifo.acquire @in1_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<8x2x8x8xbf16>>
            %in1 = aie.objectfifo.subview.access %buffer_in1[0] : !aie.objectfifosubview<memref<8x2x8x8xbf16>> -> memref<8x2x8x8xbf16>
            func.call @matmul(%in0, %in1, %out) : (memref<2x8x8x8xbf16>, memref<8x2x8x8xbf16>, memref<2x2x8x8xbf16>) -> ()
            aie.objectfifo.release @in0_L2L1_0(Consume, 1)
            aie.objectfifo.release @in1_L2L1_3(Consume, 1)
          }
          aie.objectfifo.release @out_L1L2_0_3(Produce, 1)
        }
      }
      aie.end
    } {stack_size = 1024 : i32}
    // col=1, row_idx=0  (tile_1_2)
    %core_1_2 = aie.core(%tile_1_2) {
      %c0 = arith.constant 0 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c4294967295 step %c1 {
        %c0_1 = arith.constant 0 : index
        %c4 = arith.constant 4 : index
        %c1_1 = arith.constant 1 : index
        scf.for %i_ab = %c0_1 to %c4 step %c1_1 {
          %buffer_out = aie.objectfifo.acquire @out_L1L2_1_0(Produce, 1) : !aie.objectfifosubview<memref<2x2x8x8xbf16>>
          %out = aie.objectfifo.subview.access %buffer_out[0] : !aie.objectfifosubview<memref<2x2x8x8xbf16>> -> memref<2x2x8x8xbf16>
          func.call @zero(%out) : (memref<2x2x8x8xbf16>) -> ()
          %c0_2 = arith.constant 0 : index
          %c16 = arith.constant 16 : index
          %c1_2 = arith.constant 1 : index
          scf.for %i_c = %c0_2 to %c16 step %c1_2 {
            %buffer_in0 = aie.objectfifo.acquire @in0_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<2x8x8x8xbf16>>
            %in0 = aie.objectfifo.subview.access %buffer_in0[0] : !aie.objectfifosubview<memref<2x8x8x8xbf16>> -> memref<2x8x8x8xbf16>
            %buffer_in1 = aie.objectfifo.acquire @in1_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<8x2x8x8xbf16>>
            %in1 = aie.objectfifo.subview.access %buffer_in1[0] : !aie.objectfifosubview<memref<8x2x8x8xbf16>> -> memref<8x2x8x8xbf16>
            func.call @matmul(%in0, %in1, %out) : (memref<2x8x8x8xbf16>, memref<8x2x8x8xbf16>, memref<2x2x8x8xbf16>) -> ()
            aie.objectfifo.release @in0_L2L1_1(Consume, 1)
            aie.objectfifo.release @in1_L2L1_0(Consume, 1)
          }
          aie.objectfifo.release @out_L1L2_1_0(Produce, 1)
        }
      }
      aie.end
    } {stack_size = 1024 : i32}
    // col=1, row_idx=1  (tile_1_3)
    %core_1_3 = aie.core(%tile_1_3) {
      %c0 = arith.constant 0 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c4294967295 step %c1 {
        %c0_1 = arith.constant 0 : index
        %c4 = arith.constant 4 : index
        %c1_1 = arith.constant 1 : index
        scf.for %i_ab = %c0_1 to %c4 step %c1_1 {
          %buffer_out = aie.objectfifo.acquire @out_L1L2_1_1(Produce, 1) : !aie.objectfifosubview<memref<2x2x8x8xbf16>>
          %out = aie.objectfifo.subview.access %buffer_out[0] : !aie.objectfifosubview<memref<2x2x8x8xbf16>> -> memref<2x2x8x8xbf16>
          func.call @zero(%out) : (memref<2x2x8x8xbf16>) -> ()
          %c0_2 = arith.constant 0 : index
          %c16 = arith.constant 16 : index
          %c1_2 = arith.constant 1 : index
          scf.for %i_c = %c0_2 to %c16 step %c1_2 {
            %buffer_in0 = aie.objectfifo.acquire @in0_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<2x8x8x8xbf16>>
            %in0 = aie.objectfifo.subview.access %buffer_in0[0] : !aie.objectfifosubview<memref<2x8x8x8xbf16>> -> memref<2x8x8x8xbf16>
            %buffer_in1 = aie.objectfifo.acquire @in1_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<8x2x8x8xbf16>>
            %in1 = aie.objectfifo.subview.access %buffer_in1[0] : !aie.objectfifosubview<memref<8x2x8x8xbf16>> -> memref<8x2x8x8xbf16>
            func.call @matmul(%in0, %in1, %out) : (memref<2x8x8x8xbf16>, memref<8x2x8x8xbf16>, memref<2x2x8x8xbf16>) -> ()
            aie.objectfifo.release @in0_L2L1_1(Consume, 1)
            aie.objectfifo.release @in1_L2L1_1(Consume, 1)
          }
          aie.objectfifo.release @out_L1L2_1_1(Produce, 1)
        }
      }
      aie.end
    } {stack_size = 1024 : i32}
    // col=1, row_idx=2  (tile_1_4)
    %core_1_4 = aie.core(%tile_1_4) {
      %c0 = arith.constant 0 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c4294967295 step %c1 {
        %c0_1 = arith.constant 0 : index
        %c4 = arith.constant 4 : index
        %c1_1 = arith.constant 1 : index
        scf.for %i_ab = %c0_1 to %c4 step %c1_1 {
          %buffer_out = aie.objectfifo.acquire @out_L1L2_1_2(Produce, 1) : !aie.objectfifosubview<memref<2x2x8x8xbf16>>
          %out = aie.objectfifo.subview.access %buffer_out[0] : !aie.objectfifosubview<memref<2x2x8x8xbf16>> -> memref<2x2x8x8xbf16>
          func.call @zero(%out) : (memref<2x2x8x8xbf16>) -> ()
          %c0_2 = arith.constant 0 : index
          %c16 = arith.constant 16 : index
          %c1_2 = arith.constant 1 : index
          scf.for %i_c = %c0_2 to %c16 step %c1_2 {
            %buffer_in0 = aie.objectfifo.acquire @in0_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<2x8x8x8xbf16>>
            %in0 = aie.objectfifo.subview.access %buffer_in0[0] : !aie.objectfifosubview<memref<2x8x8x8xbf16>> -> memref<2x8x8x8xbf16>
            %buffer_in1 = aie.objectfifo.acquire @in1_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<8x2x8x8xbf16>>
            %in1 = aie.objectfifo.subview.access %buffer_in1[0] : !aie.objectfifosubview<memref<8x2x8x8xbf16>> -> memref<8x2x8x8xbf16>
            func.call @matmul(%in0, %in1, %out) : (memref<2x8x8x8xbf16>, memref<8x2x8x8xbf16>, memref<2x2x8x8xbf16>) -> ()
            aie.objectfifo.release @in0_L2L1_1(Consume, 1)
            aie.objectfifo.release @in1_L2L1_2(Consume, 1)
          }
          aie.objectfifo.release @out_L1L2_1_2(Produce, 1)
        }
      }
      aie.end
    } {stack_size = 1024 : i32}
    // col=1, row_idx=3  (tile_1_5)
    %core_1_5 = aie.core(%tile_1_5) {
      %c0 = arith.constant 0 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c4294967295 step %c1 {
        %c0_1 = arith.constant 0 : index
        %c4 = arith.constant 4 : index
        %c1_1 = arith.constant 1 : index
        scf.for %i_ab = %c0_1 to %c4 step %c1_1 {
          %buffer_out = aie.objectfifo.acquire @out_L1L2_1_3(Produce, 1) : !aie.objectfifosubview<memref<2x2x8x8xbf16>>
          %out = aie.objectfifo.subview.access %buffer_out[0] : !aie.objectfifosubview<memref<2x2x8x8xbf16>> -> memref<2x2x8x8xbf16>
          func.call @zero(%out) : (memref<2x2x8x8xbf16>) -> ()
          %c0_2 = arith.constant 0 : index
          %c16 = arith.constant 16 : index
          %c1_2 = arith.constant 1 : index
          scf.for %i_c = %c0_2 to %c16 step %c1_2 {
            %buffer_in0 = aie.objectfifo.acquire @in0_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<2x8x8x8xbf16>>
            %in0 = aie.objectfifo.subview.access %buffer_in0[0] : !aie.objectfifosubview<memref<2x8x8x8xbf16>> -> memref<2x8x8x8xbf16>
            %buffer_in1 = aie.objectfifo.acquire @in1_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<8x2x8x8xbf16>>
            %in1 = aie.objectfifo.subview.access %buffer_in1[0] : !aie.objectfifosubview<memref<8x2x8x8xbf16>> -> memref<8x2x8x8xbf16>
            func.call @matmul(%in0, %in1, %out) : (memref<2x8x8x8xbf16>, memref<8x2x8x8xbf16>, memref<2x2x8x8xbf16>) -> ()
            aie.objectfifo.release @in0_L2L1_1(Consume, 1)
            aie.objectfifo.release @in1_L2L1_3(Consume, 1)
          }
          aie.objectfifo.release @out_L1L2_1_3(Produce, 1)
        }
      }
      aie.end
    } {stack_size = 1024 : i32}
    // col=2, row_idx=0  (tile_2_2)
    %core_2_2 = aie.core(%tile_2_2) {
      %c0 = arith.constant 0 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c4294967295 step %c1 {
        %c0_1 = arith.constant 0 : index
        %c4 = arith.constant 4 : index
        %c1_1 = arith.constant 1 : index
        scf.for %i_ab = %c0_1 to %c4 step %c1_1 {
          %buffer_out = aie.objectfifo.acquire @out_L1L2_2_0(Produce, 1) : !aie.objectfifosubview<memref<2x2x8x8xbf16>>
          %out = aie.objectfifo.subview.access %buffer_out[0] : !aie.objectfifosubview<memref<2x2x8x8xbf16>> -> memref<2x2x8x8xbf16>
          func.call @zero(%out) : (memref<2x2x8x8xbf16>) -> ()
          %c0_2 = arith.constant 0 : index
          %c16 = arith.constant 16 : index
          %c1_2 = arith.constant 1 : index
          scf.for %i_c = %c0_2 to %c16 step %c1_2 {
            %buffer_in0 = aie.objectfifo.acquire @in0_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<2x8x8x8xbf16>>
            %in0 = aie.objectfifo.subview.access %buffer_in0[0] : !aie.objectfifosubview<memref<2x8x8x8xbf16>> -> memref<2x8x8x8xbf16>
            %buffer_in1 = aie.objectfifo.acquire @in1_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<8x2x8x8xbf16>>
            %in1 = aie.objectfifo.subview.access %buffer_in1[0] : !aie.objectfifosubview<memref<8x2x8x8xbf16>> -> memref<8x2x8x8xbf16>
            func.call @matmul(%in0, %in1, %out) : (memref<2x8x8x8xbf16>, memref<8x2x8x8xbf16>, memref<2x2x8x8xbf16>) -> ()
            aie.objectfifo.release @in0_L2L1_2(Consume, 1)
            aie.objectfifo.release @in1_L2L1_0(Consume, 1)
          }
          aie.objectfifo.release @out_L1L2_2_0(Produce, 1)
        }
      }
      aie.end
    } {stack_size = 1024 : i32}
    // col=2, row_idx=1  (tile_2_3)
    %core_2_3 = aie.core(%tile_2_3) {
      %c0 = arith.constant 0 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c4294967295 step %c1 {
        %c0_1 = arith.constant 0 : index
        %c4 = arith.constant 4 : index
        %c1_1 = arith.constant 1 : index
        scf.for %i_ab = %c0_1 to %c4 step %c1_1 {
          %buffer_out = aie.objectfifo.acquire @out_L1L2_2_1(Produce, 1) : !aie.objectfifosubview<memref<2x2x8x8xbf16>>
          %out = aie.objectfifo.subview.access %buffer_out[0] : !aie.objectfifosubview<memref<2x2x8x8xbf16>> -> memref<2x2x8x8xbf16>
          func.call @zero(%out) : (memref<2x2x8x8xbf16>) -> ()
          %c0_2 = arith.constant 0 : index
          %c16 = arith.constant 16 : index
          %c1_2 = arith.constant 1 : index
          scf.for %i_c = %c0_2 to %c16 step %c1_2 {
            %buffer_in0 = aie.objectfifo.acquire @in0_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<2x8x8x8xbf16>>
            %in0 = aie.objectfifo.subview.access %buffer_in0[0] : !aie.objectfifosubview<memref<2x8x8x8xbf16>> -> memref<2x8x8x8xbf16>
            %buffer_in1 = aie.objectfifo.acquire @in1_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<8x2x8x8xbf16>>
            %in1 = aie.objectfifo.subview.access %buffer_in1[0] : !aie.objectfifosubview<memref<8x2x8x8xbf16>> -> memref<8x2x8x8xbf16>
            func.call @matmul(%in0, %in1, %out) : (memref<2x8x8x8xbf16>, memref<8x2x8x8xbf16>, memref<2x2x8x8xbf16>) -> ()
            aie.objectfifo.release @in0_L2L1_2(Consume, 1)
            aie.objectfifo.release @in1_L2L1_1(Consume, 1)
          }
          aie.objectfifo.release @out_L1L2_2_1(Produce, 1)
        }
      }
      aie.end
    } {stack_size = 1024 : i32}
    // col=2, row_idx=2  (tile_2_4)
    %core_2_4 = aie.core(%tile_2_4) {
      %c0 = arith.constant 0 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c4294967295 step %c1 {
        %c0_1 = arith.constant 0 : index
        %c4 = arith.constant 4 : index
        %c1_1 = arith.constant 1 : index
        scf.for %i_ab = %c0_1 to %c4 step %c1_1 {
          %buffer_out = aie.objectfifo.acquire @out_L1L2_2_2(Produce, 1) : !aie.objectfifosubview<memref<2x2x8x8xbf16>>
          %out = aie.objectfifo.subview.access %buffer_out[0] : !aie.objectfifosubview<memref<2x2x8x8xbf16>> -> memref<2x2x8x8xbf16>
          func.call @zero(%out) : (memref<2x2x8x8xbf16>) -> ()
          %c0_2 = arith.constant 0 : index
          %c16 = arith.constant 16 : index
          %c1_2 = arith.constant 1 : index
          scf.for %i_c = %c0_2 to %c16 step %c1_2 {
            %buffer_in0 = aie.objectfifo.acquire @in0_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<2x8x8x8xbf16>>
            %in0 = aie.objectfifo.subview.access %buffer_in0[0] : !aie.objectfifosubview<memref<2x8x8x8xbf16>> -> memref<2x8x8x8xbf16>
            %buffer_in1 = aie.objectfifo.acquire @in1_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<8x2x8x8xbf16>>
            %in1 = aie.objectfifo.subview.access %buffer_in1[0] : !aie.objectfifosubview<memref<8x2x8x8xbf16>> -> memref<8x2x8x8xbf16>
            func.call @matmul(%in0, %in1, %out) : (memref<2x8x8x8xbf16>, memref<8x2x8x8xbf16>, memref<2x2x8x8xbf16>) -> ()
            aie.objectfifo.release @in0_L2L1_2(Consume, 1)
            aie.objectfifo.release @in1_L2L1_2(Consume, 1)
          }
          aie.objectfifo.release @out_L1L2_2_2(Produce, 1)
        }
      }
      aie.end
    } {stack_size = 1024 : i32}
    // col=2, row_idx=3  (tile_2_5)
    %core_2_5 = aie.core(%tile_2_5) {
      %c0 = arith.constant 0 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c4294967295 step %c1 {
        %c0_1 = arith.constant 0 : index
        %c4 = arith.constant 4 : index
        %c1_1 = arith.constant 1 : index
        scf.for %i_ab = %c0_1 to %c4 step %c1_1 {
          %buffer_out = aie.objectfifo.acquire @out_L1L2_2_3(Produce, 1) : !aie.objectfifosubview<memref<2x2x8x8xbf16>>
          %out = aie.objectfifo.subview.access %buffer_out[0] : !aie.objectfifosubview<memref<2x2x8x8xbf16>> -> memref<2x2x8x8xbf16>
          func.call @zero(%out) : (memref<2x2x8x8xbf16>) -> ()
          %c0_2 = arith.constant 0 : index
          %c16 = arith.constant 16 : index
          %c1_2 = arith.constant 1 : index
          scf.for %i_c = %c0_2 to %c16 step %c1_2 {
            %buffer_in0 = aie.objectfifo.acquire @in0_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<2x8x8x8xbf16>>
            %in0 = aie.objectfifo.subview.access %buffer_in0[0] : !aie.objectfifosubview<memref<2x8x8x8xbf16>> -> memref<2x8x8x8xbf16>
            %buffer_in1 = aie.objectfifo.acquire @in1_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<8x2x8x8xbf16>>
            %in1 = aie.objectfifo.subview.access %buffer_in1[0] : !aie.objectfifosubview<memref<8x2x8x8xbf16>> -> memref<8x2x8x8xbf16>
            func.call @matmul(%in0, %in1, %out) : (memref<2x8x8x8xbf16>, memref<8x2x8x8xbf16>, memref<2x2x8x8xbf16>) -> ()
            aie.objectfifo.release @in0_L2L1_2(Consume, 1)
            aie.objectfifo.release @in1_L2L1_3(Consume, 1)
          }
          aie.objectfifo.release @out_L1L2_2_3(Produce, 1)
        }
      }
      aie.end
    } {stack_size = 1024 : i32}
    // col=3, row_idx=0  (tile_3_2)
    %core_3_2 = aie.core(%tile_3_2) {
      %c0 = arith.constant 0 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c4294967295 step %c1 {
        %c0_1 = arith.constant 0 : index
        %c4 = arith.constant 4 : index
        %c1_1 = arith.constant 1 : index
        scf.for %i_ab = %c0_1 to %c4 step %c1_1 {
          %buffer_out = aie.objectfifo.acquire @out_L1L2_3_0(Produce, 1) : !aie.objectfifosubview<memref<2x2x8x8xbf16>>
          %out = aie.objectfifo.subview.access %buffer_out[0] : !aie.objectfifosubview<memref<2x2x8x8xbf16>> -> memref<2x2x8x8xbf16>
          func.call @zero(%out) : (memref<2x2x8x8xbf16>) -> ()
          %c0_2 = arith.constant 0 : index
          %c16 = arith.constant 16 : index
          %c1_2 = arith.constant 1 : index
          scf.for %i_c = %c0_2 to %c16 step %c1_2 {
            %buffer_in0 = aie.objectfifo.acquire @in0_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<2x8x8x8xbf16>>
            %in0 = aie.objectfifo.subview.access %buffer_in0[0] : !aie.objectfifosubview<memref<2x8x8x8xbf16>> -> memref<2x8x8x8xbf16>
            %buffer_in1 = aie.objectfifo.acquire @in1_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<8x2x8x8xbf16>>
            %in1 = aie.objectfifo.subview.access %buffer_in1[0] : !aie.objectfifosubview<memref<8x2x8x8xbf16>> -> memref<8x2x8x8xbf16>
            func.call @matmul(%in0, %in1, %out) : (memref<2x8x8x8xbf16>, memref<8x2x8x8xbf16>, memref<2x2x8x8xbf16>) -> ()
            aie.objectfifo.release @in0_L2L1_3(Consume, 1)
            aie.objectfifo.release @in1_L2L1_0(Consume, 1)
          }
          aie.objectfifo.release @out_L1L2_3_0(Produce, 1)
        }
      }
      aie.end
    } {stack_size = 1024 : i32}
    // col=3, row_idx=1  (tile_3_3)
    %core_3_3 = aie.core(%tile_3_3) {
      %c0 = arith.constant 0 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c4294967295 step %c1 {
        %c0_1 = arith.constant 0 : index
        %c4 = arith.constant 4 : index
        %c1_1 = arith.constant 1 : index
        scf.for %i_ab = %c0_1 to %c4 step %c1_1 {
          %buffer_out = aie.objectfifo.acquire @out_L1L2_3_1(Produce, 1) : !aie.objectfifosubview<memref<2x2x8x8xbf16>>
          %out = aie.objectfifo.subview.access %buffer_out[0] : !aie.objectfifosubview<memref<2x2x8x8xbf16>> -> memref<2x2x8x8xbf16>
          func.call @zero(%out) : (memref<2x2x8x8xbf16>) -> ()
          %c0_2 = arith.constant 0 : index
          %c16 = arith.constant 16 : index
          %c1_2 = arith.constant 1 : index
          scf.for %i_c = %c0_2 to %c16 step %c1_2 {
            %buffer_in0 = aie.objectfifo.acquire @in0_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<2x8x8x8xbf16>>
            %in0 = aie.objectfifo.subview.access %buffer_in0[0] : !aie.objectfifosubview<memref<2x8x8x8xbf16>> -> memref<2x8x8x8xbf16>
            %buffer_in1 = aie.objectfifo.acquire @in1_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<8x2x8x8xbf16>>
            %in1 = aie.objectfifo.subview.access %buffer_in1[0] : !aie.objectfifosubview<memref<8x2x8x8xbf16>> -> memref<8x2x8x8xbf16>
            func.call @matmul(%in0, %in1, %out) : (memref<2x8x8x8xbf16>, memref<8x2x8x8xbf16>, memref<2x2x8x8xbf16>) -> ()
            aie.objectfifo.release @in0_L2L1_3(Consume, 1)
            aie.objectfifo.release @in1_L2L1_1(Consume, 1)
          }
          aie.objectfifo.release @out_L1L2_3_1(Produce, 1)
        }
      }
      aie.end
    } {stack_size = 1024 : i32}
    // col=3, row_idx=2  (tile_3_4)
    %core_3_4 = aie.core(%tile_3_4) {
      %c0 = arith.constant 0 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c4294967295 step %c1 {
        %c0_1 = arith.constant 0 : index
        %c4 = arith.constant 4 : index
        %c1_1 = arith.constant 1 : index
        scf.for %i_ab = %c0_1 to %c4 step %c1_1 {
          %buffer_out = aie.objectfifo.acquire @out_L1L2_3_2(Produce, 1) : !aie.objectfifosubview<memref<2x2x8x8xbf16>>
          %out = aie.objectfifo.subview.access %buffer_out[0] : !aie.objectfifosubview<memref<2x2x8x8xbf16>> -> memref<2x2x8x8xbf16>
          func.call @zero(%out) : (memref<2x2x8x8xbf16>) -> ()
          %c0_2 = arith.constant 0 : index
          %c16 = arith.constant 16 : index
          %c1_2 = arith.constant 1 : index
          scf.for %i_c = %c0_2 to %c16 step %c1_2 {
            %buffer_in0 = aie.objectfifo.acquire @in0_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<2x8x8x8xbf16>>
            %in0 = aie.objectfifo.subview.access %buffer_in0[0] : !aie.objectfifosubview<memref<2x8x8x8xbf16>> -> memref<2x8x8x8xbf16>
            %buffer_in1 = aie.objectfifo.acquire @in1_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<8x2x8x8xbf16>>
            %in1 = aie.objectfifo.subview.access %buffer_in1[0] : !aie.objectfifosubview<memref<8x2x8x8xbf16>> -> memref<8x2x8x8xbf16>
            func.call @matmul(%in0, %in1, %out) : (memref<2x8x8x8xbf16>, memref<8x2x8x8xbf16>, memref<2x2x8x8xbf16>) -> ()
            aie.objectfifo.release @in0_L2L1_3(Consume, 1)
            aie.objectfifo.release @in1_L2L1_2(Consume, 1)
          }
          aie.objectfifo.release @out_L1L2_3_2(Produce, 1)
        }
      }
      aie.end
    } {stack_size = 1024 : i32}
    // col=3, row_idx=3  (tile_3_5)
    %core_3_5 = aie.core(%tile_3_5) {
      %c0 = arith.constant 0 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c4294967295 step %c1 {
        %c0_1 = arith.constant 0 : index
        %c4 = arith.constant 4 : index
        %c1_1 = arith.constant 1 : index
        scf.for %i_ab = %c0_1 to %c4 step %c1_1 {
          %buffer_out = aie.objectfifo.acquire @out_L1L2_3_3(Produce, 1) : !aie.objectfifosubview<memref<2x2x8x8xbf16>>
          %out = aie.objectfifo.subview.access %buffer_out[0] : !aie.objectfifosubview<memref<2x2x8x8xbf16>> -> memref<2x2x8x8xbf16>
          func.call @zero(%out) : (memref<2x2x8x8xbf16>) -> ()
          %c0_2 = arith.constant 0 : index
          %c16 = arith.constant 16 : index
          %c1_2 = arith.constant 1 : index
          scf.for %i_c = %c0_2 to %c16 step %c1_2 {
            %buffer_in0 = aie.objectfifo.acquire @in0_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<2x8x8x8xbf16>>
            %in0 = aie.objectfifo.subview.access %buffer_in0[0] : !aie.objectfifosubview<memref<2x8x8x8xbf16>> -> memref<2x8x8x8xbf16>
            %buffer_in1 = aie.objectfifo.acquire @in1_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<8x2x8x8xbf16>>
            %in1 = aie.objectfifo.subview.access %buffer_in1[0] : !aie.objectfifosubview<memref<8x2x8x8xbf16>> -> memref<8x2x8x8xbf16>
            func.call @matmul(%in0, %in1, %out) : (memref<2x8x8x8xbf16>, memref<8x2x8x8xbf16>, memref<2x2x8x8xbf16>) -> ()
            aie.objectfifo.release @in0_L2L1_3(Consume, 1)
            aie.objectfifo.release @in1_L2L1_3(Consume, 1)
          }
          aie.objectfifo.release @out_L1L2_3_3(Produce, 1)
        }
      }
      aie.end
    } {stack_size = 1024 : i32}
    // col=4, row_idx=0  (tile_4_2)
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
    // col=4, row_idx=1  (tile_4_3)
    %core_4_3 = aie.core(%tile_4_3) {
      %c0 = arith.constant 0 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c4294967295 step %c1 {
        %c0_1 = arith.constant 0 : index
        %c4 = arith.constant 4 : index
        %c1_1 = arith.constant 1 : index
        scf.for %i_ab = %c0_1 to %c4 step %c1_1 {
          %buffer_out = aie.objectfifo.acquire @out_L1L2_4_1(Produce, 1) : !aie.objectfifosubview<memref<2x2x8x8xbf16>>
          %out = aie.objectfifo.subview.access %buffer_out[0] : !aie.objectfifosubview<memref<2x2x8x8xbf16>> -> memref<2x2x8x8xbf16>
          func.call @zero(%out) : (memref<2x2x8x8xbf16>) -> ()
          %c0_2 = arith.constant 0 : index
          %c16 = arith.constant 16 : index
          %c1_2 = arith.constant 1 : index
          scf.for %i_c = %c0_2 to %c16 step %c1_2 {
            %buffer_in0 = aie.objectfifo.acquire @in0_L2L1_4(Consume, 1) : !aie.objectfifosubview<memref<2x8x8x8xbf16>>
            %in0 = aie.objectfifo.subview.access %buffer_in0[0] : !aie.objectfifosubview<memref<2x8x8x8xbf16>> -> memref<2x8x8x8xbf16>
            %buffer_in1 = aie.objectfifo.acquire @in1_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<8x2x8x8xbf16>>
            %in1 = aie.objectfifo.subview.access %buffer_in1[0] : !aie.objectfifosubview<memref<8x2x8x8xbf16>> -> memref<8x2x8x8xbf16>
            func.call @matmul(%in0, %in1, %out) : (memref<2x8x8x8xbf16>, memref<8x2x8x8xbf16>, memref<2x2x8x8xbf16>) -> ()
            aie.objectfifo.release @in0_L2L1_4(Consume, 1)
            aie.objectfifo.release @in1_L2L1_1(Consume, 1)
          }
          aie.objectfifo.release @out_L1L2_4_1(Produce, 1)
        }
      }
      aie.end
    } {stack_size = 1024 : i32}
    // col=4, row_idx=2  (tile_4_4)
    %core_4_4 = aie.core(%tile_4_4) {
      %c0 = arith.constant 0 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c4294967295 step %c1 {
        %c0_1 = arith.constant 0 : index
        %c4 = arith.constant 4 : index
        %c1_1 = arith.constant 1 : index
        scf.for %i_ab = %c0_1 to %c4 step %c1_1 {
          %buffer_out = aie.objectfifo.acquire @out_L1L2_4_2(Produce, 1) : !aie.objectfifosubview<memref<2x2x8x8xbf16>>
          %out = aie.objectfifo.subview.access %buffer_out[0] : !aie.objectfifosubview<memref<2x2x8x8xbf16>> -> memref<2x2x8x8xbf16>
          func.call @zero(%out) : (memref<2x2x8x8xbf16>) -> ()
          %c0_2 = arith.constant 0 : index
          %c16 = arith.constant 16 : index
          %c1_2 = arith.constant 1 : index
          scf.for %i_c = %c0_2 to %c16 step %c1_2 {
            %buffer_in0 = aie.objectfifo.acquire @in0_L2L1_4(Consume, 1) : !aie.objectfifosubview<memref<2x8x8x8xbf16>>
            %in0 = aie.objectfifo.subview.access %buffer_in0[0] : !aie.objectfifosubview<memref<2x8x8x8xbf16>> -> memref<2x8x8x8xbf16>
            %buffer_in1 = aie.objectfifo.acquire @in1_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<8x2x8x8xbf16>>
            %in1 = aie.objectfifo.subview.access %buffer_in1[0] : !aie.objectfifosubview<memref<8x2x8x8xbf16>> -> memref<8x2x8x8xbf16>
            func.call @matmul(%in0, %in1, %out) : (memref<2x8x8x8xbf16>, memref<8x2x8x8xbf16>, memref<2x2x8x8xbf16>) -> ()
            aie.objectfifo.release @in0_L2L1_4(Consume, 1)
            aie.objectfifo.release @in1_L2L1_2(Consume, 1)
          }
          aie.objectfifo.release @out_L1L2_4_2(Produce, 1)
        }
      }
      aie.end
    } {stack_size = 1024 : i32}
    // col=4, row_idx=3  (tile_4_5)
    %core_4_5 = aie.core(%tile_4_5) {
      %c0 = arith.constant 0 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c4294967295 step %c1 {
        %c0_1 = arith.constant 0 : index
        %c4 = arith.constant 4 : index
        %c1_1 = arith.constant 1 : index
        scf.for %i_ab = %c0_1 to %c4 step %c1_1 {
          %buffer_out = aie.objectfifo.acquire @out_L1L2_4_3(Produce, 1) : !aie.objectfifosubview<memref<2x2x8x8xbf16>>
          %out = aie.objectfifo.subview.access %buffer_out[0] : !aie.objectfifosubview<memref<2x2x8x8xbf16>> -> memref<2x2x8x8xbf16>
          func.call @zero(%out) : (memref<2x2x8x8xbf16>) -> ()
          %c0_2 = arith.constant 0 : index
          %c16 = arith.constant 16 : index
          %c1_2 = arith.constant 1 : index
          scf.for %i_c = %c0_2 to %c16 step %c1_2 {
            %buffer_in0 = aie.objectfifo.acquire @in0_L2L1_4(Consume, 1) : !aie.objectfifosubview<memref<2x8x8x8xbf16>>
            %in0 = aie.objectfifo.subview.access %buffer_in0[0] : !aie.objectfifosubview<memref<2x8x8x8xbf16>> -> memref<2x8x8x8xbf16>
            %buffer_in1 = aie.objectfifo.acquire @in1_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<8x2x8x8xbf16>>
            %in1 = aie.objectfifo.subview.access %buffer_in1[0] : !aie.objectfifosubview<memref<8x2x8x8xbf16>> -> memref<8x2x8x8xbf16>
            func.call @matmul(%in0, %in1, %out) : (memref<2x8x8x8xbf16>, memref<8x2x8x8xbf16>, memref<2x2x8x8xbf16>) -> ()
            aie.objectfifo.release @in0_L2L1_4(Consume, 1)
            aie.objectfifo.release @in1_L2L1_3(Consume, 1)
          }
          aie.objectfifo.release @out_L1L2_4_3(Produce, 1)
        }
      }
      aie.end
    } {stack_size = 1024 : i32}
    // col=5, row_idx=0  (tile_5_2)
    %core_5_2 = aie.core(%tile_5_2) {
      %c0 = arith.constant 0 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c4294967295 step %c1 {
        %c0_1 = arith.constant 0 : index
        %c4 = arith.constant 4 : index
        %c1_1 = arith.constant 1 : index
        scf.for %i_ab = %c0_1 to %c4 step %c1_1 {
          %buffer_out = aie.objectfifo.acquire @out_L1L2_5_0(Produce, 1) : !aie.objectfifosubview<memref<2x2x8x8xbf16>>
          %out = aie.objectfifo.subview.access %buffer_out[0] : !aie.objectfifosubview<memref<2x2x8x8xbf16>> -> memref<2x2x8x8xbf16>
          func.call @zero(%out) : (memref<2x2x8x8xbf16>) -> ()
          %c0_2 = arith.constant 0 : index
          %c16 = arith.constant 16 : index
          %c1_2 = arith.constant 1 : index
          scf.for %i_c = %c0_2 to %c16 step %c1_2 {
            %buffer_in0 = aie.objectfifo.acquire @in0_L2L1_5(Consume, 1) : !aie.objectfifosubview<memref<2x8x8x8xbf16>>
            %in0 = aie.objectfifo.subview.access %buffer_in0[0] : !aie.objectfifosubview<memref<2x8x8x8xbf16>> -> memref<2x8x8x8xbf16>
            %buffer_in1 = aie.objectfifo.acquire @in1_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<8x2x8x8xbf16>>
            %in1 = aie.objectfifo.subview.access %buffer_in1[0] : !aie.objectfifosubview<memref<8x2x8x8xbf16>> -> memref<8x2x8x8xbf16>
            func.call @matmul(%in0, %in1, %out) : (memref<2x8x8x8xbf16>, memref<8x2x8x8xbf16>, memref<2x2x8x8xbf16>) -> ()
            aie.objectfifo.release @in0_L2L1_5(Consume, 1)
            aie.objectfifo.release @in1_L2L1_0(Consume, 1)
          }
          aie.objectfifo.release @out_L1L2_5_0(Produce, 1)
        }
      }
      aie.end
    } {stack_size = 1024 : i32}
    // col=5, row_idx=1  (tile_5_3)
    %core_5_3 = aie.core(%tile_5_3) {
      %c0 = arith.constant 0 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c4294967295 step %c1 {
        %c0_1 = arith.constant 0 : index
        %c4 = arith.constant 4 : index
        %c1_1 = arith.constant 1 : index
        scf.for %i_ab = %c0_1 to %c4 step %c1_1 {
          %buffer_out = aie.objectfifo.acquire @out_L1L2_5_1(Produce, 1) : !aie.objectfifosubview<memref<2x2x8x8xbf16>>
          %out = aie.objectfifo.subview.access %buffer_out[0] : !aie.objectfifosubview<memref<2x2x8x8xbf16>> -> memref<2x2x8x8xbf16>
          func.call @zero(%out) : (memref<2x2x8x8xbf16>) -> ()
          %c0_2 = arith.constant 0 : index
          %c16 = arith.constant 16 : index
          %c1_2 = arith.constant 1 : index
          scf.for %i_c = %c0_2 to %c16 step %c1_2 {
            %buffer_in0 = aie.objectfifo.acquire @in0_L2L1_5(Consume, 1) : !aie.objectfifosubview<memref<2x8x8x8xbf16>>
            %in0 = aie.objectfifo.subview.access %buffer_in0[0] : !aie.objectfifosubview<memref<2x8x8x8xbf16>> -> memref<2x8x8x8xbf16>
            %buffer_in1 = aie.objectfifo.acquire @in1_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<8x2x8x8xbf16>>
            %in1 = aie.objectfifo.subview.access %buffer_in1[0] : !aie.objectfifosubview<memref<8x2x8x8xbf16>> -> memref<8x2x8x8xbf16>
            func.call @matmul(%in0, %in1, %out) : (memref<2x8x8x8xbf16>, memref<8x2x8x8xbf16>, memref<2x2x8x8xbf16>) -> ()
            aie.objectfifo.release @in0_L2L1_5(Consume, 1)
            aie.objectfifo.release @in1_L2L1_1(Consume, 1)
          }
          aie.objectfifo.release @out_L1L2_5_1(Produce, 1)
        }
      }
      aie.end
    } {stack_size = 1024 : i32}
    // col=5, row_idx=2  (tile_5_4)
    %core_5_4 = aie.core(%tile_5_4) {
      %c0 = arith.constant 0 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c4294967295 step %c1 {
        %c0_1 = arith.constant 0 : index
        %c4 = arith.constant 4 : index
        %c1_1 = arith.constant 1 : index
        scf.for %i_ab = %c0_1 to %c4 step %c1_1 {
          %buffer_out = aie.objectfifo.acquire @out_L1L2_5_2(Produce, 1) : !aie.objectfifosubview<memref<2x2x8x8xbf16>>
          %out = aie.objectfifo.subview.access %buffer_out[0] : !aie.objectfifosubview<memref<2x2x8x8xbf16>> -> memref<2x2x8x8xbf16>
          func.call @zero(%out) : (memref<2x2x8x8xbf16>) -> ()
          %c0_2 = arith.constant 0 : index
          %c16 = arith.constant 16 : index
          %c1_2 = arith.constant 1 : index
          scf.for %i_c = %c0_2 to %c16 step %c1_2 {
            %buffer_in0 = aie.objectfifo.acquire @in0_L2L1_5(Consume, 1) : !aie.objectfifosubview<memref<2x8x8x8xbf16>>
            %in0 = aie.objectfifo.subview.access %buffer_in0[0] : !aie.objectfifosubview<memref<2x8x8x8xbf16>> -> memref<2x8x8x8xbf16>
            %buffer_in1 = aie.objectfifo.acquire @in1_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<8x2x8x8xbf16>>
            %in1 = aie.objectfifo.subview.access %buffer_in1[0] : !aie.objectfifosubview<memref<8x2x8x8xbf16>> -> memref<8x2x8x8xbf16>
            func.call @matmul(%in0, %in1, %out) : (memref<2x8x8x8xbf16>, memref<8x2x8x8xbf16>, memref<2x2x8x8xbf16>) -> ()
            aie.objectfifo.release @in0_L2L1_5(Consume, 1)
            aie.objectfifo.release @in1_L2L1_2(Consume, 1)
          }
          aie.objectfifo.release @out_L1L2_5_2(Produce, 1)
        }
      }
      aie.end
    } {stack_size = 1024 : i32}
    // col=5, row_idx=3  (tile_5_5)
    %core_5_5 = aie.core(%tile_5_5) {
      %c0 = arith.constant 0 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c4294967295 step %c1 {
        %c0_1 = arith.constant 0 : index
        %c4 = arith.constant 4 : index
        %c1_1 = arith.constant 1 : index
        scf.for %i_ab = %c0_1 to %c4 step %c1_1 {
          %buffer_out = aie.objectfifo.acquire @out_L1L2_5_3(Produce, 1) : !aie.objectfifosubview<memref<2x2x8x8xbf16>>
          %out = aie.objectfifo.subview.access %buffer_out[0] : !aie.objectfifosubview<memref<2x2x8x8xbf16>> -> memref<2x2x8x8xbf16>
          func.call @zero(%out) : (memref<2x2x8x8xbf16>) -> ()
          %c0_2 = arith.constant 0 : index
          %c16 = arith.constant 16 : index
          %c1_2 = arith.constant 1 : index
          scf.for %i_c = %c0_2 to %c16 step %c1_2 {
            %buffer_in0 = aie.objectfifo.acquire @in0_L2L1_5(Consume, 1) : !aie.objectfifosubview<memref<2x8x8x8xbf16>>
            %in0 = aie.objectfifo.subview.access %buffer_in0[0] : !aie.objectfifosubview<memref<2x8x8x8xbf16>> -> memref<2x8x8x8xbf16>
            %buffer_in1 = aie.objectfifo.acquire @in1_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<8x2x8x8xbf16>>
            %in1 = aie.objectfifo.subview.access %buffer_in1[0] : !aie.objectfifosubview<memref<8x2x8x8xbf16>> -> memref<8x2x8x8xbf16>
            func.call @matmul(%in0, %in1, %out) : (memref<2x8x8x8xbf16>, memref<8x2x8x8xbf16>, memref<2x2x8x8xbf16>) -> ()
            aie.objectfifo.release @in0_L2L1_5(Consume, 1)
            aie.objectfifo.release @in1_L2L1_3(Consume, 1)
          }
          aie.objectfifo.release @out_L1L2_5_3(Produce, 1)
        }
      }
      aie.end
    } {stack_size = 1024 : i32}
    // col=6, row_idx=0  (tile_6_2)
    %core_6_2 = aie.core(%tile_6_2) {
      %c0 = arith.constant 0 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c4294967295 step %c1 {
        %c0_1 = arith.constant 0 : index
        %c4 = arith.constant 4 : index
        %c1_1 = arith.constant 1 : index
        scf.for %i_ab = %c0_1 to %c4 step %c1_1 {
          %buffer_out = aie.objectfifo.acquire @out_L1L2_6_0(Produce, 1) : !aie.objectfifosubview<memref<2x2x8x8xbf16>>
          %out = aie.objectfifo.subview.access %buffer_out[0] : !aie.objectfifosubview<memref<2x2x8x8xbf16>> -> memref<2x2x8x8xbf16>
          func.call @zero(%out) : (memref<2x2x8x8xbf16>) -> ()
          %c0_2 = arith.constant 0 : index
          %c16 = arith.constant 16 : index
          %c1_2 = arith.constant 1 : index
          scf.for %i_c = %c0_2 to %c16 step %c1_2 {
            %buffer_in0 = aie.objectfifo.acquire @in0_L2L1_6(Consume, 1) : !aie.objectfifosubview<memref<2x8x8x8xbf16>>
            %in0 = aie.objectfifo.subview.access %buffer_in0[0] : !aie.objectfifosubview<memref<2x8x8x8xbf16>> -> memref<2x8x8x8xbf16>
            %buffer_in1 = aie.objectfifo.acquire @in1_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<8x2x8x8xbf16>>
            %in1 = aie.objectfifo.subview.access %buffer_in1[0] : !aie.objectfifosubview<memref<8x2x8x8xbf16>> -> memref<8x2x8x8xbf16>
            func.call @matmul(%in0, %in1, %out) : (memref<2x8x8x8xbf16>, memref<8x2x8x8xbf16>, memref<2x2x8x8xbf16>) -> ()
            aie.objectfifo.release @in0_L2L1_6(Consume, 1)
            aie.objectfifo.release @in1_L2L1_0(Consume, 1)
          }
          aie.objectfifo.release @out_L1L2_6_0(Produce, 1)
        }
      }
      aie.end
    } {stack_size = 1024 : i32}
    // col=6, row_idx=1  (tile_6_3)
    %core_6_3 = aie.core(%tile_6_3) {
      %c0 = arith.constant 0 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c4294967295 step %c1 {
        %c0_1 = arith.constant 0 : index
        %c4 = arith.constant 4 : index
        %c1_1 = arith.constant 1 : index
        scf.for %i_ab = %c0_1 to %c4 step %c1_1 {
          %buffer_out = aie.objectfifo.acquire @out_L1L2_6_1(Produce, 1) : !aie.objectfifosubview<memref<2x2x8x8xbf16>>
          %out = aie.objectfifo.subview.access %buffer_out[0] : !aie.objectfifosubview<memref<2x2x8x8xbf16>> -> memref<2x2x8x8xbf16>
          func.call @zero(%out) : (memref<2x2x8x8xbf16>) -> ()
          %c0_2 = arith.constant 0 : index
          %c16 = arith.constant 16 : index
          %c1_2 = arith.constant 1 : index
          scf.for %i_c = %c0_2 to %c16 step %c1_2 {
            %buffer_in0 = aie.objectfifo.acquire @in0_L2L1_6(Consume, 1) : !aie.objectfifosubview<memref<2x8x8x8xbf16>>
            %in0 = aie.objectfifo.subview.access %buffer_in0[0] : !aie.objectfifosubview<memref<2x8x8x8xbf16>> -> memref<2x8x8x8xbf16>
            %buffer_in1 = aie.objectfifo.acquire @in1_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<8x2x8x8xbf16>>
            %in1 = aie.objectfifo.subview.access %buffer_in1[0] : !aie.objectfifosubview<memref<8x2x8x8xbf16>> -> memref<8x2x8x8xbf16>
            func.call @matmul(%in0, %in1, %out) : (memref<2x8x8x8xbf16>, memref<8x2x8x8xbf16>, memref<2x2x8x8xbf16>) -> ()
            aie.objectfifo.release @in0_L2L1_6(Consume, 1)
            aie.objectfifo.release @in1_L2L1_1(Consume, 1)
          }
          aie.objectfifo.release @out_L1L2_6_1(Produce, 1)
        }
      }
      aie.end
    } {stack_size = 1024 : i32}
    // col=6, row_idx=2  (tile_6_4)
    %core_6_4 = aie.core(%tile_6_4) {
      %c0 = arith.constant 0 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c4294967295 step %c1 {
        %c0_1 = arith.constant 0 : index
        %c4 = arith.constant 4 : index
        %c1_1 = arith.constant 1 : index
        scf.for %i_ab = %c0_1 to %c4 step %c1_1 {
          %buffer_out = aie.objectfifo.acquire @out_L1L2_6_2(Produce, 1) : !aie.objectfifosubview<memref<2x2x8x8xbf16>>
          %out = aie.objectfifo.subview.access %buffer_out[0] : !aie.objectfifosubview<memref<2x2x8x8xbf16>> -> memref<2x2x8x8xbf16>
          func.call @zero(%out) : (memref<2x2x8x8xbf16>) -> ()
          %c0_2 = arith.constant 0 : index
          %c16 = arith.constant 16 : index
          %c1_2 = arith.constant 1 : index
          scf.for %i_c = %c0_2 to %c16 step %c1_2 {
            %buffer_in0 = aie.objectfifo.acquire @in0_L2L1_6(Consume, 1) : !aie.objectfifosubview<memref<2x8x8x8xbf16>>
            %in0 = aie.objectfifo.subview.access %buffer_in0[0] : !aie.objectfifosubview<memref<2x8x8x8xbf16>> -> memref<2x8x8x8xbf16>
            %buffer_in1 = aie.objectfifo.acquire @in1_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<8x2x8x8xbf16>>
            %in1 = aie.objectfifo.subview.access %buffer_in1[0] : !aie.objectfifosubview<memref<8x2x8x8xbf16>> -> memref<8x2x8x8xbf16>
            func.call @matmul(%in0, %in1, %out) : (memref<2x8x8x8xbf16>, memref<8x2x8x8xbf16>, memref<2x2x8x8xbf16>) -> ()
            aie.objectfifo.release @in0_L2L1_6(Consume, 1)
            aie.objectfifo.release @in1_L2L1_2(Consume, 1)
          }
          aie.objectfifo.release @out_L1L2_6_2(Produce, 1)
        }
      }
      aie.end
    } {stack_size = 1024 : i32}
    // col=6, row_idx=3  (tile_6_5)
    %core_6_5 = aie.core(%tile_6_5) {
      %c0 = arith.constant 0 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c4294967295 step %c1 {
        %c0_1 = arith.constant 0 : index
        %c4 = arith.constant 4 : index
        %c1_1 = arith.constant 1 : index
        scf.for %i_ab = %c0_1 to %c4 step %c1_1 {
          %buffer_out = aie.objectfifo.acquire @out_L1L2_6_3(Produce, 1) : !aie.objectfifosubview<memref<2x2x8x8xbf16>>
          %out = aie.objectfifo.subview.access %buffer_out[0] : !aie.objectfifosubview<memref<2x2x8x8xbf16>> -> memref<2x2x8x8xbf16>
          func.call @zero(%out) : (memref<2x2x8x8xbf16>) -> ()
          %c0_2 = arith.constant 0 : index
          %c16 = arith.constant 16 : index
          %c1_2 = arith.constant 1 : index
          scf.for %i_c = %c0_2 to %c16 step %c1_2 {
            %buffer_in0 = aie.objectfifo.acquire @in0_L2L1_6(Consume, 1) : !aie.objectfifosubview<memref<2x8x8x8xbf16>>
            %in0 = aie.objectfifo.subview.access %buffer_in0[0] : !aie.objectfifosubview<memref<2x8x8x8xbf16>> -> memref<2x8x8x8xbf16>
            %buffer_in1 = aie.objectfifo.acquire @in1_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<8x2x8x8xbf16>>
            %in1 = aie.objectfifo.subview.access %buffer_in1[0] : !aie.objectfifosubview<memref<8x2x8x8xbf16>> -> memref<8x2x8x8xbf16>
            func.call @matmul(%in0, %in1, %out) : (memref<2x8x8x8xbf16>, memref<8x2x8x8xbf16>, memref<2x2x8x8xbf16>) -> ()
            aie.objectfifo.release @in0_L2L1_6(Consume, 1)
            aie.objectfifo.release @in1_L2L1_3(Consume, 1)
          }
          aie.objectfifo.release @out_L1L2_6_3(Produce, 1)
        }
      }
      aie.end
    } {stack_size = 1024 : i32}
    // col=7, row_idx=0  (tile_7_2)
    %core_7_2 = aie.core(%tile_7_2) {
      %c0 = arith.constant 0 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c4294967295 step %c1 {
        %c0_1 = arith.constant 0 : index
        %c4 = arith.constant 4 : index
        %c1_1 = arith.constant 1 : index
        scf.for %i_ab = %c0_1 to %c4 step %c1_1 {
          %buffer_out = aie.objectfifo.acquire @out_L1L2_7_0(Produce, 1) : !aie.objectfifosubview<memref<2x2x8x8xbf16>>
          %out = aie.objectfifo.subview.access %buffer_out[0] : !aie.objectfifosubview<memref<2x2x8x8xbf16>> -> memref<2x2x8x8xbf16>
          func.call @zero(%out) : (memref<2x2x8x8xbf16>) -> ()
          %c0_2 = arith.constant 0 : index
          %c16 = arith.constant 16 : index
          %c1_2 = arith.constant 1 : index
          scf.for %i_c = %c0_2 to %c16 step %c1_2 {
            %buffer_in0 = aie.objectfifo.acquire @in0_L2L1_7(Consume, 1) : !aie.objectfifosubview<memref<2x8x8x8xbf16>>
            %in0 = aie.objectfifo.subview.access %buffer_in0[0] : !aie.objectfifosubview<memref<2x8x8x8xbf16>> -> memref<2x8x8x8xbf16>
            %buffer_in1 = aie.objectfifo.acquire @in1_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<8x2x8x8xbf16>>
            %in1 = aie.objectfifo.subview.access %buffer_in1[0] : !aie.objectfifosubview<memref<8x2x8x8xbf16>> -> memref<8x2x8x8xbf16>
            func.call @matmul(%in0, %in1, %out) : (memref<2x8x8x8xbf16>, memref<8x2x8x8xbf16>, memref<2x2x8x8xbf16>) -> ()
            aie.objectfifo.release @in0_L2L1_7(Consume, 1)
            aie.objectfifo.release @in1_L2L1_0(Consume, 1)
          }
          aie.objectfifo.release @out_L1L2_7_0(Produce, 1)
        }
      }
      aie.end
    } {stack_size = 1024 : i32}
    // col=7, row_idx=1  (tile_7_3)
    %core_7_3 = aie.core(%tile_7_3) {
      %c0 = arith.constant 0 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c4294967295 step %c1 {
        %c0_1 = arith.constant 0 : index
        %c4 = arith.constant 4 : index
        %c1_1 = arith.constant 1 : index
        scf.for %i_ab = %c0_1 to %c4 step %c1_1 {
          %buffer_out = aie.objectfifo.acquire @out_L1L2_7_1(Produce, 1) : !aie.objectfifosubview<memref<2x2x8x8xbf16>>
          %out = aie.objectfifo.subview.access %buffer_out[0] : !aie.objectfifosubview<memref<2x2x8x8xbf16>> -> memref<2x2x8x8xbf16>
          func.call @zero(%out) : (memref<2x2x8x8xbf16>) -> ()
          %c0_2 = arith.constant 0 : index
          %c16 = arith.constant 16 : index
          %c1_2 = arith.constant 1 : index
          scf.for %i_c = %c0_2 to %c16 step %c1_2 {
            %buffer_in0 = aie.objectfifo.acquire @in0_L2L1_7(Consume, 1) : !aie.objectfifosubview<memref<2x8x8x8xbf16>>
            %in0 = aie.objectfifo.subview.access %buffer_in0[0] : !aie.objectfifosubview<memref<2x8x8x8xbf16>> -> memref<2x8x8x8xbf16>
            %buffer_in1 = aie.objectfifo.acquire @in1_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<8x2x8x8xbf16>>
            %in1 = aie.objectfifo.subview.access %buffer_in1[0] : !aie.objectfifosubview<memref<8x2x8x8xbf16>> -> memref<8x2x8x8xbf16>
            func.call @matmul(%in0, %in1, %out) : (memref<2x8x8x8xbf16>, memref<8x2x8x8xbf16>, memref<2x2x8x8xbf16>) -> ()
            aie.objectfifo.release @in0_L2L1_7(Consume, 1)
            aie.objectfifo.release @in1_L2L1_1(Consume, 1)
          }
          aie.objectfifo.release @out_L1L2_7_1(Produce, 1)
        }
      }
      aie.end
    } {stack_size = 1024 : i32}
    // col=7, row_idx=2  (tile_7_4)
    %core_7_4 = aie.core(%tile_7_4) {
      %c0 = arith.constant 0 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c4294967295 step %c1 {
        %c0_1 = arith.constant 0 : index
        %c4 = arith.constant 4 : index
        %c1_1 = arith.constant 1 : index
        scf.for %i_ab = %c0_1 to %c4 step %c1_1 {
          %buffer_out = aie.objectfifo.acquire @out_L1L2_7_2(Produce, 1) : !aie.objectfifosubview<memref<2x2x8x8xbf16>>
          %out = aie.objectfifo.subview.access %buffer_out[0] : !aie.objectfifosubview<memref<2x2x8x8xbf16>> -> memref<2x2x8x8xbf16>
          func.call @zero(%out) : (memref<2x2x8x8xbf16>) -> ()
          %c0_2 = arith.constant 0 : index
          %c16 = arith.constant 16 : index
          %c1_2 = arith.constant 1 : index
          scf.for %i_c = %c0_2 to %c16 step %c1_2 {
            %buffer_in0 = aie.objectfifo.acquire @in0_L2L1_7(Consume, 1) : !aie.objectfifosubview<memref<2x8x8x8xbf16>>
            %in0 = aie.objectfifo.subview.access %buffer_in0[0] : !aie.objectfifosubview<memref<2x8x8x8xbf16>> -> memref<2x8x8x8xbf16>
            %buffer_in1 = aie.objectfifo.acquire @in1_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<8x2x8x8xbf16>>
            %in1 = aie.objectfifo.subview.access %buffer_in1[0] : !aie.objectfifosubview<memref<8x2x8x8xbf16>> -> memref<8x2x8x8xbf16>
            func.call @matmul(%in0, %in1, %out) : (memref<2x8x8x8xbf16>, memref<8x2x8x8xbf16>, memref<2x2x8x8xbf16>) -> ()
            aie.objectfifo.release @in0_L2L1_7(Consume, 1)
            aie.objectfifo.release @in1_L2L1_2(Consume, 1)
          }
          aie.objectfifo.release @out_L1L2_7_2(Produce, 1)
        }
      }
      aie.end
    } {stack_size = 1024 : i32}
    // col=7, row_idx=3  (tile_7_5)
    %core_7_5 = aie.core(%tile_7_5) {
      %c0 = arith.constant 0 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c4294967295 step %c1 {
        %c0_1 = arith.constant 0 : index
        %c4 = arith.constant 4 : index
        %c1_1 = arith.constant 1 : index
        scf.for %i_ab = %c0_1 to %c4 step %c1_1 {
          %buffer_out = aie.objectfifo.acquire @out_L1L2_7_3(Produce, 1) : !aie.objectfifosubview<memref<2x2x8x8xbf16>>
          %out = aie.objectfifo.subview.access %buffer_out[0] : !aie.objectfifosubview<memref<2x2x8x8xbf16>> -> memref<2x2x8x8xbf16>
          func.call @zero(%out) : (memref<2x2x8x8xbf16>) -> ()
          %c0_2 = arith.constant 0 : index
          %c16 = arith.constant 16 : index
          %c1_2 = arith.constant 1 : index
          scf.for %i_c = %c0_2 to %c16 step %c1_2 {
            %buffer_in0 = aie.objectfifo.acquire @in0_L2L1_7(Consume, 1) : !aie.objectfifosubview<memref<2x8x8x8xbf16>>
            %in0 = aie.objectfifo.subview.access %buffer_in0[0] : !aie.objectfifosubview<memref<2x8x8x8xbf16>> -> memref<2x8x8x8xbf16>
            %buffer_in1 = aie.objectfifo.acquire @in1_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<8x2x8x8xbf16>>
            %in1 = aie.objectfifo.subview.access %buffer_in1[0] : !aie.objectfifosubview<memref<8x2x8x8xbf16>> -> memref<8x2x8x8xbf16>
            func.call @matmul(%in0, %in1, %out) : (memref<2x8x8x8xbf16>, memref<8x2x8x8xbf16>, memref<2x2x8x8xbf16>) -> ()
            aie.objectfifo.release @in0_L2L1_7(Consume, 1)
            aie.objectfifo.release @in1_L2L1_3(Consume, 1)
          }
          aie.objectfifo.release @out_L1L2_7_3(Produce, 1)
        }
      }
      aie.end
    } {stack_size = 1024 : i32}
    aie.runtime_sequence(%arg0: memref<256x1024xbf16>, %arg1: memref<1024x128xbf16>, %arg2: memref<256x128xbf16>) {
      // Iteration structure: a=2 (M-blocks) x b=2 (N-blocks).
      // in0_L3L2_<col>: col selects M-rows, offset = a_i*131072 + col*16384
      // in1_L3L2_<row>: row selects N-cols, offset[3] = b_i*64 + row*16
      // out_L2L3_<col>: output arrives as y*(p*m)*(q*n)=4*16*16 in ypmqn order
      //  DMA: [1, 4, 16, 16][0, 16, 128, 1] at row (a_i*128+col*16), col (b_i*64)

      // a=0, b=0: send output prefetch for all cols, in0 all cols, in1 rows 
      aiex.npu.dma_memcpy_nd(%arg2[0, 0, 0, 0][1, 4, 16, 16][0, 16, 128, 1]) {id = 0 : i64, metadata = @out_L2L3_0} : memref<256x128xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[0, 0, 16, 0][1, 4, 16, 16][0, 16, 128, 1]) {id = 0 : i64, metadata = @out_L2L3_1} : memref<256x128xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[0, 0, 32, 0][1, 4, 16, 16][0, 16, 128, 1]) {id = 0 : i64, metadata = @out_L2L3_2} : memref<256x128xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[0, 0, 48, 0][1, 4, 16, 16][0, 16, 128, 1]) {id = 0 : i64, metadata = @out_L2L3_3} : memref<256x128xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[0, 0, 64, 0][1, 4, 16, 16][0, 16, 128, 1]) {id = 0 : i64, metadata = @out_L2L3_4} : memref<256x128xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[0, 0, 80, 0][1, 4, 16, 16][0, 16, 128, 1]) {id = 0 : i64, metadata = @out_L2L3_5} : memref<256x128xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[0, 0, 96, 0][1, 4, 16, 16][0, 16, 128, 1]) {id = 0 : i64, metadata = @out_L2L3_6} : memref<256x128xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[0, 0, 112, 0][1, 4, 16, 16][0, 16, 128, 1]) {id = 0 : i64, metadata = @out_L2L3_7} : memref<256x128xbf16>
      // in0: a=0, col 0..7 (offset = col*16384)
      aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 0][1, 16, 16, 64][0, 64, 1024, 1]) {id = 1 : i64, metadata = @in0_L3L2_0} : memref<256x1024xbf16>
      aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 16384][1, 16, 16, 64][0, 64, 1024, 1]) {id = 1 : i64, metadata = @in0_L3L2_1} : memref<256x1024xbf16>
      aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 32768][1, 16, 16, 64][0, 64, 1024, 1]) {id = 1 : i64, metadata = @in0_L3L2_2} : memref<256x1024xbf16>
      aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 49152][1, 16, 16, 64][0, 64, 1024, 1]) {id = 1 : i64, metadata = @in0_L3L2_3} : memref<256x1024xbf16>
      aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 65536][1, 16, 16, 64][0, 64, 1024, 1]) {id = 1 : i64, metadata = @in0_L3L2_4} : memref<256x1024xbf16>
      aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 81920][1, 16, 16, 64][0, 64, 1024, 1]) {id = 1 : i64, metadata = @in0_L3L2_5} : memref<256x1024xbf16>
      aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 98304][1, 16, 16, 64][0, 64, 1024, 1]) {id = 1 : i64, metadata = @in0_L3L2_6} : memref<256x1024xbf16>
      aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 114688][1, 16, 16, 64][0, 64, 1024, 1]) {id = 1 : i64, metadata = @in0_L3L2_7} : memref<256x1024xbf16>
      // in1: b=0, row 0..3 (offset[3] = row*16)
      aiex.npu.dma_memcpy_nd(%arg1[0, 0, 0, 0][1, 16, 64, 16][0, 8192, 128, 1]) {id = 2 : i64, metadata = @in1_L3L2_0} : memref<1024x128xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 0, 0, 16][1, 16, 64, 16][0, 8192, 128, 1]) {id = 2 : i64, metadata = @in1_L3L2_1} : memref<1024x128xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 0, 0, 32][1, 16, 64, 16][0, 8192, 128, 1]) {id = 2 : i64, metadata = @in1_L3L2_2} : memref<1024x128xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 0, 0, 48][1, 16, 64, 16][0, 8192, 128, 1]) {id = 2 : i64, metadata = @in1_L3L2_3} : memref<1024x128xbf16>

      // a=0, b=1 
      aiex.npu.dma_memcpy_nd(%arg2[0, 0, 0, 64][1, 4, 16, 16][0, 16, 128, 1]) {id = 8 : i64, metadata = @out_L2L3_0} : memref<256x128xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[0, 0, 16, 64][1, 4, 16, 16][0, 16, 128, 1]) {id = 8 : i64, metadata = @out_L2L3_1} : memref<256x128xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[0, 0, 32, 64][1, 4, 16, 16][0, 16, 128, 1]) {id = 8 : i64, metadata = @out_L2L3_2} : memref<256x128xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[0, 0, 48, 64][1, 4, 16, 16][0, 16, 128, 1]) {id = 8 : i64, metadata = @out_L2L3_3} : memref<256x128xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[0, 0, 64, 64][1, 4, 16, 16][0, 16, 128, 1]) {id = 8 : i64, metadata = @out_L2L3_4} : memref<256x128xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[0, 0, 80, 64][1, 4, 16, 16][0, 16, 128, 1]) {id = 8 : i64, metadata = @out_L2L3_5} : memref<256x128xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[0, 0, 96, 64][1, 4, 16, 16][0, 16, 128, 1]) {id = 8 : i64, metadata = @out_L2L3_6} : memref<256x128xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[0, 0, 112, 64][1, 4, 16, 16][0, 16, 128, 1]) {id = 8 : i64, metadata = @out_L2L3_7} : memref<256x128xbf16>
      // in0 same a=0 block (double-buffer slot 2)
      aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 0][1, 16, 16, 64][0, 64, 1024, 1]) {id = 9 : i64, metadata = @in0_L3L2_0} : memref<256x1024xbf16>
      aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 16384][1, 16, 16, 64][0, 64, 1024, 1]) {id = 9 : i64, metadata = @in0_L3L2_1} : memref<256x1024xbf16>
      aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 32768][1, 16, 16, 64][0, 64, 1024, 1]) {id = 9 : i64, metadata = @in0_L3L2_2} : memref<256x1024xbf16>
      aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 49152][1, 16, 16, 64][0, 64, 1024, 1]) {id = 9 : i64, metadata = @in0_L3L2_3} : memref<256x1024xbf16>
      aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 65536][1, 16, 16, 64][0, 64, 1024, 1]) {id = 9 : i64, metadata = @in0_L3L2_4} : memref<256x1024xbf16>
      aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 81920][1, 16, 16, 64][0, 64, 1024, 1]) {id = 9 : i64, metadata = @in0_L3L2_5} : memref<256x1024xbf16>
      aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 98304][1, 16, 16, 64][0, 64, 1024, 1]) {id = 9 : i64, metadata = @in0_L3L2_6} : memref<256x1024xbf16>
      aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 114688][1, 16, 16, 64][0, 64, 1024, 1]) {id = 9 : i64, metadata = @in0_L3L2_7} : memref<256x1024xbf16>
      // in1: b=1, row 0..3 (offset[3] = 64 + row*16)
      aiex.npu.dma_memcpy_nd(%arg1[0, 0, 0, 64][1, 16, 64, 16][0, 8192, 128, 1]) {id = 10 : i64, metadata = @in1_L3L2_0} : memref<1024x128xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 0, 0, 80][1, 16, 64, 16][0, 8192, 128, 1]) {id = 10 : i64, metadata = @in1_L3L2_1} : memref<1024x128xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 0, 0, 96][1, 16, 64, 16][0, 8192, 128, 1]) {id = 10 : i64, metadata = @in1_L3L2_2} : memref<1024x128xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 0, 0, 112][1, 16, 64, 16][0, 8192, 128, 1]) {id = 10 : i64, metadata = @in1_L3L2_3} : memref<1024x128xbf16>

      aiex.npu.dma_wait {symbol = @out_L2L3_0}
      aiex.npu.dma_wait {symbol = @out_L2L3_1}
      aiex.npu.dma_wait {symbol = @out_L2L3_2}
      aiex.npu.dma_wait {symbol = @out_L2L3_3}
      aiex.npu.dma_wait {symbol = @out_L2L3_4}
      aiex.npu.dma_wait {symbol = @out_L2L3_5}
      aiex.npu.dma_wait {symbol = @out_L2L3_6}
      aiex.npu.dma_wait {symbol = @out_L2L3_7}

      // a=1, b=0 
      aiex.npu.dma_memcpy_nd(%arg2[0, 0, 128, 0][1, 4, 16, 16][0, 16, 128, 1]) {id = 0 : i64, metadata = @out_L2L3_0} : memref<256x128xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[0, 0, 144, 0][1, 4, 16, 16][0, 16, 128, 1]) {id = 0 : i64, metadata = @out_L2L3_1} : memref<256x128xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[0, 0, 160, 0][1, 4, 16, 16][0, 16, 128, 1]) {id = 0 : i64, metadata = @out_L2L3_2} : memref<256x128xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[0, 0, 176, 0][1, 4, 16, 16][0, 16, 128, 1]) {id = 0 : i64, metadata = @out_L2L3_3} : memref<256x128xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[0, 0, 192, 0][1, 4, 16, 16][0, 16, 128, 1]) {id = 0 : i64, metadata = @out_L2L3_4} : memref<256x128xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[0, 0, 208, 0][1, 4, 16, 16][0, 16, 128, 1]) {id = 0 : i64, metadata = @out_L2L3_5} : memref<256x128xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[0, 0, 224, 0][1, 4, 16, 16][0, 16, 128, 1]) {id = 0 : i64, metadata = @out_L2L3_6} : memref<256x128xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[0, 0, 240, 0][1, 4, 16, 16][0, 16, 128, 1]) {id = 0 : i64, metadata = @out_L2L3_7} : memref<256x128xbf16>
      // in0: a=1, col 0..7 (offset = 131072 + col*16384)
      aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 131072][1, 16, 16, 64][0, 64, 1024, 1]) {id = 1 : i64, metadata = @in0_L3L2_0} : memref<256x1024xbf16>
      aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 147456][1, 16, 16, 64][0, 64, 1024, 1]) {id = 1 : i64, metadata = @in0_L3L2_1} : memref<256x1024xbf16>
      aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 163840][1, 16, 16, 64][0, 64, 1024, 1]) {id = 1 : i64, metadata = @in0_L3L2_2} : memref<256x1024xbf16>
      aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 180224][1, 16, 16, 64][0, 64, 1024, 1]) {id = 1 : i64, metadata = @in0_L3L2_3} : memref<256x1024xbf16>
      aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 196608][1, 16, 16, 64][0, 64, 1024, 1]) {id = 1 : i64, metadata = @in0_L3L2_4} : memref<256x1024xbf16>
      aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 212992][1, 16, 16, 64][0, 64, 1024, 1]) {id = 1 : i64, metadata = @in0_L3L2_5} : memref<256x1024xbf16>
      aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 229376][1, 16, 16, 64][0, 64, 1024, 1]) {id = 1 : i64, metadata = @in0_L3L2_6} : memref<256x1024xbf16>
      aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 245760][1, 16, 16, 64][0, 64, 1024, 1]) {id = 1 : i64, metadata = @in0_L3L2_7} : memref<256x1024xbf16>
      // in1: b=0, row 0..3
      aiex.npu.dma_memcpy_nd(%arg1[0, 0, 0, 0][1, 16, 64, 16][0, 8192, 128, 1]) {id = 2 : i64, metadata = @in1_L3L2_0} : memref<1024x128xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 0, 0, 16][1, 16, 64, 16][0, 8192, 128, 1]) {id = 2 : i64, metadata = @in1_L3L2_1} : memref<1024x128xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 0, 0, 32][1, 16, 64, 16][0, 8192, 128, 1]) {id = 2 : i64, metadata = @in1_L3L2_2} : memref<1024x128xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 0, 0, 48][1, 16, 64, 16][0, 8192, 128, 1]) {id = 2 : i64, metadata = @in1_L3L2_3} : memref<1024x128xbf16>

      aiex.npu.dma_wait {symbol = @out_L2L3_0}
      aiex.npu.dma_wait {symbol = @out_L2L3_1}
      aiex.npu.dma_wait {symbol = @out_L2L3_2}
      aiex.npu.dma_wait {symbol = @out_L2L3_3}
      aiex.npu.dma_wait {symbol = @out_L2L3_4}
      aiex.npu.dma_wait {symbol = @out_L2L3_5}
      aiex.npu.dma_wait {symbol = @out_L2L3_6}
      aiex.npu.dma_wait {symbol = @out_L2L3_7}

      // a=1, b=1 
      aiex.npu.dma_memcpy_nd(%arg2[0, 0, 128, 64][1, 4, 16, 16][0, 16, 128, 1]) {id = 8 : i64, metadata = @out_L2L3_0} : memref<256x128xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[0, 0, 144, 64][1, 4, 16, 16][0, 16, 128, 1]) {id = 8 : i64, metadata = @out_L2L3_1} : memref<256x128xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[0, 0, 160, 64][1, 4, 16, 16][0, 16, 128, 1]) {id = 8 : i64, metadata = @out_L2L3_2} : memref<256x128xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[0, 0, 176, 64][1, 4, 16, 16][0, 16, 128, 1]) {id = 8 : i64, metadata = @out_L2L3_3} : memref<256x128xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[0, 0, 192, 64][1, 4, 16, 16][0, 16, 128, 1]) {id = 8 : i64, metadata = @out_L2L3_4} : memref<256x128xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[0, 0, 208, 64][1, 4, 16, 16][0, 16, 128, 1]) {id = 8 : i64, metadata = @out_L2L3_5} : memref<256x128xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[0, 0, 224, 64][1, 4, 16, 16][0, 16, 128, 1]) {id = 8 : i64, metadata = @out_L2L3_6} : memref<256x128xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[0, 0, 240, 64][1, 4, 16, 16][0, 16, 128, 1]) {id = 8 : i64, metadata = @out_L2L3_7} : memref<256x128xbf16>
      // in0: a=1, col 0..7
      aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 131072][1, 16, 16, 64][0, 64, 1024, 1]) {id = 9 : i64, metadata = @in0_L3L2_0} : memref<256x1024xbf16>
      aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 147456][1, 16, 16, 64][0, 64, 1024, 1]) {id = 9 : i64, metadata = @in0_L3L2_1} : memref<256x1024xbf16>
      aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 163840][1, 16, 16, 64][0, 64, 1024, 1]) {id = 9 : i64, metadata = @in0_L3L2_2} : memref<256x1024xbf16>
      aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 180224][1, 16, 16, 64][0, 64, 1024, 1]) {id = 9 : i64, metadata = @in0_L3L2_3} : memref<256x1024xbf16>
      aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 196608][1, 16, 16, 64][0, 64, 1024, 1]) {id = 9 : i64, metadata = @in0_L3L2_4} : memref<256x1024xbf16>
      aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 212992][1, 16, 16, 64][0, 64, 1024, 1]) {id = 9 : i64, metadata = @in0_L3L2_5} : memref<256x1024xbf16>
      aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 229376][1, 16, 16, 64][0, 64, 1024, 1]) {id = 9 : i64, metadata = @in0_L3L2_6} : memref<256x1024xbf16>
      aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 245760][1, 16, 16, 64][0, 64, 1024, 1]) {id = 9 : i64, metadata = @in0_L3L2_7} : memref<256x1024xbf16>
      // in1: b=1, row 0..3
      aiex.npu.dma_memcpy_nd(%arg1[0, 0, 0, 64][1, 16, 64, 16][0, 8192, 128, 1]) {id = 10 : i64, metadata = @in1_L3L2_0} : memref<1024x128xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 0, 0, 80][1, 16, 64, 16][0, 8192, 128, 1]) {id = 10 : i64, metadata = @in1_L3L2_1} : memref<1024x128xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 0, 0, 96][1, 16, 64, 16][0, 8192, 128, 1]) {id = 10 : i64, metadata = @in1_L3L2_2} : memref<1024x128xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 0, 0, 112][1, 16, 64, 16][0, 8192, 128, 1]) {id = 10 : i64, metadata = @in1_L3L2_3} : memref<1024x128xbf16>

      aiex.npu.dma_wait {symbol = @out_L2L3_0}
      aiex.npu.dma_wait {symbol = @out_L2L3_1}
      aiex.npu.dma_wait {symbol = @out_L2L3_2}
      aiex.npu.dma_wait {symbol = @out_L2L3_3}
      aiex.npu.dma_wait {symbol = @out_L2L3_4}
      aiex.npu.dma_wait {symbol = @out_L2L3_5}
      aiex.npu.dma_wait {symbol = @out_L2L3_6}
      aiex.npu.dma_wait {symbol = @out_L2L3_7}
    }
  }
}
