; Reduced from P-256 Scalar::to_bytes in vanity-miner-rs 62f9305.
; Producer: stock Rust 1.93.0 / LLVM 21.1.8, build-metal.py --mode p256-signature.
; Contract: reverse four native i64 limbs into 32 big-endian bytes. Only iterator
; state, loads/stores, and byte swaps remain; this performs no cryptography.
; The companion .rs states the source operation; the captured iterator shape is
; canonical for this regression and must not be regenerated with arbitrary rustc.
target triple = "nvptx64-nvidia-cuda"
target datalayout = "e-p:64:64-i64:64-i128:128"
declare void @llvm.lifetime.start.p0(i64 immarg, ptr nocapture)
declare void @llvm.lifetime.end.p0(i64 immarg, ptr nocapture)
declare i64 @llvm.bswap.i64(i64)
declare void @llvm.memcpy.p0.p0.i64(ptr, ptr, i64, i1 immarg)
define internal void @encode(ptr dead_on_unwind noalias noundef nonnull writable writeonly align 1 captures(none) dereferenceable(32) %0, ptr noalias noundef nonnull readonly align 8 captures(address) dereferenceable(32) %1) unnamed_addr noinline memory(write, argmem: readwrite, inaccessiblemem: readwrite) {
  %3 = alloca [32 x i8], align 1
  call void @llvm.lifetime.start.p0(i64 32, ptr nonnull %3)
  store i8 0, ptr %3, align 1
  %4 = getelementptr inbounds nuw i8, ptr %3, i64 1
  store i8 0, ptr %4, align 1
  %5 = getelementptr inbounds nuw i8, ptr %3, i64 2
  store i8 0, ptr %5, align 1
  %6 = getelementptr inbounds nuw i8, ptr %3, i64 3
  store i8 0, ptr %6, align 1
  %7 = getelementptr inbounds nuw i8, ptr %3, i64 4
  store i8 0, ptr %7, align 1
  %8 = getelementptr inbounds nuw i8, ptr %3, i64 5
  store i8 0, ptr %8, align 1
  %9 = getelementptr inbounds nuw i8, ptr %3, i64 6
  store i8 0, ptr %9, align 1
  %10 = getelementptr inbounds nuw i8, ptr %3, i64 7
  store i8 0, ptr %10, align 1
  %11 = getelementptr inbounds nuw i8, ptr %3, i64 8
  store i8 0, ptr %11, align 1
  %12 = getelementptr inbounds nuw i8, ptr %3, i64 9
  store i8 0, ptr %12, align 1
  %13 = getelementptr inbounds nuw i8, ptr %3, i64 10
  store i8 0, ptr %13, align 1
  %14 = getelementptr inbounds nuw i8, ptr %3, i64 11
  store i8 0, ptr %14, align 1
  %15 = getelementptr inbounds nuw i8, ptr %3, i64 12
  store i8 0, ptr %15, align 1
  %16 = getelementptr inbounds nuw i8, ptr %3, i64 13
  store i8 0, ptr %16, align 1
  %17 = getelementptr inbounds nuw i8, ptr %3, i64 14
  store i8 0, ptr %17, align 1
  %18 = getelementptr inbounds nuw i8, ptr %3, i64 15
  store i8 0, ptr %18, align 1
  %19 = getelementptr inbounds nuw i8, ptr %3, i64 16
  store i8 0, ptr %19, align 1
  %20 = getelementptr inbounds nuw i8, ptr %3, i64 17
  store i8 0, ptr %20, align 1
  %21 = getelementptr inbounds nuw i8, ptr %3, i64 18
  store i8 0, ptr %21, align 1
  %22 = getelementptr inbounds nuw i8, ptr %3, i64 19
  store i8 0, ptr %22, align 1
  %23 = getelementptr inbounds nuw i8, ptr %3, i64 20
  store i8 0, ptr %23, align 1
  %24 = getelementptr inbounds nuw i8, ptr %3, i64 21
  store i8 0, ptr %24, align 1
  %25 = getelementptr inbounds nuw i8, ptr %3, i64 22
  store i8 0, ptr %25, align 1
  %26 = getelementptr inbounds nuw i8, ptr %3, i64 23
  store i8 0, ptr %26, align 1
  %27 = getelementptr inbounds nuw i8, ptr %3, i64 24
  store i8 0, ptr %27, align 1
  %28 = getelementptr inbounds nuw i8, ptr %3, i64 25
  store i8 0, ptr %28, align 1
  %29 = getelementptr inbounds nuw i8, ptr %3, i64 26
  store i8 0, ptr %29, align 1
  %30 = getelementptr inbounds nuw i8, ptr %3, i64 27
  store i8 0, ptr %30, align 1
  %31 = getelementptr inbounds nuw i8, ptr %3, i64 28
  store i8 0, ptr %31, align 1
  %32 = getelementptr inbounds nuw i8, ptr %3, i64 29
  store i8 0, ptr %32, align 1
  %33 = getelementptr inbounds nuw i8, ptr %3, i64 30
  store i8 0, ptr %33, align 1
  %34 = getelementptr inbounds nuw i8, ptr %3, i64 31
  store i8 0, ptr %34, align 1
  %35 = getelementptr inbounds nuw i8, ptr %1, i64 32
  br label %36

36:                                               ; preds = %61, %2
  %37 = phi ptr [ %35, %2 ], [ %43, %61 ]
  %38 = phi ptr [ %3, %2 ], [ %56, %61 ]
  %39 = phi i64 [ 32, %2 ], [ %57, %61 ]
  %40 = phi i64 [ undef, %2 ], [ %58, %61 ]
  %41 = icmp eq ptr %1, %37
  %42 = getelementptr inbounds i8, ptr %37, i64 -8
  %43 = select i1 %41, ptr %37, ptr %42
  br i1 %41, label %46, label %44

44:                                               ; preds = %36
  %45 = load i64, ptr %42, align 8
  br label %46

46:                                               ; preds = %44, %36
  %47 = phi i64 [ %45, %44 ], [ undef, %36 ]
  br i1 %41, label %55, label %48

48:                                               ; preds = %46
  %49 = icmp ult i64 %39, 8
  %50 = add i64 %39, -8
  %.sroa.sel.idx.sroa.sel.idx.sroa.sel.idx.sroa.sel.idx.sroa.sel.idx.sroa.sel.idx.sroa.sel.idx.sroa.sel.idx.sroa.sel.idx.sroa.sel.idx.sroa.sel.idx.sroa.sel.idx.sroa.sel.idx = select i1 %49, i64 0, i64 8
  %.sroa.sel.idx.sroa.sel.idx.sroa.sel.idx.sroa.sel.idx.sroa.sel.idx.sroa.sel.idx.sroa.sel.idx.sroa.sel.idx.sroa.sel.idx.sroa.sel.idx.sroa.sel.idx.sroa.sel.idx.sroa.sel = getelementptr inbounds nuw i8, ptr %38, i64 %.sroa.sel.idx.sroa.sel.idx.sroa.sel.idx.sroa.sel.idx.sroa.sel.idx.sroa.sel.idx.sroa.sel.idx.sroa.sel.idx.sroa.sel.idx.sroa.sel.idx.sroa.sel.idx.sroa.sel.idx.sroa.sel.idx
  %51 = select i1 %49, i64 %39, i64 %50
  %52 = select i1 %49, ptr null, ptr %38
  %53 = icmp eq ptr %52, null
  %54 = select i1 %53, i64 %40, i64 %47
  br label %55

55:                                               ; preds = %48, %46
  %56 = phi ptr [ %.sroa.sel.idx.sroa.sel.idx.sroa.sel.idx.sroa.sel.idx.sroa.sel.idx.sroa.sel.idx.sroa.sel.idx.sroa.sel.idx.sroa.sel.idx.sroa.sel.idx.sroa.sel.idx.sroa.sel.idx.sroa.sel, %48 ], [ %38, %46 ]
  %57 = phi i64 [ %51, %48 ], [ %39, %46 ]
  %58 = phi i64 [ %54, %48 ], [ %40, %46 ]
  %59 = phi ptr [ %52, %48 ], [ null, %46 ]
  %60 = icmp eq ptr %59, null
  br i1 %60, label %63, label %61

61:                                               ; preds = %55
  %62 = call i64 @llvm.bswap.i64(i64 %58)
  store i64 %62, ptr %59, align 1
  br label %36

63:                                               ; preds = %55
  call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 1 dereferenceable(32) %0, ptr noundef nonnull align 1 dereferenceable(32) %3, i64 32, i1 false)
  call void @llvm.lifetime.end.p0(i64 32, ptr nonnull %3)
  ret void
}

define void @kernel(ptr %p) {
  %input = alloca [4 x i64], align 8
  %output = alloca [32 x i8], align 1
  call void @llvm.memcpy.p0.p0.i64(ptr align 8 %input, ptr align 8 %p, i64 32, i1 false)
  call void @encode(ptr %output, ptr %input)
  %destination = getelementptr i8, ptr %p, i64 32
  call void @llvm.memcpy.p0.p0.i64(ptr %destination, ptr %output, i64 32, i1 false)
  ret void
}
