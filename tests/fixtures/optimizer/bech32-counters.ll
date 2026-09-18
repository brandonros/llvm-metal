; Compile-time regression, not a complete Bech32 encoder.
; Reduced from consumer_bitcoin_bech32 after the first producer O3 pass:
; llvm-metal f4ef33c, vanity-miner a85a548, Rust 1.93.0 / LLVM 21.1.8.
; llvm-reduce retained the two unrolled 8-to-5 bit-count loops. Restore the
; reducer's null store base to output and remove its infinite trailing loop.
; Contract: output points to 64 writable bytes; write zero to bytes 1, 2, 3
; and leave every other byte unchanged. The outer loop consumes 20 bytes.
; Old second O3 exceeds the 10-second regression deadline in ScalarEvolution;
; explicit indvars + unroll + scalar cleanup before O3 finishes in <0.1 s.
source_filename = "llvm-link"
target datalayout = "e-p6:32:32-i64:64-i128:128-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

define void @bech32_counters(ptr %output) unnamed_addr {
start:
  br label %14

0:                                                ; preds = %23
  %1 = add nuw nsw i64 %16, 2
  %2 = or disjoint i8 %24, 8
  br label %3

3:                                                ; preds = %7, %0
  %4 = phi i8 [ %2, %0 ], [ %8, %7 ]
  %5 = phi i64 [ %25, %0 ], [ %10, %7 ]
  %6 = icmp ult i64 %5, 64
  br i1 %6, label %7, label %.loopexit24.i

7:                                                ; preds = %3
  %8 = add i8 %4, -5
  %9 = getelementptr inbounds nuw i8, ptr %output, i64 %5
  store i8 0, ptr %9, align 1
  %10 = add nuw nsw i64 %5, 1
  %11 = icmp ugt i8 %8, 4
  br i1 %11, label %3, label %12

12:                                               ; preds = %7
  %13 = icmp eq i64 %1, 20
  br i1 %13, label %18, label %14

14:                                               ; preds = %12, %start
  %15 = phi i8 [ 0, %start ], [ %8, %12 ]
  %16 = phi i64 [ 0, %start ], [ %1, %12 ]
  %17 = or disjoint i8 %15, 8
  br label %19

18:                                               ; preds = %12
  br label %27

19:                                               ; preds = %23, %14
  %20 = phi i8 [ %17, %14 ], [ %24, %23 ]
  %21 = phi i64 [ 0, %14 ], [ %25, %23 ]
  %22 = icmp ult i64 %21, 64
  br i1 %22, label %23, label %.loopexit24.i

23:                                               ; preds = %19
  %24 = add nsw i8 %20, -5
  %25 = add nuw nsw i64 %21, 1
  %26 = icmp ugt i8 %24, 4
  br i1 %26, label %19, label %0

.loopexit24.i:                                    ; preds = %19, %3
  unreachable

27:
  ret void

; uselistorder directives
  uselistorder i8 %24, { 0, 2, 1 }
  uselistorder i64 %25, { 1, 0 }
}
