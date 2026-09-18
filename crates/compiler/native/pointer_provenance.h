// Shared pointer classification for memory validation and native transforms.
#pragma once
#include <llvm-c/Core.h>

namespace llvm { class Value; }
namespace llvm_metal {
// Unknown provenance, escaping helpers and unresolved recursion fail closed.
bool privatePointer(llvm::Value *pointer);
bool constantPointer(llvm::Value *pointer);
}

// Instruction-level predicate also called from Rust and wide-integer lowering.
extern "C" bool LLVMMetalPrivateMemory(LLVMValueRef value);
