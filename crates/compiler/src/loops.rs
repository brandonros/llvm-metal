//! Dominators and natural loops for one function, as far as retained-call
//! selection needs them. LLVM's C API exposes neither analysis.
use inkwell::llvm_sys::{core::*, prelude::*};
use std::collections::{HashMap, HashSet};

pub(crate) struct Loops {
    // Each header with the blocks of every natural loop that shares it.
    loops: Vec<(LLVMBasicBlockRef, HashSet<LLVMBasicBlockRef>)>,
}

unsafe fn successors(block: LLVMBasicBlockRef) -> Vec<LLVMBasicBlockRef> {
    unsafe {
        let terminator = LLVMGetBasicBlockTerminator(block);
        if terminator.is_null() {
            return Vec::new();
        }
        (0..LLVMGetNumSuccessors(terminator))
            .map(|i| LLVMGetSuccessor(terminator, i))
            .collect()
    }
}

impl Loops {
    /// Unreachable blocks belong to no loop, and a cycle without a dominating
    /// header is not a natural loop. Both match LLVM's LoopInfo.
    pub(crate) unsafe fn of(function: LLVMValueRef) -> Self {
        unsafe {
            // Reverse postorder over reachable blocks, without host recursion.
            let entry = LLVMGetEntryBasicBlock(function);
            let mut postorder = Vec::new();
            let mut seen = HashSet::from([entry]);
            let mut stack = vec![(entry, successors(entry), 0)];
            while let Some((block, next, position)) = stack.last_mut() {
                if let Some(&successor) = next.get(*position) {
                    *position += 1;
                    if seen.insert(successor) {
                        stack.push((successor, successors(successor), 0));
                    }
                } else {
                    postorder.push(*block);
                    stack.pop();
                }
            }
            let order: Vec<_> = postorder.into_iter().rev().collect();
            let number: HashMap<_, _> = order.iter().enumerate().map(|(i, &b)| (b, i)).collect();
            let mut predecessors = vec![Vec::new(); order.len()];
            for (i, &block) in order.iter().enumerate() {
                for successor in successors(block) {
                    predecessors[number[&successor]].push(i);
                }
            }
            // Cooper, Harvey and Kennedy's iterative immediate dominators.
            let mut idom = vec![usize::MAX; order.len()];
            idom[0] = 0;
            let mut changed = true;
            while changed {
                changed = false;
                for block in 1..order.len() {
                    let mut new = usize::MAX;
                    for &predecessor in &predecessors[block] {
                        if idom[predecessor] == usize::MAX {
                            continue;
                        }
                        new = if new == usize::MAX {
                            predecessor
                        } else {
                            let (mut a, mut b) = (predecessor, new);
                            while a != b {
                                while a > b {
                                    a = idom[a];
                                }
                                while b > a {
                                    b = idom[b];
                                }
                            }
                            a
                        };
                    }
                    if idom[block] != new {
                        idom[block] = new;
                        changed = true;
                    }
                }
            }
            let dominates = |a: usize, mut b: usize| {
                while b != a && b != 0 {
                    b = idom[b];
                }
                a == b
            };
            // A back edge targets a block that dominates its source. Its loop is
            // every block that reaches the source without passing the header.
            let mut bodies: HashMap<usize, HashSet<usize>> = HashMap::new();
            for (source, &block) in order.iter().enumerate() {
                for successor in successors(block) {
                    let header = number[&successor];
                    if !dominates(header, source) {
                        continue;
                    }
                    let body = bodies
                        .entry(header)
                        .or_insert_with(|| HashSet::from([header]));
                    let mut pending = vec![source];
                    while let Some(block) = pending.pop() {
                        if body.insert(block) {
                            pending.extend(&predecessors[block]);
                        }
                    }
                }
            }
            let loops = bodies
                .into_iter()
                .map(|(header, body)| (order[header], body.into_iter().map(|b| order[b]).collect()))
                .collect();
            Self { loops }
        }
    }

    /// Headers of every loop containing the block, innermost or not.
    pub(crate) fn headers_containing(
        &self,
        block: LLVMBasicBlockRef,
    ) -> impl Iterator<Item = LLVMBasicBlockRef> + '_ {
        self.loops
            .iter()
            .filter(move |(_, body)| body.contains(&block))
            .map(|(header, _)| *header)
    }
}
