//! What a kernel author imports.
//!
//! A kernel is an ordinary Rust function whose parameters are handles: the
//! thread it runs as, and the buffers it may read or write. Device memory is
//! reachable only through those handles, so every pointer the kernel itself
//! creates is thread memory. The same function runs on the host (see `host`),
//! which is the reference for what it must compute on the GPU.
#![cfg_attr(target_arch = "nvptx64", no_std)]

mod device;
#[cfg(not(target_arch = "nvptx64"))]
pub mod host;

use core::{marker::PhantomData, mem::MaybeUninit};

/// Plain data: any bit pattern is a value, and there is no padding to leak.
///
/// # Safety
/// The type is `repr(C)` or primitive, holds no pointers or references, and has
/// no padding bytes.
pub unsafe trait Plain: Copy {}

macro_rules! plain {
    ($($ty:ty),*) => { $(unsafe impl Plain for $ty {})* };
}
plain!(u8, u16, u32, u64, i8, i16, i32, i64);
unsafe impl<T: Plain, const N: usize> Plain for [T; N] {}

/// A parameter the entry point can create from nothing: it names a thread or a
/// buffer slot, and holds no data.
pub trait Handle {
    const HANDLE: Self;
}

/// The invocation this kernel is running as.
pub struct Thread(());

impl Thread {
    pub fn index(&self) -> u32 {
        device::thread_index()
    }
}

/// A buffer the kernel reads. `SLOT` is its Metal buffer index.
pub struct In<T: ?Sized, const SLOT: u32>(PhantomData<T>);

/// A buffer the kernel writes. `SLOT` is its Metal buffer index.
pub struct Out<T: ?Sized, const SLOT: u32>(PhantomData<T>);

impl Handle for Thread {
    const HANDLE: Self = Thread(());
}
impl<T: ?Sized, const SLOT: u32> Handle for In<T, SLOT> {
    const HANDLE: Self = In(PhantomData);
}
impl<T: ?Sized, const SLOT: u32> Handle for Out<T, SLOT> {
    const HANDLE: Self = Out(PhantomData);
}

impl<T: Plain, const SLOT: u32> In<T, SLOT> {
    /// Copy the value out of device memory, if the bound buffer holds one.
    pub fn read(&self) -> Option<T> {
        (device::length(SLOT) as usize >= size_of::<T>())
            // SAFETY: the buffer holds at least one `T`.
            .then(|| unsafe { load::<T, SLOT>(0) })
    }
}

impl<T: Plain, const SLOT: u32> In<[T], SLOT> {
    pub fn len(&self) -> usize {
        device::length(SLOT) as usize / size_of::<T>()
    }

    /// Copy one element out of device memory.
    pub fn get(&self, index: usize) -> Option<T> {
        // SAFETY: the element is inside the buffer.
        (index < self.len()).then(|| unsafe { load::<T, SLOT>(index * size_of::<T>()) })
    }
}

impl<T: Plain, const SLOT: u32> Out<[T], SLOT> {
    pub fn len(&self) -> usize {
        device::length(SLOT) as usize / size_of::<T>()
    }

    /// Copy one element into device memory. An index past the end writes
    /// nothing: a dispatch may run more threads than the buffer has elements.
    pub fn write(&self, index: usize, value: T) {
        if index >= self.len() {
            return;
        }
        let bytes = (&raw const value).cast::<u8>();
        for offset in 0..size_of::<T>() {
            // SAFETY: `offset` is inside `value`, which is `Plain`, and the
            // element is inside the buffer.
            unsafe {
                let byte = bytes.add(offset).read();
                device::store(SLOT, (index * size_of::<T>() + offset) as u64, byte);
            }
        }
    }
}

/// The slot is a const generic, never an argument: the compiler requires a
/// constant slot at every intrinsic call, whatever was or was not inlined.
///
/// # Safety
/// `start + size_of::<T>()` is inside the buffer.
unsafe fn load<T: Plain, const SLOT: u32>(start: usize) -> T {
    let mut value = MaybeUninit::<T>::uninit();
    let bytes = value.as_mut_ptr().cast::<u8>();
    for offset in 0..size_of::<T>() {
        // SAFETY: `offset` is inside `value`, and the caller checked the buffer.
        unsafe {
            bytes
                .add(offset)
                .write(device::load(SLOT, (start + offset) as u64))
        };
    }
    // SAFETY: every byte was written, and any bit pattern is a `Plain` value.
    unsafe { value.assume_init() }
}

/// Declare a kernel: the function itself, callable on the host, and on the GPU
/// target an entry point named `kernel.<name>` that calls it with its handles.
#[macro_export]
macro_rules! kernel {
    ($(#[$meta:meta])* pub fn $name:ident($($argument:ident: $ty:ty),* $(,)?) $body:block) => {
        $(#[$meta])*
        pub fn $name($($argument: $ty),*) $body

        #[cfg(target_arch = "nvptx64")]
        const _: () = {
            #[unsafe(export_name = concat!("kernel.", stringify!($name)))]
            pub extern "C" fn entry() {
                $name($(<$ty as $crate::Handle>::HANDLE),*)
            }
        };
    };
}
