//! Target-evaluated, allocation-free declarations for the Metal buffer ABI.
#![no_std]

pub const PREFIX: &str = "__llvm_metal_descriptor_";
pub const VERSION: u32 = 1;
pub const MAX_DESCRIPTOR_BYTES: usize = 16384;

#[derive(Clone, Copy)]
pub struct Field {
    pub name: &'static str,
    pub offset: usize,
    pub layout: &'static Layout,
}
#[derive(Clone, Copy)]
pub enum Kind {
    Unsigned,
    Signed,
    Array(usize, &'static Layout),
    Record(&'static [Field]),
}
#[derive(Clone, Copy)]
pub struct Layout {
    pub size: usize,
    pub alignment: usize,
    pub kind: Kind,
}
/// A padding-free, pointer-free transferable type accepting every bit pattern.
///
/// # Safety
/// LAYOUT must describe every field, recursively, with the actual target layout.
/// Use `record!` for records; pointers, bool, enums and usize are unsupported.
pub unsafe trait DeviceLayout: Copy {
    const LAYOUT: Layout;
}
macro_rules! integers {
    ($kind:ident; $($ty:ty),*) => {$ (
        // SAFETY: fixed-width integers accept every bit pattern and have no padding.
        unsafe impl DeviceLayout for $ty {
            const LAYOUT: Layout = Layout { size: size_of::<Self>(), alignment: align_of::<Self>(), kind: Kind::$kind };
        }
    )*};
}
integers!(Unsigned; u8, u16, u32, u64);
integers!(Signed; i8, i16, i32, i64);
// SAFETY: contiguous elements inherit the element's layout and validity.
unsafe impl<T: DeviceLayout, const N: usize> DeviceLayout for [T; N] {
    const LAYOUT: Layout = Layout {
        size: size_of::<Self>(),
        alignment: align_of::<Self>(),
        kind: Kind::Array(N, &T::LAYOUT),
    };
}

/// Declare a C record and its complete layout together. Padding is rejected.
///
/// ```compile_fail
/// llvm_metal_kernel::record! {
///     #[derive(Clone, Copy)] struct Padded { tag: u8, word: u32 }
/// }
/// ```
/// ```compile_fail
/// llvm_metal_kernel::record! {
///     #[derive(Clone, Copy)] struct InvalidBits { flag: bool }
/// }
#[macro_export]
macro_rules! record {
    ($(#[$attr:meta])* $vis:vis struct $name:ident { $($(#[$fa:meta])* $fv:vis $field:ident: $ty:ty),* $(,)? }) => {
        $(#[$attr])* #[repr(C)] $vis struct $name { $($(#[$fa])* $fv $field: $ty),* }
        // SAFETY: repr(C), all fields have DeviceLayout, and total size excludes padding.
        unsafe impl $crate::DeviceLayout for $name {
            const LAYOUT: $crate::Layout = {
                assert!(core::mem::size_of::<Self>() == 0 $(+ core::mem::size_of::<$ty>())*);
                $crate::Layout { size: core::mem::size_of::<Self>(), alignment: core::mem::align_of::<Self>(), kind: $crate::Kind::Record(&[
                    $($crate::Field { name: stringify!($field), offset: core::mem::offset_of!($name, $field), layout: &<$ty as $crate::DeviceLayout>::LAYOUT }),*
                ]) }
            };
        }
        const _: () = { let _ = <$name as $crate::DeviceLayout>::LAYOUT; };
    };
}

#[derive(Clone, Copy)]
pub enum Access {
    Read = 0,
    Write = 1,
    ReadWrite = 2,
}
#[derive(Clone, Copy)]
pub enum Shape {
    Fixed = 0,
    Slice = 1,
}
#[derive(Clone, Copy)]
pub enum Dispatch {
    Single = 0,
    Grid1d = 1,
}
#[derive(Clone, Copy)]
pub struct Argument {
    pub name: &'static str,
    pub access: Access,
    pub shape: Shape,
    pub layout: Layout,
}
pub const fn argument<T: DeviceLayout>(
    name: &'static str,
    access: Access,
    shape: Shape,
) -> Argument {
    Argument {
        name,
        access,
        shape,
        layout: T::LAYOUT,
    }
}

/// Canonical little-endian encoding. Only `bytes[..len]` is emitted into LLVM.
pub struct Encoded {
    pub bytes: [u8; MAX_DESCRIPTOR_BYTES],
    pub len: usize,
}
impl Encoded {
    const fn byte(&mut self, value: u8) {
        assert!(self.len < MAX_DESCRIPTOR_BYTES, "descriptor too large");
        self.bytes[self.len] = value;
        self.len += 1;
    }
    const fn number(&mut self, value: usize) {
        assert!(value <= u32::MAX as usize, "descriptor integer overflow");
        let data = (value as u32).to_le_bytes();
        let mut i = 0;
        while i < 4 {
            self.byte(data[i]);
            i += 1;
        }
    }
    const fn text(&mut self, text: &str) {
        self.number(text.len());
        let mut i = 0;
        while i < text.len() {
            self.byte(text.as_bytes()[i]);
            i += 1;
        }
    }
    const fn layout(&mut self, layout: &Layout) {
        self.number(layout.size);
        self.number(layout.alignment);
        match layout.kind {
            Kind::Unsigned => self.byte(0),
            Kind::Signed => self.byte(1),
            Kind::Array(count, element) => {
                self.byte(2);
                self.number(count);
                self.layout(element);
            }
            Kind::Record(fields) => {
                self.byte(3);
                self.number(fields.len());
                let mut i = 0;
                while i < fields.len() {
                    self.text(fields[i].name);
                    self.number(fields[i].offset);
                    self.layout(fields[i].layout);
                    i += 1;
                }
            }
        }
    }
    pub const fn exact<const N: usize>(&self) -> [u8; N] {
        assert!(N == self.len);
        let mut out = [0; N];
        let mut i = 0;
        while i < N {
            out[i] = self.bytes[i];
            i += 1;
        }
        out
    }
    pub fn as_bytes(&self) -> &[u8] {
        &self.bytes[..self.len]
    }
}
pub const fn encode(entry: &str, dispatch: Dispatch, args: &[Argument]) -> Encoded {
    let mut out = Encoded {
        bytes: [0; MAX_DESCRIPTOR_BYTES],
        len: 0,
    };
    out.number(0x444d4c); // LMD\0
    out.number(VERSION as usize);
    out.text(entry);
    out.byte(dispatch as u8);
    // Current profile: little-endian records and disjoint buffers only.
    out.byte(if cfg!(target_endian = "little") { 0 } else { 1 });
    out.number(args.len());
    let mut i = 0;
    while i < args.len() {
        out.text(args[i].name);
        out.byte(args[i].access as u8);
        out.byte(args[i].shape as u8);
        out.layout(&args[i].layout);
        i += 1;
    }
    out
}

/// One declaration generates the pointer entry, target descriptor and named
/// host argument container. The unsafe body still owns all dynamic accesses.
#[macro_export]
macro_rules! kernel {
    ($vis:vis mod $module:ident; entry $export:expr; $(#[$attr:meta])* $fv:vis unsafe extern "C" fn $entry:ident (
        $($arg:ident: $access:ident $shape:ident $ty:ty),* $(,)?
    ) $body:block dispatch $dispatch:ident;) => {
        $vis mod $module {
            #[allow(unused_imports)] use super::*;
            pub const ENTRY: &str = $export;
            pub const ARGUMENTS: &[$crate::Argument] = &[
                $($crate::argument::<$ty>(stringify!($arg), $crate::Access::$access, $crate::Shape::$shape)),*
            ];
            const ENCODED: $crate::Encoded = $crate::encode(ENTRY, $crate::Dispatch::$dispatch, ARGUMENTS);
            pub const DESCRIPTOR: [u8; ENCODED.len] = ENCODED.exact();
            #[cfg(target_arch = "nvptx64")]
            #[used]
            #[unsafe(export_name = concat!("__llvm_metal_descriptor_", $export))]
            static DEVICE_DESCRIPTOR: [u8; ENCODED.len] = DESCRIPTOR;
            /// Values are serialized in the generated entry's exact argument order.
            pub struct Arguments<T> { $(pub $arg: T),* }
            impl<T> Arguments<T> {
                pub fn into_array(self) -> [T; ARGUMENTS.len()] { [$(self.$arg),*] }
                pub fn from_fn(mut f: impl FnMut(usize, $crate::Argument) -> T) -> Self {
                    let mut args = ARGUMENTS.iter().copied().enumerate();
                    Self { $($arg: { let (index, a) = args.next().unwrap(); f(index, a) }),* }
                }
            }
        }
        $(#[$attr])*
        #[unsafe(export_name = $export)]
        $fv unsafe extern "C" fn $entry($($arg: $crate::kernel!(@ptr $access $ty)),*) $body
    };
    ($vis:vis mod $module:ident; $(#[$attr:meta])* $fv:vis unsafe extern "C" fn $entry:ident (
        $($arg:ident: $access:ident $shape:ident $ty:ty),* $(,)?
    ) $body:block dispatch $dispatch:ident;) => {
        $crate::kernel! { $vis mod $module; entry stringify!($entry);
            $(#[$attr])* $fv unsafe extern "C" fn $entry($($arg: $access $shape $ty),*) $body dispatch $dispatch;
        }
    };
    (@ptr Read $ty:ty) => { *const $ty };
    (@ptr Write $ty:ty) => { *mut $ty };
    (@ptr ReadWrite $ty:ty) => { *mut $ty };
}
