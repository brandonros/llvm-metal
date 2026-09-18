//! Host decoding and validation of the kernel crate's versioned byte encoding.
use crate::{Access, BufferArgument, Dispatch, KernelInterface, MetalBindings};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Descriptor {
    pub version: u32,
    pub entry: String,
    pub dispatch: Dispatch,
    pub arguments: Vec<Argument>,
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Argument {
    pub name: String,
    pub access: Access,
    pub slice: bool,
    pub layout: Layout,
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Layout {
    pub size: u32,
    pub alignment: u32,
    pub kind: Kind,
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Kind {
    Unsigned,
    Signed,
    Array { count: u32, element: Box<Layout> },
    Record { fields: Vec<Field> },
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Field {
    pub name: String,
    pub offset: u32,
    pub layout: Layout,
}

struct Reader<'a>(&'a [u8]);
impl Reader<'_> {
    fn take(&mut self, n: usize) -> Result<&[u8], String> {
        if n > self.0.len() {
            return Err("truncated descriptor".into());
        }
        let (head, tail) = self.0.split_at(n);
        self.0 = tail;
        Ok(head)
    }
    fn byte(&mut self) -> Result<u8, String> {
        Ok(self.take(1)?[0])
    }
    fn number(&mut self) -> Result<u32, String> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
    fn text(&mut self) -> Result<String, String> {
        let n = self.number()? as usize;
        let text =
            std::str::from_utf8(self.take(n)?).map_err(|_| "descriptor text must be UTF-8")?;
        if text.is_empty() || text.contains('\0') {
            return Err("invalid descriptor name".into());
        }
        Ok(text.to_owned())
    }
    fn layout(&mut self, depth: usize) -> Result<Layout, String> {
        if depth > 32 {
            return Err("descriptor layout nesting exceeds 32".into());
        }
        let size = self.number()?;
        let alignment = self.number()?;
        let kind = match self.byte()? {
            0 => Kind::Unsigned,
            1 => Kind::Signed,
            2 => Kind::Array {
                count: self.number()?,
                element: Box::new(self.layout(depth + 1)?),
            },
            3 => {
                let count = self.number()?;
                if count > 256 {
                    return Err("too many record fields".into());
                }
                let mut fields = Vec::new();
                for _ in 0..count {
                    fields.push(Field {
                        name: self.text()?,
                        offset: self.number()?,
                        layout: self.layout(depth + 1)?,
                    });
                }
                Kind::Record { fields }
            }
            _ => return Err("unknown descriptor layout kind".into()),
        };
        Ok(Layout {
            size,
            alignment,
            kind,
        })
    }
}
impl Layout {
    fn validate(&self, depth: usize) -> Result<(), String> {
        if depth > 32
            || self.size == 0
            || !self.alignment.is_power_of_two()
            || self.alignment > 16
            || self.size % self.alignment != 0
        {
            return Err("invalid or unsupported record layout".into());
        }
        match &self.kind {
            Kind::Unsigned | Kind::Signed => {
                if !matches!(self.size, 1 | 2 | 4 | 8) || self.alignment != self.size {
                    return Err("unsupported integer layout".into());
                }
            }
            Kind::Array { count, element } => {
                element.validate(depth + 1)?;
                if element.size.checked_mul(*count) != Some(self.size)
                    || element.alignment != self.alignment
                {
                    return Err("invalid array layout".into());
                }
            }
            Kind::Record { fields } => {
                if fields.is_empty() || fields.len() > 256 {
                    return Err("invalid record field count".into());
                }
                let mut offset = 0u32;
                let mut alignment = 1;
                let mut names = std::collections::HashSet::new();
                for field in fields {
                    if field.name.is_empty()
                        || field.name.contains('\0')
                        || !names.insert(&field.name)
                    {
                        return Err("invalid or duplicate field name".into());
                    }
                    field.layout.validate(depth + 1)?;
                    if field.offset != offset || field.offset % field.layout.alignment != 0 {
                        return Err("record padding, overlap or misalignment".into());
                    }
                    offset = offset
                        .checked_add(field.layout.size)
                        .ok_or("record size overflow")?;
                    alignment = alignment.max(field.layout.alignment);
                }
                if offset != self.size || alignment != self.alignment {
                    return Err("record size/alignment mismatch".into());
                }
            }
        }
        Ok(())
    }
}
impl Descriptor {
    pub fn decode(bytes: &[u8]) -> Result<Self, String> {
        if bytes.len() > llvm_metal_kernel::MAX_DESCRIPTOR_BYTES {
            return Err("descriptor too large".into());
        }
        let mut r = Reader(bytes);
        if r.number()? != 0x444d4c {
            return Err("invalid descriptor magic".into());
        }
        let version = r.number()?;
        if version != llvm_metal_kernel::VERSION {
            return Err("unsupported descriptor version".into());
        }
        let entry = r.text()?;
        let dispatch = match r.byte()? {
            0 => Dispatch::Single,
            1 => Dispatch::Grid1d,
            _ => return Err("unsupported dispatch".into()),
        };
        if r.byte()? != 0 {
            return Err("only little-endian device records are supported".into());
        }
        let count = r.number()?;
        if count == 0 || count > 31 {
            return Err("invalid descriptor argument count".into());
        }
        let mut arguments = Vec::new();
        for _ in 0..count {
            let name = r.text()?;
            let access = match r.byte()? {
                0 => Access::Read,
                1 => Access::Write,
                2 => Access::ReadWrite,
                _ => return Err("invalid buffer access".into()),
            };
            let slice = match r.byte()? {
                0 => false,
                1 => true,
                _ => return Err("invalid buffer shape".into()),
            };
            arguments.push(Argument {
                name,
                access,
                slice,
                layout: r.layout(0)?,
            });
        }
        if !r.0.is_empty() {
            return Err("trailing descriptor bytes".into());
        }
        let descriptor = Self {
            version,
            entry,
            dispatch,
            arguments,
        };
        descriptor.interface()?;
        Ok(descriptor)
    }
    pub fn interface(&self) -> Result<KernelInterface, String> {
        if self.version != llvm_metal_kernel::VERSION {
            return Err("unsupported descriptor version".into());
        }
        let mut names = std::collections::HashSet::new();
        let mut arguments = Vec::new();
        for a in &self.arguments {
            if a.name.is_empty() || !names.insert(&a.name) {
                return Err("empty or duplicate argument name".into());
            }
            a.layout.validate(0)?;
            arguments.push(BufferArgument {
                name: a.name.clone(),
                kind: "buffer".into(),
                access: a.access,
                bytes: a.layout.size as usize,
                alignment: a.layout.alignment as usize,
            });
        }
        let interface = KernelInterface {
            schema: 1,
            entry: self.entry.clone(),
            calling_convention: "C".into(),
            invocations: (self.dispatch == Dispatch::Single).then_some(1),
            dispatch: self.dispatch,
            aliasing: "all buffers disjoint".into(),
            arguments,
        };
        interface.validate()?;
        Ok(interface)
    }
    pub fn bindings(&self) -> Result<MetalBindings, String> {
        let mut bindings = self.interface()?.validate()?;
        bindings.descriptor = Some(self.clone());
        Ok(bindings)
    }
    /// Counts are logical element counts, supplied by the typed launcher after
    /// validating workload-specific length relationships. Empty slices still
    /// require one backed element. This does not prove arbitrary kernel accesses.
    pub fn validate_lengths(&self, bytes: &[usize], counts: &[usize]) -> Result<(), String> {
        // Layout validation happens at decode/binding creation. Keep repeated
        // launch checks allocation-free on success.
        if self.version != llvm_metal_kernel::VERSION || self.arguments.is_empty() {
            return Err("invalid descriptor".into());
        }
        if bytes.len() != self.arguments.len() || counts.len() != bytes.len() {
            return Err("argument count mismatch".into());
        }
        for ((a, &bytes), &count) in self.arguments.iter().zip(bytes).zip(counts) {
            if a.layout.size == 0 {
                return Err("zero-sized buffer element".into());
            }
            if !a.slice && count != 1 {
                return Err(format!("{} is a fixed record", a.name));
            }
            let needed = (a.layout.size as usize)
                .checked_mul(count.max(1))
                .ok_or("buffer length overflow")?;
            if bytes < needed {
                return Err(format!("{} buffer is undersized", a.name));
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use llvm_metal_kernel as k;
    k::record! { #[derive(Clone, Copy)] pub struct Request { pub count: u32, pub add: u32 } }
    fn encoded() -> k::Encoded {
        k::encode(
            "kernel",
            k::Dispatch::Single,
            &[
                k::argument::<Request>("request", k::Access::Read, k::Shape::Fixed),
                k::argument::<u32>("data", k::Access::ReadWrite, k::Shape::Slice),
            ],
        )
    }
    #[test]
    fn target_layout_roundtrip_and_dynamic_bounds() {
        let d = Descriptor::decode(encoded().as_bytes()).unwrap();
        assert_eq!(d.arguments[0].layout.size, 8);
        assert!(d.validate_lengths(&[8, 12], &[1, 3]).is_ok());
        assert!(d.validate_lengths(&[8, 8], &[1, 3]).is_err());
        assert!(d.validate_lengths(&[8, 4], &[1, 0]).is_ok());
        assert!(d.validate_lengths(&[8, 0], &[1, 0]).is_err());
        assert!(
            d.validate_lengths(&[8, usize::MAX], &[1, usize::MAX])
                .is_err()
        );
        let mut swapped = d.clone();
        if let Kind::Record { fields } = &mut swapped.arguments[0].layout.kind {
            fields[0].name = "add".into();
            fields[1].name = "count".into();
        }
        assert_eq!(swapped.arguments[0].layout.size, d.arguments[0].layout.size);
        assert_ne!(swapped, d);
        let encoded = encoded();
        assert!(
            swapped
                .bindings()
                .unwrap()
                .validate_host_descriptor(encoded.as_bytes())
                .is_err()
        );
        let mut legacy = d.bindings().unwrap();
        legacy.descriptor = None;
        assert!(legacy.validate_host_descriptor(encoded.as_bytes()).is_err());
    }
    #[test]
    fn reject_truncation_version_and_padding() {
        let e = encoded();
        let bytes = e.as_bytes();
        for end in 0..bytes.len() {
            assert!(Descriptor::decode(&bytes[..end]).is_err());
        }
        let mut bad = bytes.to_vec();
        bad[4] = 99;
        assert!(Descriptor::decode(&bad).is_err());
        let mut bad = bytes.to_vec();
        bad.push(0);
        assert!(Descriptor::decode(&bad).is_err());
        let mut d = Descriptor::decode(bytes).unwrap();
        if let Kind::Record { fields } = &mut d.arguments[0].layout.kind {
            fields[1].offset = 0;
        }
        assert!(d.interface().is_err());
    }
}
