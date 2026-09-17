use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct KernelInterface {
    pub schema: u32,
    pub entry: String,
    pub calling_convention: String,
    pub invocations: Option<u32>,
    #[serde(default)]
    pub dispatch: Dispatch,
    pub arguments: Vec<BufferArgument>,
    pub aliasing: String,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct BufferArgument {
    pub name: String,
    pub kind: String,
    pub access: Access,
    pub bytes: usize,
    pub alignment: usize,
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Access {
    Read,
    Write,
    ReadWrite,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Dispatch {
    #[default]
    Single,
    Grid1d,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct MetalBindings {
    pub entry: String,
    pub dispatch: Dispatch,
    pub buffers: Vec<MetalBufferBinding>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct MetalBufferBinding {
    pub argument: usize,
    pub index: usize,
    pub minimum_bytes: usize,
    pub alignment: usize,
    pub access: Access,
}

impl KernelInterface {
    pub fn validate(&self) -> Result<MetalBindings, String> {
        if self.schema != 1
            || self.calling_convention != "C"
            || match self.dispatch {
                Dispatch::Single => self.invocations != Some(1),
                Dispatch::Grid1d => self.invocations.is_some(),
            }
        {
            return Err(
                "supported contract: schema 1, C ABI, single invocation or dynamic 1D grid".into(),
            );
        }
        if self.entry.is_empty()
            || self.entry.contains('\0')
            || self.arguments.is_empty()
            || self.arguments.len() > 31
        {
            return Err("invalid entry or buffer count".into());
        }
        for arg in &self.arguments {
            if arg.kind != "buffer"
                || arg.bytes == 0
                || !arg.alignment.is_power_of_two()
                || arg.alignment > 16
                || arg.name.contains('\0')
            {
                return Err(format!("unsupported buffer contract: {}", arg.name));
            }
        }
        Ok(MetalBindings {
            entry: self.entry.clone(),
            dispatch: self.dispatch,
            buffers: self
                .arguments
                .iter()
                .enumerate()
                .map(|(index, arg)| MetalBufferBinding {
                    argument: index,
                    index,
                    minimum_bytes: arg.bytes,
                    alignment: arg.alignment,
                    access: arg.access,
                })
                .collect(),
        })
    }
}
