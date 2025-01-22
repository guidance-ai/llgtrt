use std::collections::HashMap;
use std::sync::Mutex;

struct LoraCacheInternal {
    id_map: HashMap<String, u64>,
    max_id: u64,
}

pub struct LoraCache {
    inner: Mutex<LoraCacheInternal>,
}

impl LoraCache {
    pub fn new() -> Self {
        let inner = LoraCacheInternal {
            id_map: HashMap::new(),
            max_id: 1,
        };
        LoraCache {
            inner: Mutex::new(inner),
        }
    }

    /// Returns the unsigned integer Id to use for a given lora directory string, finding one if necessary.
    pub fn resolve_id(&self, lora_dir: &str) -> u64 {
        let mut inner = self.inner.lock().unwrap();
        if let Some(id) = inner.id_map.get(lora_dir) {
            *id
        } else {
            inner.max_id += 1;
            let lora_id = inner.max_id;
            inner.id_map.insert(lora_dir.to_string(), lora_id);
            lora_id
        }
    }
}
