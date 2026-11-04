//! Everything baked into, or mapped from, flash: the model partition, the
//! tokenizer and the test vectors, plus the little-endian readers they share.

use esp_idf_svc::sys;
use std::ffi::{c_void, CString};
use std::sync::OnceLock;

pub static TOKENIZER_BIN: &[u8] = include_bytes!("../models/tokenizer.bin");

// 16 reference tokens + PyTorch logits from the export notebook (131 KB, compiled into the firmware)
pub static TEST_VECTORS: &[u8] = include_bytes!("../models/test_vectors.bin");

static MODEL: OnceLock<&'static [u8]> = OnceLock::new();

pub fn rd_u32(b: &[u8], o: usize) -> u32 {
    u32::from_le_bytes(b[o..o + 4].try_into().unwrap())
}

pub fn rd_f32(b: &[u8], o: usize) -> f32 {
    f32::from_le_bytes(b[o..o + 4].try_into().unwrap())
}

/// Memory-map the "model" flash partition (zero-copy, read-only). Mapped once, cached.
pub fn model() -> Result<&'static [u8], String> {
    if let Some(m) = MODEL.get() {
        return Ok(m);
    }
    let label = CString::new("model").unwrap();
    let m: &'static [u8] = unsafe {
        let part = sys::esp_partition_find_first(
            sys::esp_partition_type_t_ESP_PARTITION_TYPE_DATA,
            sys::esp_partition_subtype_t_ESP_PARTITION_SUBTYPE_ANY,
            label.as_ptr(),
        );
        if part.is_null() {
            return Err("partition 'model' not found".into());
        }
        let size = (*part).size as usize;
        let mut ptr: *const c_void = std::ptr::null();
        let mut handle: sys::esp_partition_mmap_handle_t = 0;
        let err = sys::esp_partition_mmap(
            part,
            0,
            size,
            sys::esp_partition_mmap_memory_t_ESP_PARTITION_MMAP_DATA,
            &mut ptr,
            &mut handle,
        );
        if err != 0 {
            return Err(format!("esp_partition_mmap failed: {}", err));
        }
        // Copy the model into PSRAM (u32 buffer keeps it 4-byte aligned). Done once, ~2 MB.
        let mut copy = vec![0u32; size / 4];
        std::ptr::copy_nonoverlapping(ptr as *const u8, copy.as_mut_ptr() as *mut u8, size);
        let leaked: &'static mut [u32] = Box::leak(copy.into_boxed_slice());
        std::slice::from_raw_parts(leaked.as_ptr() as *const u8, size)
    };
    let _ = MODEL.set(m);
    Ok(m)
}
