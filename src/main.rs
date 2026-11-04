use esp_idf_svc::sys;
use std::ffi::{c_void, CString};
use std::io::{self, ErrorKind, Read, Write};
use std::sync::OnceLock;
use std::thread::sleep;
use std::time::Duration;
mod engine;
mod runner;
const MAX_LINE: usize = 128;

fn prompt(stdout: &mut io::Stdout) {
    print!("esp32> ");
    let _ = stdout.flush();
}
static MODEL: OnceLock<&'static [u8]> = OnceLock::new();

fn rd_u32(b: &[u8], o: usize) -> u32 {
    u32::from_le_bytes(b[o..o + 4].try_into().unwrap())
}
fn rd_f32(b: &[u8], o: usize) -> f32 {
    f32::from_le_bytes(b[o..o + 4].try_into().unwrap())
}

/// Memory-map the "model" flash partition (zero-copy, read-only). Mapped once, cached.
fn model() -> Result<&'static [u8], String> {
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
        std::slice::from_raw_parts(ptr as *const u8, size)
    };
    let _ = MODEL.set(m);
    Ok(m)
}
// 16 reference tokens + PyTorch logits from your export notebook (131 KB, compiled into the firmware)
static TEST_VECTORS: &[u8] = include_bytes!("../models/test_vectors.bin");

fn run_test() -> Result<(), String> {
    let (m, _) = engine::Model::parse(model()?)?;
    let tv = TEST_VECTORS;
    let n = rd_u32(tv, 0) as usize;
    let v = rd_u32(tv, 4) as usize;
    if v != m.cfg.vocab || tv.len() != 8 + 4 * n + 4 * n * v {
        return Err("test_vectors.bin has an unexpected size".into());
    }

    let mut r = runner::Runner::new(m);
    let (mut max_d, mut sum_d, mut agree) = (0f32, 0f64, 0usize);
    let t0 = std::time::Instant::now();
    for i in 0..n {
        let tok = rd_u32(tv, 8 + 4 * i) as usize;
        let logits = r.step(tok, i);
        let base = 8 + 4 * n + 4 * i * v;
        let (mut best, mut best_ref) = (0usize, 0usize);
        let (mut bv, mut bvr) = (f32::MIN, f32::MIN);
        for j in 0..v {
            let want = rd_f32(tv, base + 4 * j);
            let d = (logits[j] - want).abs();
            max_d = max_d.max(d);
            sum_d += d as f64;
            if logits[j] > bv {
                bv = logits[j];
                best = j;
            }
            if want > bvr {
                bvr = want;
                best_ref = j;
            }
        }
        if best == best_ref {
            agree += 1;
        }
    }
    let ms = t0.elapsed().as_millis();
    println!(
        "tokens: {}  time: {} ms ({} ms/token)",
        n,
        ms,
        ms / n as u128
    );
    println!("argmax agreement: {}/{}", agree, n);
    println!(
        "max |diff|: {:.5}  mean |diff|: {:.6}",
        max_d,
        sum_d / (n * v) as f64
    );
    Ok(())
}
fn run_command(command: &str) {
    match command {
        "help" => {
            println!("Available commands:");
            println!("  help  - Show commands");
            println!("  ping  - Test connection");
            println!("  info  - Device information");
            println!("  models  - Show model information");
            println!("  load  - Parse model and dump fields");
            println!("  mem   - Show heap memory usage");
            println!("  test  - Run test vectors and report accuracy");
            println!("  exit  - Informational exit");
        }
        "ping" => println!("connected"),
        "model" => match model() {
            Ok(m) if &m[0..4] == b"BTLM" => println!(
                "BTLM v{} | vocab {} | ctx {} | layers {} | heads {} | dim {} | hidden {} | theta {} | eps {:e}",
                rd_u32(m, 4), rd_u32(m, 8), rd_u32(m, 12), rd_u32(m, 16),
                rd_u32(m, 20), rd_u32(m, 24), rd_u32(m, 28), rd_f32(m, 32), rd_f32(m, 36),
            ),
            Ok(m) => println!("bad magic: {:02x?}", &m[0..4]),
            Err(e) => println!("model error: {}", e),
        },
        "info" => {
            println!("Device: ESP32-S3");
            println!("Firmware: Rust");
            println!("Transport: USB Serial/JTAG");
        }
        "exit" => println!("Firmware keeps running."),
        "load" => match model().and_then(engine::Model::parse) {
            Ok((m, used)) => {
                let (l0, l5) = (&m.layers[0], &m.layers[5]);
                println!("parsed {} bytes (file is 1773288)", used);
                println!("emb_scale[0]={:e} emb_scale[2047]={:e} emb[0..4]={:?}", m.emb_scale[0], m.emb_scale[2047], &m.emb[0..4]);
                println!("final_norm[0]={:e}", m.final_norm[0]);
                println!("L0 attn_norm[0]={:e} down_norm[0]={:e} wq.gamma={:e} wq.packed[0..4]={:?}", l0.attn_norm[0], l0.down_norm[0], l0.wq.gamma, &l0.wq.packed[0..4]);
                println!("L5 w_down.gamma={:e} rows={} cols={}", l5.w_down.gamma, l5.w_down.rows, l5.w_down.cols);
            }
            Err(e) => println!("parse error: {}", e),
        },
        "mem" => unsafe {
            println!("internal free: {} bytes", sys::heap_caps_get_free_size(sys::MALLOC_CAP_INTERNAL));
            println!("psram free:    {} bytes", sys::heap_caps_get_free_size(sys::MALLOC_CAP_SPIRAM));
        },
        "test" => match run_test() {
            Ok(()) => {}
            Err(e) => println!("test error: {}", e),
        },
        "" => {}
        _ => {
            println!("Unknown command: {}", command);
            println!("Type 'help' for commands.");
        }
    }
}

fn main() {
    // Apply ESP-IDF patches required by esp-idf-svc.
    esp_idf_svc::sys::link_patches();

    println!();
    println!("==============================");
    println!("   ESP32-S3 Rust Firmware");
    println!("==============================");
    println!("USB Serial/JTAG initialized.");
    println!("Type 'help' for available commands.");
    println!();

    let mut stdin = io::stdin().lock();
    let mut stdout = io::stdout();

    // Console stdin on ESP-IDF is non-blocking (reads fail with EAGAIN when
    // nothing was typed), and the terminal does not echo. So we read one byte
    // at a time and echo it ourselves.
    let mut line = String::new();
    let mut last_was_cr = false;
    prompt(&mut stdout);

    let mut byte = [0u8; 1];
    loop {
        match stdin.read(&mut byte) {
            Ok(1) => {
                let c = byte[0];
                let was_cr = last_was_cr;
                last_was_cr = c == b'\r';
                match c {
                    // Enter. Swallow the '\n' of a "\r\n" pair.
                    b'\n' if was_cr => {}
                    b'\r' | b'\n' => {
                        println!();
                        run_command(line.trim());
                        line.clear();
                        prompt(&mut stdout);
                    }
                    // Backspace / Delete: remove the last character.
                    0x08 | 0x7f => {
                        if line.pop().is_some() {
                            print!("\x08 \x08");
                            let _ = stdout.flush();
                        }
                    }
                    // Printable ASCII: store and echo.
                    0x20..=0x7e => {
                        if line.len() < MAX_LINE {
                            line.push(c as char);
                            print!("{}", c as char);
                            let _ = stdout.flush();
                        }
                    }
                    _ => {}
                }
            }
            // Nothing available right now (or a zero-byte read): wait a moment.
            Ok(_) => sleep(Duration::from_millis(10)),
            Err(e) if matches!(e.kind(), ErrorKind::WouldBlock | ErrorKind::Interrupted) => {
                sleep(Duration::from_millis(10));
            }
            // A real error: report it without flooding the console.
            Err(e) => {
                println!("Input error: {}", e);
                sleep(Duration::from_millis(500));
            }
        }
    }
}
