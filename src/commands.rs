//! Command dispatch for the serial console. One function per command family,
//! all reachable from `run`.

use esp_idf_svc::sys;
use std::io::Write;
use std::sync::Mutex;

use crate::engine;
use crate::flash::{self, rd_f32, rd_u32};
use crate::job;
use crate::web;

/// Longest story the serial console will produce. The web page caps at 64 for a
/// booth; from the console we can afford to let it run longer.
const REPL_MAX_NEW_TOKENS: usize = 200;

/// Sampling seed for `gen`. `None` means "draw a fresh one from the hardware RNG
/// every run"; `Some` is used verbatim, so a run can be repeated token for token.
/// (A Mutex rather than an AtomicU64: Xtensa has no 64-bit atomics.)
static FIXED_SEED: Mutex<Option<u64>> = Mutex::new(None);

fn next_seed() -> u64 {
    match *FIXED_SEED.lock().unwrap() {
        Some(s) => s,
        None => unsafe { ((sys::esp_random() as u64) << 32) | sys::esp_random() as u64 },
    }
}

fn print_seed() {
    match *FIXED_SEED.lock().unwrap() {
        Some(s) => println!("seed: {} (fixed)", s),
        None => println!("seed: random (fresh every run)"),
    }
}

fn set_seed(seed: Option<u64>) {
    // Bind the guard to a statement so it is released before print_seed locks again.
    *FIXED_SEED.lock().unwrap() = seed;
    print_seed();
}

fn gen_cmd(prompt: &str) -> Result<(), String> {
    let rx = job::submit_generate(prompt.to_string(), REPL_MAX_NEW_TOKENS, next_seed())?;
    print!("{}", prompt);
    let _ = std::io::stdout().flush();
    for msg in rx {
        match msg {
            job::Msg::Text(bytes) => {
                print!("{}", String::from_utf8_lossy(&bytes));
                let _ = std::io::stdout().flush();
            }
            job::Msg::Generated(st) => {
                println!();
                println!(
                    "[prompt {} tokens in {} ms | generated {} tokens in {} ms = {:.1} tok/s | context {}/{} | seed {}]",
                    st.prompt_tokens,
                    st.prompt_ms,
                    st.generated,
                    st.gen_ms,
                    st.generated as f32 * 1000.0 / st.gen_ms as f32,
                    st.prompt_tokens + st.generated,
                    st.ctx,
                    st.seed
                );
            }
            job::Msg::Failed(e) => println!("\ngen error: {}", e),
        }
    }
    Ok(())
}

pub fn run(command: &str) {
    if command == "gen" || command.starts_with("gen ") {
        if let Err(e) = gen_cmd(command[3..].trim()) {
            println!("gen error: {}", e);
        }
        return;
    }
    // `seed <n>` / `seed random`. Kept out of the match below because the
    // argument is free-form.
    if let Some(arg) = command.strip_prefix("seed ") {
        let arg = arg.trim();
        if arg == "random" {
            set_seed(None);
        } else {
            match arg.parse::<u64>() {
                Ok(s) => set_seed(Some(s)),
                Err(_) => println!("seed error: expected 'seed <n>' or 'seed random'"),
            }
        }
        return;
    }
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
            println!("  gen   - Generate a story: gen <prompt>");
            println!("  seed  - Show or fix the sampling seed: seed <n> | seed random");
            println!("  wifi  - Show access point status");
            println!("  exit  - Informational exit");
        }
        "ping" => println!("connected"),
        "wifi" => web::ensure(),
        "seed" => print_seed(),
        "model" => match flash::model() {
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
        "load" => match flash::model().and_then(engine::Model::parse) {
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
        "test" => match job::submit_selfcheck() {
            Ok(Ok(())) => {}
            Ok(Err(e)) => println!("test error: {}", e),
            Err(e) => println!("test error: {}", e),
        },
        "" => {}
        _ => {
            println!("Unknown command: {}", command);
            println!("Type 'help' for commands.");
        }
    }
}
