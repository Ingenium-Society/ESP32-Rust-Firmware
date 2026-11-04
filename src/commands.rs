//! Command dispatch for the serial console. One function per command family,
//! all reachable from `run`.

use esp_idf_svc::sys;

use crate::engine;
use crate::flash::{self, rd_f32, rd_u32};
use crate::generate;
use crate::runner;
use crate::selfcheck;
use crate::tokenizer;

fn gen_cmd(prompt: &str) -> Result<(), String> {
    let (m, _) = engine::Model::parse(flash::model()?)?;
    let ctx = m.cfg.ctx;
    let tok = tokenizer::Tokenizer::parse(flash::TOKENIZER_BIN)?;
    if tok.vocab() != m.cfg.vocab {
        return Err("tokenizer and model vocab sizes differ".into());
    }
    let mut r = runner::Runner::new(m);
    let seed = unsafe { ((sys::esp_random() as u64) << 32) | sys::esp_random() as u64 };
    generate::generate(&mut r, &tok, prompt, 200, seed, ctx)
}

pub fn run(command: &str) {
    if command == "gen" || command.starts_with("gen ") {
        if let Err(e) = gen_cmd(command[3..].trim()) {
            println!("gen error: {}", e);
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
            println!("  exit  - Informational exit");
        }
        "ping" => println!("connected"),
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
        "test" => match selfcheck::run() {
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
