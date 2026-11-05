//! Mini-Tritty firmware: boots, prints a banner, starts the generation task and
//! hands the console to the REPL. Everything else lives in the modules below.
//!
//! engine     BTLM file parser (borrowed view over flash/PSRAM)
//! runner     forward pass with a KV cache, split across both cores
//! generate   sampling and the token loop
//! tokenizer  byte-level BPE
//! flash      the model partition and the assets compiled into the firmware
//! selfcheck  the `test` command: engine logits vs PyTorch
//! job        the single generation task both front ends submit to
//! commands   serial command dispatch
//! console    the line editor and read loop
//! web        SoftAP and the visitor-facing page

use esp_idf_svc::sys;

mod commands;
mod console;
mod engine;
mod flash;
mod generate;
mod job;
mod runner;
mod selfcheck;
mod tokenizer;
mod web;

fn main() {
    // Apply ESP-IDF patches required by esp-idf-svc.
    sys::link_patches();

    println!();
    println!("==============================");
    println!("   ESP32-S3 Rust Firmware");
    println!("==============================");
    println!("USB Serial/JTAG initialized.");
    println!("Type 'help' for available commands.");
    println!();

    // Owns the only Runner: 2 MB model copy + 3 MB KV cache, done up front so the
    // first prompt is not the slow one.
    job::start();

    console::run();
}
