//! Mini-Tritty firmware: boots, prints a banner, then hands the console to the
//! REPL. Everything else lives in the modules below.
//!
//! engine     BTLM file parser (borrowed view over flash/PSRAM)
//! runner     forward pass with a KV cache, split across both cores
//! generate   sampling and the token loop
//! tokenizer  byte-level BPE
//! flash      the model partition and the assets compiled into the firmware
//! selfcheck  the `test` command: engine logits vs PyTorch
//! commands   serial command dispatch
//! console    the line editor and read loop

use esp_idf_svc::sys;

mod commands;
mod console;
mod engine;
mod flash;
mod generate;
mod runner;
mod selfcheck;
mod tokenizer;

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

    console::run();
}
