use std::io::{self, Write};

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

    let stdin = io::stdin();
    let mut stdout = io::stdout();

    let mut input = String::new();

    loop {
        input.clear();

        print!("esp32> ");
        let _ = stdout.flush();

        match stdin.read_line(&mut input) {
            Ok(0) => continue,

            Ok(_) => {
                let command = input.trim();

                match command {
                    "help" => {
                        println!("Available commands:");
                        println!("  help  - Show commands");
                        println!("  ping  - Test connection");
                        println!("  info  - Device information");
                        println!("  exit  - Informational exit");
                    }

                    "ping" => {
                        println!("connected");
                    }

                    "info" => {
                        println!("Device: ESP32-S3");
                        println!("Firmware: Rust");
                        println!("Transport: USB Serial/JTAG");
                    }

                    "exit" => {
                        println!("Firmware keeps running.");
                    }

                    "" => {}

                    _ => {
                        println!("Unknown command: {}", command);
                        println!("Type 'help' for commands.");
                    }
                }
            }

            Err(error) => {
                println!("Input error: {}", error);
            }
        }
    }
}
