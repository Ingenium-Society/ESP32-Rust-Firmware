use std::io::{self, ErrorKind, Read, Write};
use std::thread::sleep;
use std::time::Duration;

const MAX_LINE: usize = 128;

fn prompt(stdout: &mut io::Stdout) {
    print!("esp32> ");
    let _ = stdout.flush();
}

fn run_command(command: &str) {
    match command {
        "help" => {
            println!("Available commands:");
            println!("  help  - Show commands");
            println!("  ping  - Test connection");
            println!("  info  - Device information");
            println!("  exit  - Informational exit");
        }
        "ping" => println!("connected"),
        "info" => {
            println!("Device: ESP32-S3");
            println!("Firmware: Rust");
            println!("Transport: USB Serial/JTAG");
        }
        "exit" => println!("Firmware keeps running."),
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
