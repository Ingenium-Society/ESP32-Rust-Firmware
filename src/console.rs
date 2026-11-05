//! Line-based serial console.
//!
//! ESP-IDF's console stdin is non-blocking (reads fail with EAGAIN when nothing
//! was typed) and the terminal does not echo, so this reads one byte at a time
//! and does the line editing itself.

use std::io::{self, ErrorKind, Read, Write};
use std::thread::sleep;
use std::time::Duration;

use crate::commands;

const MAX_LINE: usize = 128;

fn prompt(stdout: &mut io::Stdout) {
    print!("esp32> ");
    let _ = stdout.flush();
}

pub fn run() {
    let mut stdin = io::stdin().lock();
    let mut stdout = io::stdout();

    let mut line = String::new();
    let mut last_was_cr = false;
    prompt(&mut stdout);

    let mut byte = [0u8; 1];
    let mut errors = 0usize;
    loop {
        match stdin.read(&mut byte) {
            Ok(1) => {
                errors = 0;
                let c = byte[0];
                let was_cr = last_was_cr;
                last_was_cr = c == b'\r';
                match c {
                    // Enter. Swallow the '\n' of a "\r\n" pair.
                    b'\n' if was_cr => {}
                    b'\r' | b'\n' => {
                        println!();
                        commands::run(line.trim());
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
                    0x20..=0x7e
                        if line.len() < MAX_LINE => {
                            line.push(c as char);
                            print!("{}", c as char);
                            let _ = stdout.flush();
                        }
                    _ => {}
                }
            }
            // Nothing available right now (or a zero-byte read): wait a moment.
            Ok(_) => sleep(Duration::from_millis(10)),
            Err(e) if matches!(e.kind(), ErrorKind::WouldBlock | ErrorKind::Interrupted) => {
                sleep(Duration::from_millis(10));
            }
            // A real error. Running standalone off a charger there is no USB host,
            // so this repeats forever: report it once, then back right off.
            Err(e) => {
                errors += 1;
                if errors == 1 {
                    println!("Input error: {} (no console host? running headless)", e);
                }
                sleep(Duration::from_millis(if errors < 5 { 500 } else { 5000 }));
            }
        }
    }
}
