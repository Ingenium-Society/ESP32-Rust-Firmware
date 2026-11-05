//! SoftAP + the visitor-facing page.
//!
//! `wifi` brings up an open access point and an HTTP server. Visitors join
//! `Mini-Tritty`, open http://192.168.4.1, type a prompt and watch the story
//! stream in as the chip produces it.
//!
//! The AP is one-shot: reset the board to get back to a wired-only build. There
//! is no `wifi off` on purpose -- tearing down a live httpd + netif pair is a
//! failure mode we do not need at a booth.
//!
//! All the arithmetic happens in the generation task (see `job`); these handlers
//! only move bytes.

use esp_idf_svc::eventloop::EspSystemEventLoop;
use esp_idf_svc::hal::peripherals::Peripherals;
use esp_idf_svc::http::server::{Configuration as HttpConfig, EspHttpServer};
use esp_idf_svc::http::Method;
use esp_idf_svc::io::Write;
use esp_idf_svc::nvs::EspDefaultNvsPartition;
use esp_idf_svc::sys;
use esp_idf_svc::wifi::{
    AccessPointConfiguration, AuthMethod, BlockingWifi, Configuration, EspWifi,
};
use std::sync::Mutex;

use crate::job;

const SSID: &str = "Mini-Tritty";

/// One story at a booth. 64 tokens is ~14 s at 4.6 tok/s; 200 would be 40 s.
const MAX_NEW_TOKENS: usize = 64;

/// Longest prompt we accept, in bytes.
const MAX_PROMPT: usize = 200;

const PAGE: &str = include_str!("page.html");

type Ap = BlockingWifi<EspWifi<'static>>;

static AP: Mutex<Option<Ap>> = Mutex::new(None);

fn heap_free() -> (usize, usize) {
    unsafe {
        (
            sys::heap_caps_get_free_size(sys::MALLOC_CAP_INTERNAL),
            sys::heap_caps_get_free_size(sys::MALLOC_CAP_SPIRAM),
        )
    }
}

fn print_heap(label: &str) {
    let (i, p) = heap_free();
    println!("  {:<7} internal {} B | psram {} B", label, i, p);
}

fn random_seed() -> u64 {
    unsafe { ((sys::esp_random() as u64) << 32) | sys::esp_random() as u64 }
}

pub fn start() -> Result<(), String> {
    let mut slot = AP.lock().unwrap();
    if slot.is_some() {
        return Err("wifi is already up".into());
    }

    print_heap("before");

    let peripherals = Peripherals::take().map_err(|e| format!("peripherals: {}", e))?;
    let sysloop = EspSystemEventLoop::take().map_err(|e| format!("event loop: {}", e))?;
    let nvs = EspDefaultNvsPartition::take().ok();

    let driver = EspWifi::new(peripherals.modem, sysloop.clone(), nvs)
        .map_err(|e| format!("driver: {}", e))?;
    let mut wifi = BlockingWifi::wrap(driver, sysloop).map_err(|e| format!("wrap: {}", e))?;

    let ap = AccessPointConfiguration {
        ssid: SSID.try_into().map_err(|_| "ssid too long")?,
        ssid_hidden: false,
        channel: 1,
        secondary_channel: None,
        protocols: Default::default(),
        auth_method: AuthMethod::None,
        password: "".try_into().map_err(|_| "password too long")?,
        max_connections: 4,
    };

    wifi.set_configuration(&Configuration::AccessPoint(ap))
        .map_err(|e| format!("ap config: {}", e))?;
    wifi.start().map_err(|e| format!("ap start: {}", e))?;

    let ip = wifi
        .wifi()
        .ap_netif()
        .get_ip_info()
        .map_err(|e| format!("ip: {}", e))?;

    let mut server = EspHttpServer::new(&HttpConfig {
        max_open_sockets: 4,
        max_uri_handlers: 4,
        stack_size: 8192,
        ..Default::default()
    })
    .map_err(|e| format!("httpd: {}", e))?;

    server
        .fn_handler("/", Method::Get, |req| {
            req.into_response(
                200,
                Some("OK"),
                &[("Content-Type", "text/html; charset=utf-8")],
            )?
            .write_all(PAGE.as_bytes())
        })
        .map_err(|e| format!("route /: {}", e))?;

    server
        .fn_handler("/gen", Method::Post, |mut req| {
            let mut buf = [0u8; MAX_PROMPT];
            let mut len = 0;
            while len < buf.len() {
                match req.read(&mut buf[len..]) {
                    Ok(0) => break,
                    Ok(n) => len += n,
                    Err(_) => break,
                }
            }
            let prompt = String::from_utf8_lossy(&buf[..len]).trim().to_string();

            if prompt.is_empty() {
                return req
                    .into_response(400, Some("Bad Request"), &[("Content-Type", "text/plain")])?
                    .write_all(b"Empty prompt.");
            }

            let rx = match job::submit_generate(prompt, MAX_NEW_TOKENS, random_seed()) {
                Ok(rx) => rx,
                Err(e) => {
                    return req
                        .into_response(
                            503,
                            Some("Busy"),
                            &[("Content-Type", "text/plain; charset=utf-8")],
                        )?
                        .write_all(e.as_bytes());
                }
            };

            // No Content-Length: every write goes out as a chunk, so the browser
            // shows text as the chip produces it.
            let mut resp = req.into_response(
                200,
                Some("OK"),
                &[("Content-Type", "text/plain; charset=utf-8")],
            )?;

            for msg in rx {
                match msg {
                    job::Msg::Text(bytes) => resp.write_all(&bytes)?,
                    job::Msg::Failed(e) => {
                        resp.write_all(format!("\n[error: {}]", e).as_bytes())?;
                        break;
                    }
                    job::Msg::Generated(_) => {}
                }
            }
            Ok(())
        })
        .map_err(|e| format!("route /gen: {}", e))?;

    // The server stops when dropped, so it has to outlive this call.
    std::mem::forget(server);

    print_heap("after");
    println!("ssid: {} (open, no password)", SSID);
    println!("page: http://{}", ip.ip);
    println!("{} tokens per story, {} prompt slots", MAX_NEW_TOKENS, job::QUEUE_DEPTH);

    *slot = Some(wifi);
    Ok(())
}
