//! WiFi smoke test.
//!
//! Brings up a SoftAP and serves one page, to answer three questions before we
//! build anything bigger:
//!
//!   1. does the radio come up at all with octal PSRAM at 80 MHz?
//!   2. how much internal RAM does it cost?
//!   3. does `gen` still work while the AP is running, and at what speed?
//!
//! The AP is one-shot: bring it up with `wifi`, and reset the board to get back
//! to a wired-only build. There is no `wifi off` on purpose -- tearing down a
//! live httpd + netif pair is a failure mode we do not need during a smoke test.

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

const SSID: &str = "Mini-Tritty";

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

    print_heap("after");

    let mut server =
        EspHttpServer::new(&HttpConfig::default()).map_err(|e| format!("httpd: {}", e))?;
    server
        .fn_handler("/", Method::Get, |req| {
            let (i, p) = heap_free();
            let body = format!(
                "<!doctype html><meta name=viewport content=\"width=device-width,initial-scale=1\">\
                 <body style=\"font:16px system-ui;padding:2rem;line-height:1.6\">\
                 <h1>Mini-Tritty is up</h1>\
                 <p>This page is served by the ESP32-S3 itself, over its own access point.</p>\
                 <p>internal {i} B free &middot; psram {p} B free</p>\
                 </body>"
            );
            req.into_ok_response()?.write_all(body.as_bytes())
        })
        .map_err(|e| format!("route: {}", e))?;

    // The server stops when dropped, so it has to outlive this call.
    std::mem::forget(server);

    println!("ssid: {} (open, no password)", SSID);
    println!("open: http://{}", ip.ip);
    println!("now run `gen Lily is` and compare ms/token against the wired run.");

    *slot = Some(wifi);
    Ok(())
}
