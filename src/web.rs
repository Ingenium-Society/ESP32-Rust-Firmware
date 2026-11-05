//! SoftAP, captive portal and the visitor-facing page.
//!
//! Starts at boot so the board runs standalone off a USB charger. Visitors join
//! `Mini-Trits`; a DNS responder answers every query with the AP's own address,
//! so the phone's captive-portal probe lands on us and offers to open the page.
//! From there they type a prompt and watch the story stream in.
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

const SSID: &str = "Mini-Trits";

/// No output cap: the story runs until the model emits EOS or the context fills.
/// The context (256 tokens) is the only bound -- roughly 25 s typical, 55 s worst
/// case at 4.6 tok/s.
const MAX_NEW_TOKENS: usize = usize::MAX;

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

pub fn status() {
    println!(
        "wifi: {}",
        if AP.lock().unwrap().is_some() {
            "up"
        } else {
            "down"
        }
    );
}

/// Report status, bringing the AP up if boot-time start failed.
pub fn ensure() {
    if AP.lock().unwrap().is_some() {
        status();
        return;
    }
    if let Err(e) = start() {
        println!("wifi error: {}", e);
    }
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
        .map_err(|e| format!("ip: {}", e))?
        .ip
        .octets();

    let mut server = EspHttpServer::new(&HttpConfig {
        max_open_sockets: 4,
        max_uri_handlers: 4,
        stack_size: 8192,
        uri_match_wildcard: true,
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

            // Returns at once. The story is not sent down this connection -- every
            // client reads it from /state, so they all see the same thing and no
            // socket is tied up for 30 seconds.
            match job::submit_generate(prompt, MAX_NEW_TOKENS, random_seed()) {
                Ok(_rx) => req
                    .into_response(202, Some("Accepted"), &[("Content-Type", "text/plain")])?
                    .write_all(b"queued"),
                Err(e) => req
                    .into_response(
                        503,
                        Some("Busy"),
                        &[("Content-Type", "text/plain; charset=utf-8")],
                    )?
                    .write_all(e.as_bytes()),
            }
        })
        .map_err(|e| format!("route /gen: {}", e))?;

    server
        .fn_handler("/state", Method::Get, |req| {
            let s = job::snapshot();
            let body = format!(
                "{{\"id\":{},\"generating\":{},\"queued\":{},\"full\":{},\"prompt\":\"{}\",\"text\":\"{}\"}}",
                s.id,
                s.generating,
                s.queued,
                s.full(),
                json_escape(&s.prompt),
                json_escape(&s.text)
            );
            req.into_response(
                200,
                Some("OK"),
                &[
                    ("Content-Type", "application/json"),
                    ("Cache-Control", "no-store"),
                ],
            )?
            .write_all(body.as_bytes())
        })
        .map_err(|e| format!("route /state: {}", e))?;

    // Everything else is a captive-portal probe (captive.apple.com,
    // connectivitycheck.gstatic.com, ...). Bounce it to the page: that is what
    // makes the phone offer "Sign in to network".
    let home = format!("http://{}.{}.{}.{}/", ip[0], ip[1], ip[2], ip[3]);
    let target = home.clone();
    server
        .fn_handler("/*", Method::Get, move |req| {
            req.into_response(302, Some("Found"), &[("Location", target.as_str())])?
                .write_all(b"")
        })
        .map_err(|e| format!("route /*: {}", e))?;

    // The server stops when dropped, so it has to outlive this call.
    std::mem::forget(server);

    spawn_dns(ip);

    print_heap("after");
    println!("ssid: {} (open, no password)", SSID);
    println!("page: {home}");
    println!(
        "one shared story, {} prompt slot, captive portal on",
        job::QUEUE_DEPTH
    );

    *slot = Some(wifi);
    Ok(())
}

/// Minimal JSON string escaping. Generated prose is full of quotes.
fn json_escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 8);
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push(' '),
            c => out.push(c),
        }
    }
    out
}

/// Answer every DNS query with the AP's address.
///
/// DHCP already hands clients this device as their DNS server
/// (CONFIG_LWIP_DHCPS_ADD_DNS), so a phone joining the AP resolves the captive
/// portal probe hostnames to us, gets our 302, and shows the sign-in sheet.
fn spawn_dns(ip: [u8; 4]) {
    let _ = std::thread::Builder::new()
        .stack_size(8 * 1024)
        .spawn(move || unsafe {
            let sock = sys::lwip_socket(
                sys::AF_INET as i32,
                sys::SOCK_DGRAM as i32,
                sys::IPPROTO_UDP as i32,
            );
            if sock < 0 {
                println!("dns: socket failed");
                return;
            }

            let mut addr: sys::sockaddr_in = std::mem::zeroed();
            addr.sin_len = std::mem::size_of::<sys::sockaddr_in>() as u8;
            addr.sin_family = sys::AF_INET as u8;
            addr.sin_port = 53u16.to_be();
            addr.sin_addr.s_addr = u32::from_ne_bytes(ip);

            let rc = sys::lwip_bind(
                sock,
                &addr as *const _ as *const sys::sockaddr,
                std::mem::size_of::<sys::sockaddr_in>() as sys::socklen_t,
            );
            if rc != 0 {
                println!("dns: bind failed");
                return;
            }

            let mut query = [0u8; 512];
            loop {
                let mut from: sys::sockaddr_in = std::mem::zeroed();
                let mut fromlen = std::mem::size_of::<sys::sockaddr_in>() as sys::socklen_t;
                let n = sys::lwip_recvfrom(
                    sock,
                    query.as_mut_ptr() as *mut _,
                    query.len(),
                    0,
                    &mut from as *mut _ as *mut sys::sockaddr,
                    &mut fromlen,
                );
                if n <= 0 {
                    continue;
                }
                if let Some(reply) = dns_reply(&query[..n as usize], ip) {
                    sys::lwip_sendto(
                        sock,
                        reply.as_ptr() as *const _,
                        reply.len(),
                        0,
                        &from as *const _ as *const sys::sockaddr,
                        fromlen,
                    );
                }
            }
        });
}

/// Turn a query into a reply. A records get our address; everything else gets an
/// empty NOERROR so the resolver falls back to A instead of retrying.
fn dns_reply(query: &[u8], ip: [u8; 4]) -> Option<Vec<u8>> {
    if query.len() < 12 {
        return None;
    }

    // Walk the QNAME labels to find where the question ends.
    let mut i = 12;
    loop {
        let len = *query.get(i)? as usize;
        i += 1;
        if len == 0 {
            break;
        }
        if len & 0xC0 != 0 {
            return None; // compression pointers do not belong in a question
        }
        i = i.checked_add(len)?;
        if i > query.len() {
            return None;
        }
    }
    if i + 4 > query.len() {
        return None;
    }

    let qtype = u16::from_be_bytes([query[i], query[i + 1]]);
    let end = i + 4;
    let answer = qtype == 1; // A

    let mut out = Vec::with_capacity(end + 16);
    out.extend_from_slice(&query[0..2]); // transaction id
    out.extend_from_slice(&[0x81, 0x80]); // response, recursion available
    out.extend_from_slice(&[0x00, 0x01]); // one question
    out.extend_from_slice(&if answer { [0x00u8, 0x01] } else { [0x00u8, 0x00] });
    out.extend_from_slice(&[0x00, 0x00, 0x00, 0x00]); // no authority, no additional
    out.extend_from_slice(&query[12..end]);

    if answer {
        out.extend_from_slice(&[
            0xC0, 0x0C, // name: pointer to offset 12
            0x00, 0x01, // type A
            0x00, 0x01, // class IN
            0x00, 0x00, 0x00, 0x3C, // ttl 60
            0x00, 0x04, // rdlength
        ]);
        out.extend_from_slice(&ip);
    }

    Some(out)
}
