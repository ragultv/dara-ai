//! Live check of OAuth discovery against real servers (no credentials, no browser):
//! `cargo run -p dara-mcp --example probe_oauth -- https://calendarmcp.googleapis.com/mcp/v1 ...`
use dara_mcp::oauth::discover;

#[tokio::main]
async fn main() {
    let urls: Vec<String> = std::env::args().skip(1).collect();
    for u in urls {
        let url = match url::Url::parse(&u) {
            Ok(x) => x,
            Err(e) => {
                println!("{u}\n   not a URL: {e}");
                continue;
            }
        };
        println!("{u}");
        match discover(&url, None).await {
            Ok(d) => println!("   resource: {}\n   {:?}", d.resource, d.server),
            Err(e) => println!("   FAILED: {e}"),
        }
    }
}
