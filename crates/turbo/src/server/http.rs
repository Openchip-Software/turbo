// SSE Example code (MIT licence) from the Axum SSE example
// Harry added OC performance data/structs etc

use axum::{
    extract::{Path, State},
    http::{header, StatusCode},
    response::sse::{Event, Sse},
    response::{Html, IntoResponse, Response},
    routing::get,
    Router,
};
use futures_util::{stream::unfold, Stream};

use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use std::{convert::Infallible, path::PathBuf, time::Duration};
use tokio_stream::StreamExt as _;
use tower_http::cors::CorsLayer;
use tower_http::{services::ServeDir, trace::TraceLayer};

use crate::common::PerformanceData;
use crate::server::sse::{DataType, SseData};
use crate::RerfError;

/// Shared store of rendered annotation pages, keyed by logical filename
/// (e.g. `annotate_index.html`, `annotate.css`). Values are the **gzip-compressed**
/// page bytes, served verbatim with `Content-Encoding: gzip`. Populated by the
/// enrichment thread once the report is ready (or from disk in load mode) and
/// read by the `/annotate/{file}` handler per request.
pub type AnnotatePages = Arc<RwLock<HashMap<String, Vec<u8>>>>;

/// Shared state for the Axum server
#[derive(Clone)]
struct AppState {
    /// latest data to serve to SSE clients
    data_watch: tokio::sync::watch::Receiver<PerformanceData>,
    /// Broadcast channel for shutdown signal
    shutdown_broadcast: tokio::sync::broadcast::Sender<()>,
    /// In-memory annotation report pages served via `/annotate/{file}`.
    annotate_pages: AnnotatePages,
    /// Whether an annotation report is expected for this run. When false, the
    /// `/annotate/{file}` handler serves a terminal "not available" page instead
    /// of the auto-refreshing "pending" page (which would otherwise wait forever).
    annotate_enabled: bool,
    /// Output/work directory, used by `/roi-flamegraph` to read
    /// `roi_flamegraph.html` directly off disk once it's written.
    output_path: Option<PathBuf>,
    /// Whether a ROI flamegraph is expected for this run (`--roi-flamegraph`
    /// was passed). Mirrors `annotate_enabled`'s pending/disabled distinction.
    roi_flamegraph_enabled: bool,
}

/// Server sink that serves aggregated data via Server-Sent Events
///
/// Receives Vec<RoiInfo> containing aggregated performance data and serves
/// them to connected web clients via SSE.
pub struct HttpServer {
    data_watch: tokio::sync::watch::Receiver<PerformanceData>,
    port: u16,
    assets_path: Option<PathBuf>,
    /// Output/work directory, served under `/out/` so the browser can reach
    /// generated artifacts such as the `annotate_*.html` report.
    output_path: Option<PathBuf>,
    /// In-memory annotation report pages served under `/annotate/{file}`.
    annotate_pages: AnnotatePages,
    /// Whether an annotation report is expected (see [`AppState::annotate_enabled`]).
    annotate_enabled: bool,
    /// Whether a ROI flamegraph is expected (see [`AppState::roi_flamegraph_enabled`]).
    roi_flamegraph_enabled: bool,
}

pub const DEFAULT_PORT: u16 = 9829;

impl HttpServer {
    /// Create new server sink with default port
    pub fn new(data_watch: tokio::sync::watch::Receiver<PerformanceData>) -> Self {
        HttpServer {
            data_watch,
            port: DEFAULT_PORT,
            assets_path: None,
            output_path: None,
            annotate_pages: Arc::new(RwLock::new(HashMap::new())),
            annotate_enabled: false,
            roi_flamegraph_enabled: true,
        }
    }

    /// Declare whether an annotation report is expected for this run, so the
    /// `/annotate/{file}` handler can distinguish "still rendering" (pending,
    /// auto-refresh) from "not enabled / none available" (terminal page).
    pub fn annotate_enabled(mut self, enabled: bool) -> Self {
        self.annotate_enabled = enabled;
        self
    }

    /// Declare whether a ROI flamegraph is expected for this run, so the
    /// `/roi-flamegraph` handler can distinguish "still rendering" from
    /// "not enabled / none available".
    pub fn roi_flamegraph_enabled(mut self, enabled: bool) -> Self {
        self.roi_flamegraph_enabled = enabled;
        self
    }

    /// Set custom assets directory for static files
    pub fn assets_path<P: Into<PathBuf>>(mut self, path: P) -> Self {
        self.assets_path = Some(path.into());
        self
    }

    /// Share the in-memory annotation page store. The enrichment thread fills
    /// this map once the report is rendered; the server reads it per request.
    pub fn annotate_pages(mut self, pages: AnnotatePages) -> Self {
        self.annotate_pages = pages;
        self
    }

    /// Set the output/work directory, served under `/out/` for generated
    /// artifacts (e.g. the `annotate_*.html` source & assembly report).
    pub fn output_path<P: Into<PathBuf>>(mut self, path: P) -> Self {
        self.output_path = Some(path.into());
        self
    }

    pub fn run(self) -> Result<(), RerfError> {
        log::info!("Starting ServerSink on port {}", self.port);

        // Create a tokio runtime for the server
        let runtime = match tokio::runtime::Builder::new_multi_thread()
            .thread_name("server-worker")
            .worker_threads(2)
            .enable_io()
            .enable_time()
            .build()
        {
            Ok(rt) => rt,
            Err(e) => {
                log::error!("Failed to create tokio runtime: {}", e);
                return Err(RerfError::server_bind_failed(format!(
                    "Failed to create tokio runtime: {}",
                    e
                )));
            }
        };

        // Run the HTTP server
        let result = runtime.block_on(async {
            // Create broadcast channel for shutdown
            let (shutdown_tx, _) = tokio::sync::broadcast::channel(1);

            // Build a state for serving each request/SSE async event
            let app_state = AppState {
                data_watch: self.data_watch,
                shutdown_broadcast: shutdown_tx,
                annotate_pages: self.annotate_pages,
                annotate_enabled: self.annotate_enabled,
                output_path: self.output_path.clone(),
                roi_flamegraph_enabled: self.roi_flamegraph_enabled,
            };

            // Set up static file serving
            // Try multiple locations: custom path, production install, dev path
            // Resolution (packaged path, else the in-tree one) lives in
            // `crate::assets`, which the annotation report also reads its
            // stylesheet and scripts through -- one directory, resolved once, so
            // the dashboard and the reports can never disagree about where the
            // assets are. A custom path is registered there for the same reason.
            let assets_path = if let Some(ref custom_path) = self.assets_path {
                crate::assets::set_dir(custom_path.clone());
                custom_path.clone()
            } else {
                crate::assets::dir().to_path_buf()
            };
            log::info!("Using assets path: {}", assets_path.display());

            // Log assets path and verify it exists
            log::trace!("========================================");
            log::trace!("Assets path resolved to: {}", assets_path.display());
            log::trace!("Assets path exists: {}", assets_path.exists());
            log::trace!("Assets path is_dir: {}", assets_path.is_dir());

            if assets_path.exists() {
                if let Ok(entries) = std::fs::read_dir(&assets_path) {
                    log::trace!("Assets directory contents:");
                    for entry in entries.flatten() {
                        let path = entry.path();
                        let file_type = if path.is_dir() { "DIR" } else { "FILE" };
                        log::trace!(
                            "  - {} [{}]",
                            entry.file_name().to_string_lossy(),
                            file_type
                        );
                    }
                }

                // Check for index.html specifically
                let index_path = assets_path.join("index.html");
                log::trace!("index.html path: {}", index_path.display());
                log::trace!("index.html exists: {}", index_path.exists());
            } else {
                log::error!("Assets directory does NOT exist!");
            }
            log::trace!("========================================");

            let static_service =
                ServeDir::new(assets_path.clone()).append_index_html_on_directories(true);

            // Serve the output/work directory under /out/ so the frontend can
            // open generated artifacts (e.g. annotate_*.html) directly.
            let out_service = self
                .output_path
                .clone()
                .map(|p| ServeDir::new(p).append_index_html_on_directories(true));

            // Build router with CORS
            let cors = CorsLayer::new()
                .allow_origin([
                    "http://localhost:9829".parse().unwrap(),
                    "http://127.0.0.1:9829".parse().unwrap(),
                ])
                .allow_methods([axum::http::Method::GET])
                .allow_headers([axum::http::header::CONTENT_TYPE]);

            log::info!("Building router with routes: /sse, /health, and static files");

            let router = match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                let mut router = Router::new()
                    .route("/sse", get(sse_handler))
                    .route("/health", get(|| async { "OK" }))
                    .route("/annotate/{file}", get(annotate_handler))
                    .route("/roi-flamegraph", get(roi_flamegraph_handler));
                if let Some(out_service) = out_service {
                    router = router.nest_service("/out", out_service);
                }
                router
                    .layer(cors)
                    .layer(TraceLayer::new_for_http())
                    .with_state(app_state)
                    .fallback_service(static_service)
            })) {
                Ok(router) => router,
                Err(e) => {
                    log::error!("Panic while building router!");
                    let msg = if let Some(s) = e.downcast_ref::<&str>() {
                        log::error!("Router panic: {}", s);
                        format!("Failed to build router: {}", s)
                    } else if let Some(s) = e.downcast_ref::<String>() {
                        log::error!("Router panic: {}", s);
                        format!("Failed to build router: {}", s)
                    } else {
                        log::error!("Router panic with unknown error type");
                        "Failed to build router with unknown error".to_string()
                    };
                    return Err(RerfError::server_bind_failed(msg));
                }
            };

            log::info!("Router built successfully");

            // Bind and serve
            let addr = format!("127.0.0.1:{}", self.port);
            log::info!("Performance server running at http://{}", addr);

            let listener = match tokio::net::TcpListener::bind(&addr).await {
                Ok(listener) => listener,
                Err(e) => {
                    log::error!("Failed to bind to address {}: {}", addr, e);
                    return Err(RerfError::server_bind_failed(format!(
                        "Failed to bind to address {}: {}",
                        addr, e
                    )));
                }
            };

            log::info!("Successfully bound to {}", addr);
            if let Err(e) = axum::serve(listener, router).await {
                log::error!("Server error: {}", e);
                return Err(RerfError::server_bind_failed(format!(
                    "Server failed: {}",
                    e
                )));
            }
            log::info!("server at {} shuts down now.", addr);
            Ok(())
        });

        log::info!("ServerSink completed");
        result
    }
}

/// Serve one rendered annotation page from the in-memory store.
///
/// The store is empty until the enrichment thread finishes and publishes the
/// report, so an empty store means the run is still in progress: we return a
/// friendly "available at end of run" page (200) rather than a bare 404, which
/// would read as a broken link. Once the report exists, an unknown filename is
/// a genuine 404.
async fn annotate_handler(State(app_state): State<AppState>, Path(file): Path<String>) -> Response {
    let (page, ready) = match app_state.annotate_pages.read() {
        Ok(m) => (m.get(&file).cloned(), !m.is_empty()),
        Err(_) => (None, false),
    };
    match page {
        // Stored gzip-compressed; serve the bytes verbatim with the matching
        // Content-Encoding and a Content-Type inferred from the extension
        // (.css/.js are linked as shared assets and must not be sniffed as HTML).
        Some(gz) => (
            [
                (header::CONTENT_TYPE, content_type_for(&file)),
                (header::CONTENT_ENCODING, "gzip"),
            ],
            gz,
        )
            .into_response(),
        // Annotate not enabled for this run: never promise a page that will
        // never arrive — serve a terminal page telling the user how to get one.
        None if !app_state.annotate_enabled => {
            (StatusCode::OK, Html(ANNOTATE_DISABLED_HTML)).into_response()
        }
        // Enabled but not yet published: the run is still rendering the report.
        None if !ready => (StatusCode::OK, Html(ANNOTATE_PENDING_HTML)).into_response(),
        None => (StatusCode::NOT_FOUND, "annotation page not found").into_response(),
    }
}

/// Serve the ROI flamegraph page, generated at the end of the run when
/// `--roi-flamegraph` is passed. Unlike annotation pages, the page is written
/// straight to disk (see `run_perf_workflow_opt`), so this handler reads it
/// off `output_path` directly rather than from an in-memory store.
async fn roi_flamegraph_handler(State(app_state): State<AppState>) -> Response {
    if !app_state.roi_flamegraph_enabled {
        return (StatusCode::OK, Html(ROI_FLAMEGRAPH_DISABLED_HTML)).into_response();
    }
    let Some(page_path) = app_state
        .output_path
        .as_ref()
        .map(|p| p.join("roi_flamegraph.html"))
    else {
        return (StatusCode::OK, Html(ROI_FLAMEGRAPH_DISABLED_HTML)).into_response();
    };
    match std::fs::read(&page_path) {
        Ok(page) => ([(header::CONTENT_TYPE, "text/html; charset=utf-8")], page).into_response(),
        Err(_) => (StatusCode::OK, Html(ROI_FLAMEGRAPH_PENDING_HTML)).into_response(),
    }
}

/// Placeholder shown when the ROI flamegraph is requested before the run has
/// finished generating it. Auto-refreshes so the real page appears once written.
const ROI_FLAMEGRAPH_PENDING_HTML: &str = r#"<!doctype html><html><head><meta charset="utf-8">
<title>flamegraph pending</title><meta http-equiv="refresh" content="3">
<style>body{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;background:#16161a;color:#d0d0d8;margin:0;display:flex;align-items:center;justify-content:center;height:100vh}div{text-align:center}</style>
</head><body><div><h1>ROI flamegraph will be available at end of run</h1>
<p>The flamegraph is generated when the run completes. This page refreshes automatically.</p></div></body></html>"#;

/// Shown when this run did not generate a ROI flamegraph (`--roi-flamegraph`
/// not passed). Static — no auto-refresh, since nothing will arrive.
const ROI_FLAMEGRAPH_DISABLED_HTML: &str = r#"<!doctype html><html><head><meta charset="utf-8">
<title>flamegraph not available</title>
<style>body{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;background:#16161a;color:#d0d0d8;margin:0;display:flex;align-items:center;justify-content:center;height:100vh}div{text-align:center;max-width:40rem;padding:1rem}code{color:#5ab3f0}</style>
</head><body><div><h1>ROI flamegraph not available</h1>
<p>This run did not generate a ROI flamegraph. Re-run with <code>--roi-flamegraph</code> to produce one.</p></div></body></html>"#;

/// Content-Type for an annotation artifact, by extension. The shared
/// `annotate.css` / `annotate.js` assets must be served with the correct MIME
/// (browsers refuse a stylesheet/module sent as `text/html`); everything else
/// is an HTML page.
fn content_type_for(file: &str) -> &'static str {
    if file.ends_with(".css") {
        "text/css; charset=utf-8"
    } else if file.ends_with(".js") {
        "text/javascript; charset=utf-8"
    } else {
        "text/html; charset=utf-8"
    }
}

/// Placeholder shown when an annotation page is requested before the run has
/// finished generating the report. Auto-refreshes so the real page appears once
/// the enrichment thread publishes it.
const ANNOTATE_PENDING_HTML: &str = r#"<!doctype html><html><head><meta charset="utf-8">
<title>annotation pending</title><meta http-equiv="refresh" content="3">
<style>body{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;background:#16161a;color:#d0d0d8;margin:0;display:flex;align-items:center;justify-content:center;height:100vh}div{text-align:center}</style>
</head><body><div><h1>Annotation will be available at end of run</h1>
<p>The source &amp; assembly report is generated when the run completes. This page refreshes automatically.</p></div></body></html>"#;

/// Shown when the source & assembly report was not generated for this run
/// (annotation disabled). Static — no auto-refresh, since nothing will arrive.
const ANNOTATE_DISABLED_HTML: &str = r#"<!doctype html><html><head><meta charset="utf-8">
<title>annotation not available</title>
<style>body{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;background:#16161a;color:#d0d0d8;margin:0;display:flex;align-items:center;justify-content:center;height:100vh}div{text-align:center;max-width:40rem;padding:1rem}code{color:#5ab3f0}</style>
</head><body><div><h1>Source &amp; assembly report not available</h1>
<p>This run did not generate an annotation report. Re-run without <code>--no-annotate</code>, or re-process the trace with <code>process --annotate</code>, to produce it.</p></div></body></html>"#;

/// SSE handler that streams aggregated data to clients
async fn sse_handler(
    State(app_state): State<AppState>,
) -> Sse<impl Stream<Item = Result<Event, Infallible>>> {
    log::trace!("SSE client connected");

    let fps = 10;
    let shutdown_rx = app_state.shutdown_broadcast.subscribe();
    let latest_data = app_state.data_watch.clone();

    let stream = unfold(shutdown_rx, move |mut shutdown_rx| {
        let mut latest_data = latest_data.clone();
        async move {
            // Check for shutdown signal
            if shutdown_rx.try_recv().is_ok() {
                log::debug!("SSE stream terminating due to shutdown signal");
                return None;
            }

            // Get latest data and convert to SSE format
            let _res = latest_data.changed().await;
            let data = latest_data.borrow_and_update();

            // drop all but the top 50 funcs by instruction count?
            let mut functions = data.functions.clone();
            functions.sort_by_key(|f| std::cmp::Reverse(f.instructions));
            functions.truncate(50);
            // funcs_sorted.first().inspect(|r| println!("{r:?}"));

            let sse_data = SseData::new(vec![
                DataType::Machine {
                    vlen_bits: data.vlen_bits,
                },
                DataType::Functions { functions },
                DataType::Rois {
                    rois: data.rois.clone(),
                },
            ]);
            let json_data = serde_json::to_string(&sse_data).expect("JSON encoding must succeed");
            // to inspect the HdrHistogram encoded data as JSON.. very verbose!
            // println!("json data =\n{}", json_data);

            Some((Ok(Event::default().data(json_data)), shutdown_rx))
        }
    })
    .throttle(Duration::from_millis(1000 / fps));

    Sse::new(stream).keep_alive(
        axum::response::sse::KeepAlive::new()
            .interval(Duration::from_millis(500))
            .text("keep-alive-text"),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn gz(s: &str) -> Vec<u8> {
        crate::processing::enricher::annotate::gzip(s.as_bytes())
    }

    fn state(pages: &[(&str, &str)]) -> AppState {
        let (tx, rx) = tokio::sync::watch::channel(PerformanceData::default());
        // Keep the sender alive for the lifetime of the receiver.
        std::mem::forget(tx);
        let (shutdown_tx, _) = tokio::sync::broadcast::channel(1);
        // The store holds gzipped bytes, as the live/load paths do.
        let map: HashMap<String, Vec<u8>> =
            pages.iter().map(|(k, v)| (k.to_string(), gz(v))).collect();
        AppState {
            data_watch: rx,
            shutdown_broadcast: shutdown_tx,
            annotate_pages: Arc::new(RwLock::new(map)),
            annotate_enabled: true,
            output_path: None,
            roi_flamegraph_enabled: false,
        }
    }

    /// Read a response body to a string, transparently gunzipping when the
    /// response is `Content-Encoding: gzip` (as annotation pages are).
    async fn body_of(resp: Response) -> String {
        let gzipped = resp
            .headers()
            .get(header::CONTENT_ENCODING)
            .is_some_and(|v| v == "gzip");
        let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        if gzipped {
            use std::io::Read as _;
            let mut d = flate2::read::GzDecoder::new(&bytes[..]);
            let mut s = String::new();
            d.read_to_string(&mut s).unwrap();
            s
        } else {
            String::from_utf8_lossy(&bytes).into_owned()
        }
    }

    /// Empty store while an enabled report is still rendering: the handler
    /// returns 200 with the auto-refreshing "available at end of run"
    /// placeholder, not a bare 404.
    #[tokio::test]
    async fn pending_while_run_in_progress() {
        let resp =
            annotate_handler(State(state(&[])), Path("annotate_index.html".to_string())).await;
        assert_eq!(resp.status(), StatusCode::OK);
        assert!(body_of(resp).await.contains("available at end of run"));
    }

    /// When annotation is not enabled for the run, an empty store must not show
    /// the auto-refreshing pending page (it would wait forever) — it serves the
    /// terminal "not available" page instead.
    #[tokio::test]
    async fn disabled_serves_terminal_page() {
        let mut st = state(&[]);
        st.annotate_enabled = false;
        let resp = annotate_handler(State(st), Path("annotate_index.html".to_string())).await;
        assert_eq!(resp.status(), StatusCode::OK);
        let body = body_of(resp).await;
        assert!(body.contains("not available"));
        assert!(!body.contains("refresh"), "must not auto-refresh");
    }

    /// Once the report exists, an unknown filename is a genuine 404 (the store
    /// is non-empty, so we no longer assume the run is in progress).
    #[tokio::test]
    async fn missing_page_after_report_ready() {
        let st = state(&[("annotate_index.html", "<html>index</html>")]);
        let resp = annotate_handler(State(st.clone()), Path("nope.html".to_string())).await;
        assert_eq!(resp.status(), StatusCode::NOT_FOUND);

        // A known page is served verbatim with 200.
        let resp = annotate_handler(State(st), Path("annotate_index.html".to_string())).await;
        assert_eq!(resp.status(), StatusCode::OK);
        assert!(body_of(resp).await.contains("index"));
    }

    /// `--roi-flamegraph` not passed: terminal "not available" page, regardless
    /// of whether a workdir is set.
    #[tokio::test]
    async fn roi_flamegraph_disabled_serves_terminal_page() {
        let mut st = state(&[]);
        st.roi_flamegraph_enabled = false;
        let resp = roi_flamegraph_handler(State(st)).await;
        assert_eq!(resp.status(), StatusCode::OK);
        assert!(body_of(resp).await.contains("not available"));
    }

    /// Enabled but the SVG hasn't been written to disk yet: pending page.
    #[tokio::test]
    async fn roi_flamegraph_pending_before_page_written() {
        let dir = std::env::temp_dir().join(format!(
            "turbo_roi_flamegraph_test_pending_{:?}",
            std::thread::current().id()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let mut st = state(&[]);
        st.roi_flamegraph_enabled = true;
        st.output_path = Some(dir.clone());
        let resp = roi_flamegraph_handler(State(st)).await;
        assert_eq!(resp.status(), StatusCode::OK);
        assert!(body_of(resp).await.contains("available at end of run"));
        std::fs::remove_dir_all(&dir).ok();
    }

    /// Enabled and the page exists on disk: served verbatim as `text/html`.
    #[tokio::test]
    async fn roi_flamegraph_serves_page_once_written() {
        let dir = std::env::temp_dir().join(format!(
            "turbo_roi_flamegraph_test_ready_{:?}",
            std::thread::current().id()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("roi_flamegraph.html"), "<html>test</html>").unwrap();
        let mut st = state(&[]);
        st.roi_flamegraph_enabled = true;
        st.output_path = Some(dir.clone());
        let resp = roi_flamegraph_handler(State(st)).await;
        assert_eq!(resp.status(), StatusCode::OK);
        assert_eq!(
            resp.headers().get(header::CONTENT_TYPE).unwrap(),
            "text/html; charset=utf-8"
        );
        let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        assert_eq!(&bytes[..], b"<html>test</html>");
        std::fs::remove_dir_all(&dir).ok();
    }

    /// The dashboard's favicon comes from the same assets directory as its
    /// stylesheets, through the static fallback rather than a route of its
    /// own. Browsers ignore an icon served as the wrong type, so this pins
    /// both that `/icon.png` resolves and that it arrives as an image.
    #[tokio::test]
    async fn static_fallback_serves_favicon() {
        let mut svc = ServeDir::new(crate::assets::dir());
        let req = axum::http::Request::builder()
            .uri("/icon.png")
            .body(axum::body::Body::empty())
            .unwrap();
        let resp = svc.try_call(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        assert_eq!(resp.headers()[header::CONTENT_TYPE], "image/png");
    }
}
