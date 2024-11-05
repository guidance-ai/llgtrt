use std::sync::Arc;

use axum::{
    extract::State,
    http::{HeaderMap, StatusCode},
    response::{IntoResponse, Response},
    Json,
};
use serde_json::{json, Value};

use crate::{
    error::AppError,
    routes::{completions, openai::CompletionCreateParams},
    state::AppState,
};

pub async fn ready_check() {
    log::info!("ready_check -> all good");
}

pub async fn live_check(
    headers: HeaderMap,
    State(app_state): State<Arc<AppState>>,
) -> Result<Response, AppError> {
    let req: CompletionCreateParams = serde_json::from_value(json!({
        "model": "model",
        "prompt": "Hi",
        "max_tokens": 2
    }))?;
    let resp = completions::route_completions(headers, State(app_state), Json(req)).await?;
    let status = resp.status();
    let body = axum::body::to_bytes(resp.into_body(), 1024 * 1024).await?;
    if status == StatusCode::OK {
        let body: Value = serde_json::from_slice(&body)?;
        log::debug!(
            "Liveness check response: {}",
            serde_json::to_string_pretty(&body)?
        );
        if body["choices"][0]["text"].as_str().is_some() {
            log::info!("Liveness check complete");
            Ok((StatusCode::OK, "Check complete").into_response())
        } else {
            log::error!(
                "Liveness check failed: response body does not contain expected field; body: {}",
                serde_json::to_string_pretty(&body)?
            );
            Ok((StatusCode::INTERNAL_SERVER_ERROR, "Check failed").into_response())
        }
        
    } else {
        log::error!(
            "Liveness check failed with status code: {}; body {}",
            status,
            String::from_utf8_lossy(&body)
        );
        Ok((status, body).into_response())
    }
}
