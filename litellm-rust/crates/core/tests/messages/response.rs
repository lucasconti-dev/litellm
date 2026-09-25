use litellm_core::messages::{messages, types::MessagesRequest};
use litellm_http::transport::Error as TransportError;
use rstest::rstest;

use super::*;

#[rstest]
#[tokio::test]
async fn the_provider_message_is_returned(call: MessagesCall) {
    let upstream = upstream([message_response()]).await;

    let message = run_message(MessagesCall {
        api_key: Some("sk".into()),
        api_base: Some(upstream.uri()),
        ..call
    })
    .await;

    assert_eq!(message.id, "msg_1");
    assert_eq!(message.content, [json!({"type": "text", "text": "hi"})]);
    assert_eq!(message.stop_reason.as_deref(), Some("end_turn"));
}

#[rstest]
#[case::bad_request(400)]
#[case::unauthorized(401)]
#[case::rate_limited(429)]
#[case::server_error(500)]
#[case::overloaded(529)]
#[tokio::test]
async fn an_upstream_error_keeps_its_status_and_body(call: MessagesCall, #[case] status: u16) {
    let upstream =
        upstream([ResponseTemplate::new(status).set_body_string("upstream said no")]).await;

    let error = run(MessagesCall {
        api_key: Some("sk".into()),
        api_base: Some(upstream.uri()),
        ..call
    })
    .await
    .err()
    .expect("upstream error propagates");

    assert_eq!(
        error,
        Error::Transport(TransportError::Http {
            status,
            body: "upstream said no".into()
        })
    );
}

#[rstest]
#[tokio::test]
async fn an_invalid_thinking_signature_retries_without_replayed_thinking(call: MessagesCall) {
    let upstream = upstream([
        status_response(
            400,
            json!({
                "type": "error",
                "error": {
                    "type": "invalid_request_error",
                    "message": "messages.3.content.0.thinking.signature.str: Input should be a valid string"
                }
            }),
        ),
        message_response(),
    ])
    .await;
    let history = json!([
        {"role": "user", "content": [{"type": "text", "text": "first question"}]},
        {"role": "assistant", "content": [{"type": "text", "text": "first answer"}]},
        {"role": "user", "content": [{"type": "text", "text": "second question"}]},
        {"role": "assistant", "content": [
            {"type": "thinking", "thinking": "replayed from another provider", "signature": null},
            {"type": "tool_use", "id": "call-1", "name": "lookup", "input": {"key": "value"}}
        ]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "call-1", "content": "found"}]}
    ]);

    let response = run(MessagesCall {
        api_key: Some("sk-ant".into()),
        api_base: Some(upstream.uri()),
        body: object(json!({
            "model": MODEL,
            "max_tokens": 64,
            "thinking": {"type": "enabled", "budget_tokens": 1024},
            "tools": [{"name": "lookup", "input_schema": {"type": "object", "properties": {"key": {"type": "string"}}}}],
            "messages": history,
        })),
        ..call
    })
    .await;
    let requests = received(&upstream).await;

    assert_eq!(
        requests.len(),
        2,
        "expected one recovery retry after the signature error"
    );
    let first = requests[0].json();
    let retry = requests[1].json();
    assert_eq!(first["messages"][3]["content"][0]["type"], "thinking");
    assert_eq!(
        first["thinking"],
        json!({"type": "enabled", "budget_tokens": 1024})
    );
    assert_eq!(
        retry["messages"],
        json!([
            {"role": "user", "content": [{"type": "text", "text": "first question"}]},
            {"role": "assistant", "content": [{"type": "text", "text": "first answer"}]},
            {"role": "user", "content": [{"type": "text", "text": "second question"}]},
            {"role": "assistant", "content": [{"type": "tool_use", "id": "call-1", "name": "lookup", "input": {"key": "value"}}]},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "call-1", "content": "found"}]}
        ])
    );
    assert!(retry.get("thinking").is_none());
    assert!(matches!(response, Ok(MessagesOutput::Message(_))));
}

#[rstest]
#[case::not_json(ResponseTemplate::new(200).set_body_string("not json"))]
#[case::not_a_message(json_response(json!({"unexpected": true})))]
#[tokio::test]
async fn an_unreadable_success_body_is_an_invalid_response(
    call: MessagesCall,
    #[case] response: ResponseTemplate,
) {
    let upstream = upstream([response]).await;

    let error = run(MessagesCall {
        api_key: Some("sk".into()),
        api_base: Some(upstream.uri()),
        ..call
    })
    .await
    .err()
    .expect("an unreadable body fails");

    assert!(error.is_response(), "{error:?}");
}

#[rstest]
#[tokio::test]
async fn a_provider_slower_than_the_timeout_fails_the_call(call: MessagesCall) {
    let upstream = upstream([message_response().set_delay(Duration::from_secs(5))]).await;

    let error = run(MessagesCall {
        api_key: Some("sk".into()),
        api_base: Some(upstream.uri()),
        timeout: Some(Duration::from_millis(100)),
        ..call
    })
    .await
    .err()
    .expect("the call times out");

    assert!(matches!(error, Error::Transport(_)), "{error:?}");
}

fn facade_request(body: Value, api_base: &str) -> MessagesRequest<'_> {
    MessagesRequest {
        model: MODEL,
        body,
        api_key: Some("sk-ant"),
        api_base: Some(api_base),
        custom_llm_provider: Some("anthropic"),
        extra_headers: None,
        provider_specific_header: None,
        timeout: Some(Duration::from_secs(5)),
        shaping: MessagesShaping::default(),
    }
}

#[tokio::test]
async fn the_facade_runs_the_route_in_process() {
    let upstream = upstream([message_response()]).await;
    let base = upstream.uri();

    let message = messages(facade_request(
        json!({"model": MODEL, "max_tokens": 16, "messages": [{"role": "user", "content": "hi"}]}),
        &base,
    ))
    .await
    .expect("messages request succeeds");

    assert_eq!(message.id, "msg_1");
    assert_eq!(
        only_request(&upstream).await.header("x-api-key"),
        Some("sk-ant")
    );
}

#[tokio::test]
async fn the_facade_rejects_a_body_that_is_not_an_object() {
    let error = messages(facade_request(json!([]), UNREACHABLE_BASE))
        .await
        .expect_err("a non-object body is rejected");

    assert_eq!(
        error,
        Error::InvalidRequest("messages body must be an object".into())
    );
}
