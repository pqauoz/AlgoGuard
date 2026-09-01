def test_logout_requires_confirmation(authenticated_client):
    response = authenticated_client.get("/")
    html = response.get_data(as_text=True)

    assert response.status_code == 200
    assert 'data-bs-target="#logoutConfirmModal"' in html
    assert 'id="logoutConfirmModal"' in html
    assert 'aria-labelledby="logoutConfirmTitle"' in html
    assert 'data-bs-dismiss="modal">Stay signed in</button>' in html
    assert 'method="post" action="/logout"' in html
    assert 'class="btn btn-logout-confirm" type="submit"' in html
    assert 'name="_csrf_token"' in html


def test_login_page_does_not_render_logout_modal(client):
    response = client.get("/login")
    html = response.get_data(as_text=True)

    assert response.status_code == 200
    assert 'id="logoutConfirmModal"' not in html
    assert 'name="_csrf_token"' in html


def test_authenticated_pages_are_not_cached(authenticated_client):
    response = authenticated_client.get("/")
    html = response.get_data(as_text=True)

    assert response.headers["Cache-Control"] == (
        "no-store, no-cache, must-revalidate, max-age=0, private"
    )
    assert response.headers["Pragma"] == "no-cache"
    assert response.headers["Expires"] == "0"
    assert 'window.addEventListener("pageshow"' in html
    assert 'navigationEntries[0].type === "back_forward"' in html


def test_browser_security_headers_are_set(client):
    response = client.get("/login")

    assert response.headers["X-Content-Type-Options"] == "nosniff"
    assert response.headers["X-Frame-Options"] == "DENY"
    assert response.headers["Referrer-Policy"] == "no-referrer"
    assert response.headers["Permissions-Policy"] == ("camera=(), geolocation=(), microphone=()")
    policy = response.headers["Content-Security-Policy"]
    assert "script-src 'self' 'nonce-" in policy
    assert "script-src 'self' 'unsafe-inline'" not in policy
    assert "object-src 'none'" in policy
    assert "frame-ancestors 'none'" in policy
    assert "form-action 'self'" in policy
    html = response.get_data(as_text=True)
    nonce = policy.split("'nonce-", 1)[1].split("'", 1)[0]
    assert f'<script nonce="{nonce}"' in html


def test_logout_clears_session_and_back_navigation_requires_login(authenticated_client):
    logout_response = authenticated_client.post("/logout")

    assert logout_response.status_code == 302
    assert logout_response.headers["Location"].endswith("/login?logged_out=1")
    assert "session=;" in logout_response.headers.get("Set-Cookie", "")

    protected_response = authenticated_client.get("/")
    assert protected_response.status_code == 302
    assert "/login?next=/" in protected_response.headers["Location"]

    login_response = authenticated_client.get(protected_response.headers["Location"])
    assert login_response.status_code == 200
    assert 'id="logoutConfirmModal"' not in login_response.get_data(as_text=True)

    logged_out_page = authenticated_client.get(logout_response.headers["Location"])
    assert "You have been securely logged out." in logged_out_page.get_data(as_text=True)


def test_runtime_defaults_are_safe_for_local_use():
    """Debug off and loopback-only unless the operator opts in explicitly."""
    import re

    with open("app.py") as handle:
        source = handle.read()
    main = source[source.index('if __name__ == "__main__":') :]
    assert 'os.environ.get("FLASK_DEBUG", "0")' in main, "debug must default to off"
    assert 'os.environ.get("ALGOGUARD_HOST", "127.0.0.1")' in main, (
        "the server must default to loopback only"
    )
    assert not re.search(r'host="0\.0\.0\.0"', main), (
        "0.0.0.0 must be opt-in via ALGOGUARD_HOST, not hardcoded"
    )


def test_session_secret_has_no_known_fallback():
    with open("app.py") as handle:
        source = handle.read()

    assert "algoguard-dev-secret" not in source
    assert "secrets.token_hex" in source


def test_login_rejects_missing_csrf_token(client):
    response = client.post(
        "/login",
        data={"username": "admin", "password": "admin123"},
    )

    assert response.status_code == 400
    assert "Invalid or expired request token" in response.get_data(as_text=True)


def test_json_endpoint_rejects_missing_csrf_token(app_module):
    raw_client = app_module.app.test_client()
    raw_client.get("/login")
    with raw_client.session_transaction() as test_session:
        token = test_session["_csrf_token"]
    assert (
        raw_client.post(
            "/login",
            data={
                "username": "admin",
                "password": "admin123",
                "_csrf_token": token,
            },
        ).status_code
        == 302
    )

    response = raw_client.post("/monitor/stop")

    assert response.status_code == 400
    assert response.get_json() == {
        "status": "error",
        "message": "Invalid or expired request token.",
    }


def test_logout_is_post_only(authenticated_client):
    response = authenticated_client.get("/logout")

    assert response.status_code == 405


def test_monitor_feed_does_not_render_event_values_as_html():
    with open("templates/monitor.html") as handle:
        source = handle.read()

    assert "innerHTML" not in source
    assert "verdictBadge.textContent = verdict" in source


def test_safe_next_url_rejects_browser_normalized_redirects(app_module):
    with app_module.app.test_request_context("/"):
        assert app_module._safe_next_url("/dashboard") == "/dashboard"
        assert app_module._safe_next_url("//example.com/path") == "/"
        assert app_module._safe_next_url("/\\example.com/path") == "/"
        assert app_module._safe_next_url("https://example.com/path") == "/"


def test_prediction_api_hides_unexpected_exception_details(
    authenticated_client,
    app_module,
    monkeypatch,
):
    monkeypatch.setattr(
        app_module,
        "get_simulation_schema",
        lambda: {"deployment": {"run_id": 1, "model_name": "Stacking Ensemble"}},
    )

    def fail_prediction(payload):
        raise RuntimeError("private artifact path C:/secret/model.joblib")

    monkeypatch.setattr(app_module, "run_simulation", fail_prediction)
    response = authenticated_client.post("/predict", json={})

    assert response.status_code == 500
    assert response.get_json() == {
        "status": "error",
        "message": "Prediction failed safely.",
    }
    assert "secret" not in response.get_data(as_text=True)
