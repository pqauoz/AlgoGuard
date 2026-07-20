import io
import re


def extract_progress_token(html: str) -> str:
    match = re.search(
        r'id="progressToken" name="progress_token" value="([^"]+)"',
        html,
    )
    assert match is not None, "The upload page did not render a progress token."
    return match.group(1)


def test_logout_requires_confirmation(authenticated_client):
    response = authenticated_client.get("/")
    html = response.get_data(as_text=True)

    assert response.status_code == 200
    assert 'data-bs-target="#logoutConfirmModal"' in html
    assert 'id="logoutConfirmModal"' in html
    assert 'aria-labelledby="logoutConfirmTitle"' in html
    assert 'data-bs-dismiss="modal">Stay signed in</button>' in html
    assert 'class="btn btn-logout-confirm" href="/logout"' in html


def test_login_page_does_not_render_logout_modal(client):
    response = client.get("/login")
    html = response.get_data(as_text=True)

    assert response.status_code == 200
    assert 'id="logoutConfirmModal"' not in html


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


def test_logout_clears_session_and_back_navigation_requires_login(authenticated_client):
    logout_response = authenticated_client.get("/logout")

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


def test_upload_page_exposes_owned_training_progress(authenticated_client):
    response = authenticated_client.get("/upload")
    html = response.get_data(as_text=True)
    token = extract_progress_token(html)

    assert response.status_code == 200
    assert 'id="trainingProgress"' in html
    assert "new XMLHttpRequest()" in html

    progress_response = authenticated_client.get(f"/training-progress/{token}")
    payload = progress_response.get_json()
    assert progress_response.status_code == 200
    assert payload["percent"] == 0
    assert payload["stage"] == "Ready to upload"
    assert payload["state"] == "ready"


def test_training_progress_requires_authentication(
    authenticated_client,
    app_module,
):
    response = authenticated_client.get("/upload")
    token = extract_progress_token(response.get_data(as_text=True))

    anonymous_client = app_module.app.test_client()
    unauthorized_response = anonymous_client.get(f"/training-progress/{token}")
    assert unauthorized_response.status_code == 302
    assert "/login" in unauthorized_response.headers["Location"]


def test_csv_workflow_reaches_one_hundred_percent(
    authenticated_client,
    binary_dataframe,
):
    upload_page = authenticated_client.get("/upload")
    token = extract_progress_token(upload_page.get_data(as_text=True))
    csv_bytes = binary_dataframe.to_csv(index=False).encode("utf-8")

    response = authenticated_client.post(
        "/upload",
        data={
            "progress_token": token,
            "dataset": (io.BytesIO(csv_bytes), "progress-test.csv"),
        },
        content_type="multipart/form-data",
    )

    assert response.status_code == 302
    progress_response = authenticated_client.get(f"/training-progress/{token}")
    payload = progress_response.get_json()
    assert progress_response.status_code == 200
    assert payload["percent"] == 100
    assert payload["stage"] == "Analysis complete"
    assert payload["state"] == "complete"
