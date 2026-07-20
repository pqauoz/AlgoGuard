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
