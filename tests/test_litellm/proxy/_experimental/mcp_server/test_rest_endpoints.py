from typing import Dict, Optional

import pytest
from starlette.requests import Request

from litellm.proxy._experimental.mcp_server import rest_endpoints
from litellm.proxy._experimental.mcp_server.auth import (
    user_api_key_auth_mcp as auth_mcp,
)
from litellm.proxy._types import NewMCPServerRequest, UpdateMCPServerRequest, UserAPIKeyAuth
from litellm.types.mcp import MCPAuth


def _build_request(headers: Optional[Dict[str, str]] = None) -> Request:
    headers = headers or {}
    raw_headers = [
        (key.lower().encode("latin-1"), value.encode("latin-1"))
        for key, value in headers.items()
    ]
    scope = {
        "type": "http",
        "http_version": "1.1",
        "method": "POST",
        "path": "/mcp-rest/test/tools/list",
        "headers": raw_headers,
    }

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    return Request(scope, receive=receive)


@pytest.mark.asyncio
async def test_test_tools_list_forwards_mcp_auth_header(monkeypatch):
    """Ensure credential-based auth forwards the auth_value to the MCP client."""

    captured: dict = {}

    async def fake_execute(request, operation, mcp_auth_header=None, oauth2_headers=None):
        captured["mcp_auth_header"] = mcp_auth_header
        captured["oauth2_headers"] = oauth2_headers
        return {
            "tools": [],
            "error": None,
            "message": "Successfully retrieved tools",
        }

    monkeypatch.setattr(
        rest_endpoints, "_execute_with_mcp_client", fake_execute, raising=False
    )

    oauth_call_counter = {"count": 0}

    def fake_oauth(headers):
        oauth_call_counter["count"] += 1
        return {"Authorization": "Bearer oauth"}

    monkeypatch.setattr(
        auth_mcp.MCPRequestHandler,
        "_get_oauth2_headers_from_headers",
        staticmethod(fake_oauth),
        raising=False,
    )

    request = _build_request()
    payload = NewMCPServerRequest(
        server_name="example",
        url="https://example.com",
        auth_type=MCPAuth.api_key,
        credentials={"auth_value": "secret-key"},
    )

    from litellm.proxy._types import LitellmUserRoles

    result = await rest_endpoints.test_tools_list(
        request,
        payload,
        user_api_key_dict=UserAPIKeyAuth(user_role=LitellmUserRoles.PROXY_ADMIN),
    )

    assert result["message"] == "Successfully retrieved tools"
    assert captured["mcp_auth_header"] == "secret-key"
    assert captured["oauth2_headers"] is None
    assert oauth_call_counter["count"] == 0


@pytest.mark.asyncio
async def test_test_tools_list_extracts_oauth2_headers(monkeypatch):
    """Ensure oauth2 auth type pulls oauth headers and omits MCP auth header."""

    captured: dict = {}

    async def fake_execute(request, operation, mcp_auth_header=None, oauth2_headers=None):
        captured["mcp_auth_header"] = mcp_auth_header
        captured["oauth2_headers"] = oauth2_headers
        return {
            "tools": [],
            "error": None,
            "message": "Successfully retrieved tools",
        }

    monkeypatch.setattr(
        rest_endpoints, "_execute_with_mcp_client", fake_execute, raising=False
    )

    oauth_headers = {"Authorization": "Bearer oauth"}
    oauth_call_counter = {"count": 0}

    def fake_oauth(headers):
        oauth_call_counter["count"] += 1
        return oauth_headers

    monkeypatch.setattr(
        auth_mcp.MCPRequestHandler,
        "_get_oauth2_headers_from_headers",
        staticmethod(fake_oauth),
        raising=False,
    )

    request = _build_request({"authorization": "Bearer incoming"})
    payload = NewMCPServerRequest(
        server_name="example",
        url="https://example.com",
        auth_type=MCPAuth.oauth2,
    )

    from litellm.proxy._types import LitellmUserRoles

    result = await rest_endpoints.test_tools_list(
        request,
        payload,
        user_api_key_dict=UserAPIKeyAuth(user_role=LitellmUserRoles.PROXY_ADMIN),
    )

    assert result["message"] == "Successfully retrieved tools"
    assert captured["mcp_auth_header"] is None
    assert captured["oauth2_headers"] == oauth_headers
    assert oauth_call_counter["count"] == 1


class TestStdioCommandAllowlist:
    """Tests for MCP stdio command allowlist validation."""

    def test_allowed_command_passes_validation(self):
        """npx, uvx, python, etc. should be accepted."""
        req = NewMCPServerRequest(
            server_name="test",
            transport="stdio",
            command="npx",
            args=["-y", "@modelcontextprotocol/server-filesystem"],
        )
        assert req.command == "npx"

    def test_disallowed_command_raises(self):
        """Arbitrary commands like bash should be rejected."""
        with pytest.raises(ValueError, match="not in the allowed commands list"):
            NewMCPServerRequest(
                server_name="test",
                transport="stdio",
                command="bash",
                args=["-c", "echo pwned"],
            )

    def test_sh_command_raises(self):
        """sh should be rejected."""
        with pytest.raises(ValueError, match="not in the allowed commands list"):
            NewMCPServerRequest(
                server_name="test",
                transport="stdio",
                command="sh",
                args=["-c", "id > /tmp/output.txt"],
            )

    def test_absolute_path_bypass_blocked(self):
        """/bin/bash should be blocked (basename is 'bash')."""
        with pytest.raises(ValueError, match="not in the allowed commands list"):
            NewMCPServerRequest(
                server_name="test",
                transport="stdio",
                command="/bin/bash",
                args=["-c", "echo pwned"],
            )

    def test_absolute_path_to_allowed_command_works(self):
        """/usr/bin/python3 should pass (basename is 'python3')."""
        req = NewMCPServerRequest(
            server_name="test",
            transport="stdio",
            command="/usr/bin/python3",
            args=["-m", "some_module"],
        )
        assert req.command == "/usr/bin/python3"

    def test_http_transport_ignores_allowlist(self):
        """HTTP/SSE transport should not trigger command validation."""
        req = NewMCPServerRequest(
            server_name="test",
            transport="sse",
            url="https://example.com/mcp",
        )
        assert req.transport == "sse"

    def test_uvx_command_passes(self):
        req = NewMCPServerRequest(
            server_name="test",
            transport="stdio",
            command="uvx",
            args=["mcp-server-sqlite"],
        )
        assert req.command == "uvx"

    def test_node_command_passes(self):
        req = NewMCPServerRequest(
            server_name="test",
            transport="stdio",
            command="node",
            args=["server.js"],
        )
        assert req.command == "node"

    def test_update_request_disallowed_command_raises(self):
        """UpdateMCPServerRequest should also block non-allowlisted commands."""
        with pytest.raises(ValueError, match="not in the allowed commands list"):
            UpdateMCPServerRequest(
                server_id="some-id",
                transport="stdio",
                command="bash",
                args=["-c", "echo pwned"],
            )


class TestEndpointRoleChecks:
    """Tests for PROXY_ADMIN role checks on MCP test endpoints."""

    @pytest.mark.asyncio
    async def test_test_connection_rejects_non_admin(self):
        """Non-admin users should get 403 from test_connection."""
        from fastapi import HTTPException
        from litellm.proxy._types import LitellmUserRoles

        payload = NewMCPServerRequest(
            server_name="test",
            url="https://example.com/mcp",
            auth_type=MCPAuth.none,
        )
        user_key = UserAPIKeyAuth(
            user_role=LitellmUserRoles.INTERNAL_USER,
            user_id="non_admin",
            api_key="sk-test",
        )
        request = _build_request()

        with pytest.raises(HTTPException) as exc_info:
            await rest_endpoints.test_connection(
                request=request,
                new_mcp_server_request=payload,
                user_api_key_dict=user_key,
            )
        assert exc_info.value.status_code == 403

    @pytest.mark.asyncio
    async def test_test_tools_list_rejects_non_admin(self):
        """Non-admin users should get 403 from test_tools_list."""
        from fastapi import HTTPException
        from litellm.proxy._types import LitellmUserRoles

        payload = NewMCPServerRequest(
            server_name="test",
            url="https://example.com/mcp",
            auth_type=MCPAuth.none,
        )
        user_key = UserAPIKeyAuth(
            user_role=LitellmUserRoles.INTERNAL_USER,
            user_id="non_admin",
            api_key="sk-test",
        )
        request = _build_request()

        with pytest.raises(HTTPException) as exc_info:
            await rest_endpoints.test_tools_list(
                request=request,
                new_mcp_server_request=payload,
                user_api_key_dict=user_key,
            )
        assert exc_info.value.status_code == 403

    @pytest.mark.asyncio
    async def test_test_connection_allows_admin(self, monkeypatch):
        """PROXY_ADMIN should pass the role check."""
        from litellm.proxy._types import LitellmUserRoles

        async def fake_execute(*args, **kwargs):
            return {"status": "ok"}

        monkeypatch.setattr(
            rest_endpoints,
            "_execute_with_mcp_client",
            fake_execute,
        )

        payload = NewMCPServerRequest(
            server_name="test",
            url="https://example.com/mcp",
            auth_type=MCPAuth.none,
        )
        user_key = UserAPIKeyAuth(
            user_role=LitellmUserRoles.PROXY_ADMIN,
            user_id="admin",
            api_key="sk-admin",
        )
        request = _build_request()

        result = await rest_endpoints.test_connection(
            request=request,
            new_mcp_server_request=payload,
            user_api_key_dict=user_key,
        )
        assert result["status"] == "ok"
