"""Tests for Health and Readiness Probes."""

import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_healthz_liveness(async_client: AsyncClient):
    """Test /healthz endpoint returns 200 OK and healthy status."""
    response = await async_client.get("/healthz")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "timestamp" in data
    assert data["service"] == "DarGlobal Knowledge Assistant"
    assert data["version"] == "1.0.0"


@pytest.mark.asyncio
async def test_readyz_readiness(async_client: AsyncClient):
    """Test /readyz endpoint returns ready status with valid storage checks."""
    response = await async_client.get("/readyz")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ready"
    assert "checks" in data
    assert "storage_data" in data["checks"]
    assert "storage_chroma" in data["checks"]
