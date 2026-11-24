"""LTX-2.5 demo web app (the lbw web_app.py pattern): serves the page under URL_PREFIX and relays /api/* to the
backend (LTX_URL: the ltx (co-located) or ltx-dit service). The session is SSE pass-through; each event carries one
chunk as a base64 mp4 (video + audio) that the page plays back to back."""
import os
from pathlib import Path

import httpx
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse, StreamingResponse

# LTX_URLS: "label=url,label=url" -- the page offers each backend (co-located ltx, disaggregated ltx-dit)
LTX_URLS = dict(kv.split("=", 1) for kv in os.environ.get(
    "LTX_URLS", "colo=http://ltx.default.svc.cluster.local:8000").split(","))
PREFIX = os.environ.get("URL_PREFIX", "").rstrip("/")
INDEX = (Path(__file__).parent / "index.html").read_text()
clients = {k: httpx.AsyncClient(base_url=u, timeout=httpx.Timeout(10, read=None)) for k, u in LTX_URLS.items()}


def _client(name):
    if name not in clients:
        raise HTTPException(400, f"unknown backend {name!r}")
    return clients[name]
app = FastAPI()


@app.get("/", response_class=HTMLResponse)
def index():
    opts = "".join(f'<option value="{k}">{k}</option>' for k in LTX_URLS)
    return INDEX.replace("__BACKENDS__", opts)


@app.get("/api/status")
async def status(backend: str = "colo"):
    try:
        r = await _client(backend).get("/readiness")
        return {"ready": r.status_code == 200, "detail": r.json() if r.status_code == 200 else r.text[:200]}
    except Exception as e:  # noqa: BLE001
        return {"ready": False, "detail": str(e)[:200]}


@app.post("/api/session")
async def session(request: Request):
    body = await request.json()
    if not (body.get("prompt") or "").strip():
        raise HTTPException(400, "prompt required")
    client = _client(body.get("backend", "colo"))
    req = client.build_request("POST", "/session", json={"prompt": body["prompt"], "seed": int(body.get("seed", 42)),
                                                          "live": bool(body.get("live", True))})
    r = await client.send(req, stream=True)
    if r.status_code != 200:
        txt = (await r.aread()).decode()[:300]
        await r.aclose()
        raise HTTPException(r.status_code, txt)

    async def relay():
        try:
            async for chunk in r.aiter_raw():
                yield chunk
        finally:
            await r.aclose()
    return StreamingResponse(relay(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.post("/api/stop")
async def stop(backend: str = "colo"):
    r = await _client(backend).post("/stop")
    return r.json()


if PREFIX:
    root = FastAPI()
    root.add_api_route(PREFIX, lambda: RedirectResponse(PREFIX + "/"))
    root.mount(PREFIX, app)
else:
    root = app

if __name__ == "__main__":
    uvicorn.run(root, host=os.environ.get("HOST", "127.0.0.1"), port=int(os.environ.get("PORT", "7860")))
