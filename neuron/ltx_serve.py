"""LTX-2.5 demo server on Neuron (the LingBot serve.py pattern): torchrun SPMD, rank 0 serves HTTP on a thread, jobs
are broadcast to every rank, all device work runs on each rank's main thread.

Roles (LTX_ROLE):
  colo  the whole pipeline on this pod (DiT, upsampler, video VAE, audio VAE + vocoder).
  dit   DiT + upsampler here; each chunk's latents go to the VAE pod (VAE_URL, HTTP) and the DiT starts the next
        chunk while the VAE decodes (DiffusionDuet disaggregation).
  vae   the video VAE + audio VAE + vocoder: POST /decode (a pickled chunk) -> mp4 bytes.

HTTP (colo / dit):
  POST /session {prompt, seed=42, live=true}  -> SSE: data: {"chunk", "mp4" (b64, video + audio), "gen_s", ...}
       live: chunk after chunk (seed + k) until /stop or disconnect; otherwise one chunk. Ends with {"done"}.
  POST /stop,  GET /health,  GET /readiness (200 after the warmup chunks).
Launched by run_tp.py when LTX_SERVE=1 (all of run_tp's Neuron patches apply). LTX_ARGS: the model / size args.
"""
import base64
import io
import json
import os
import queue
import shlex
import sys
import threading
import time

import torch
import torch.distributed as dist

READY = threading.Event()


class Job:
    def __init__(self, prompt, seed, live):
        self.prompt, self.seed, self.live = prompt, seed, live
        self.events = queue.Queue()
        self.stop = threading.Event()


class Jobs:
    """rank 0: one session at a time; a new session supersedes the running one."""

    def __init__(self):
        self.pending = queue.Queue()
        self.current = None

    def submit(self, job):
        if self.current is not None:
            self.current.stop.set()
            self.current.events.put({"superseded": True})
        self.current = job
        self.pending.put(job)


def _argv(base, prompt, seed, out):
    return ["distilled"] + base + ["--prompt", prompt, "--seed", str(seed), "--output-path", out]


def _http_server(role, jobs, decodes):
    import uvicorn
    from fastapi import FastAPI, HTTPException, Request
    from fastapi.responses import Response, StreamingResponse
    app = FastAPI()

    @app.get("/health")
    def health():
        return {"ok": True}

    @app.get("/readiness")
    def readiness():
        if not READY.is_set():
            raise HTTPException(503, "warming up")
        return {"ready": True, "role": role}

    if role == "vae":
        @app.post("/decode")
        async def decode(request: Request):
            body = await request.body()
            box = {"done": threading.Event()}
            decodes.put((body, box))
            # wait off the event loop: a blocking wait here held uvicorn's only loop, so the pod read the next request
            # only after this one's mp4 was sent (one chunk at a time end to end, ~1 chunk/s with the main thread
            # idle half the time under 3 streams)
            import asyncio
            await asyncio.get_running_loop().run_in_executor(None, box["done"].wait)
            if "error" in box:
                raise HTTPException(500, box["error"])
            return Response(box["mp4"], media_type="video/mp4")
    else:
        @app.post("/session")
        async def session(request: Request):
            if not READY.is_set():
                raise HTTPException(503, "warming up")
            b = await request.json()
            prompt = (b.get("prompt") or "").strip()
            if not prompt:
                raise HTTPException(400, "prompt required")
            job = Job(prompt, int(b.get("seed", 42)), bool(b.get("live", True)))
            jobs.submit(job)

            async def stream():
                import asyncio
                try:
                    while True:
                        try:
                            ev = job.events.get_nowait()
                        except queue.Empty:
                            if await request.is_disconnected():
                                job.stop.set()
                                return
                            await asyncio.sleep(0.05)
                            continue
                        if ev is None:
                            return
                        yield f"data: {json.dumps(ev)}\n\n"
                finally:
                    job.stop.set()
            return StreamingResponse(stream(), media_type="text/event-stream",
                                     headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

        @app.post("/stop")
        def stop():
            if jobs.current is not None:
                jobs.current.stop.set()
            return {"stopped": True}

    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "8000")), log_level="warning")


def _poll(q, timeout=float(os.environ.get("LTX_IDLE_BEAT_S", "30"))):
    """rank 0: the next queued item, or None after `timeout` s (an idle heartbeat broadcast to the other ranks, which
    otherwise sit in the broadcast past the collective timeout and abort)."""
    try:
        return q.get(timeout=timeout)
    except queue.Empty:
        return None


def _bcast(obj, rank):
    box = [obj if rank == 0 else None]
    dist.broadcast_object_list(box, src=0)
    return box[0]


def main(rt):
    rank, world = rt["rank"], rt["world"]
    D, move = rt["D"], rt["_move"]
    role = os.environ.get("LTX_ROLE", "colo")
    import extra2
    extra2.install(rt)
    base = shlex.split(os.environ["LTX_ARGS"])
    fps = float(base[base.index("--frame-rate") + 1]) if "--frame-rate" in base else 24.0
    jobs, decodes = Jobs(), queue.Queue()
    if rank == 0:
        threading.Thread(target=_http_server, args=(role, jobs, decodes), daemon=True).start()
    out = "/tmp/ltx_chunk.mp4" if rank == 0 else f"/tmp/ltx_chunk_r{rank}.mp4"

    if role == "vae":
        _vae_loop(rt, base, fps, decodes, out)
        return

    if role == "dit":
        _dit_hooks(rt, fps)
    enc_async = role == "colo" and os.environ.get("LTX_ASYNC_ENCODE", "0") == "1"
    if enc_async:
        _async_encode_hooks(rt)
    stage_log = os.environ.get("LTX_STAGE_LOG", "0") == "1"

    def chunk(prompt, seed):
        sys.argv = _argv(base, prompt, seed, out)
        rt["T"].clear(); rt["N"].clear()
        t0 = time.perf_counter()
        D.main()
        if role == "dit":
            # rank 0 alone copies the latents to host after the DiT; keep the other ranks from racing into the next
            # chunk's collectives meanwhile (that mismatch failed the stage-2 all-reduce: "Invalid NEFF")
            _bcast(None, rank)
        s = time.perf_counter() - t0
        if stage_log and rank == 0:
            print("[STAGE] " + " ".join(f"{k.split()[0]}={v * 1e3:.0f}" for k, v in rt["T"].items()) +
                  f" total={s * 1e3:.0f}", flush=True)
        return s

    warm = os.environ.get("LTX_WARMUP_PROMPT", "A red fox walks through fresh snow in a quiet pine forest at dawn.")
    for i in range(int(os.environ.get("LTX_WARMUP_RUNS", "3"))):   # run 0 compiles, run 1 recompiles once
        s = chunk(warm, 42)
        if role == "dit":
            rt["_DIT"]["drain"]()          # the VAE pod compiles on the first decodes: rank 0 waits, all ranks beat
        if rank == 0:
            print(f"[SERVE] warmup {i}: {s:.1f} s", flush=True)
    if rank == 0:
        READY.set()
        print("[SERVE] ready", flush=True)

    while True:
        job = _poll(jobs.pending) if rank == 0 else None
        spec = _bcast(None if job is None else (job.prompt, job.seed, job.live), rank)
        if spec is None:               # idle heartbeat: keeps the other ranks' broadcast under the collective timeout
            continue
        prompt, seed, live = spec
        k = 0
        while True:
            s = chunk(prompt, seed + k)
            if rank == 0:
                if role == "dit":
                    rt["_DIT"]["emit_to"](job, k, s)
                elif enc_async:
                    rt["_ENC"]["emit_to"](job, k, s)
                else:
                    job.events.put({"chunk": k, "mp4": base64.b64encode(open(out, "rb").read()).decode(),
                                    "gen_s": round(s, 3)})
            k += 1
            if role == "dit":
                rt["_DIT"]["throttle"]()
            go = _bcast(rank == 0 and live and not job.stop.is_set(), rank)
            if not go:
                break
        if role == "dit":
            rt["_DIT"]["drain"]()
        if enc_async and rank == 0:
            rt["_ENC"]["drain"]()
        if rank == 0:
            job.events.put({"done": True, "chunks": k})
            job.events.put(None)


def _async_encode_hooks(rt):
    """colo with LTX_ASYNC_ENCODE=1 (the fair co-located baseline: the same overlap the disagg VAE pod gives the DiT
    pod's host): the decoders run on the main thread as before (device work, all ranks), the mp4 mux / H.264 encode
    of the decoded frames and audio runs on one rank-0 worker thread while the next chunk denoises. Exact: the same
    frames and audio go to the same encoder."""
    D, rank = rt["D"], rt["rank"]
    enc = rt["_enc"]
    q, state, lock = queue.Queue(), {"inflight": 0}, threading.Condition()
    pending = {}

    def worker():
        while True:
            segs, kw, job, k, gen_s = q.get()
            path = f"/tmp/ltx_chunk_a{k % 4}.mp4"
            try:
                enc(iter(segs), **{**kw, "output_path": path})
                ev = {"chunk": k, "mp4": base64.b64encode(open(path, "rb").read()).decode(), "gen_s": round(gen_s, 3)}
            except Exception as e:  # noqa: BLE001
                ev = {"error": f"encode failed: {e}"}
            if job is not None:
                job.events.put(ev)
            with lock:
                state["inflight"] -= 1
                lock.notify_all()

    if rank == 0:
        threading.Thread(target=worker, daemon=True).start()

    def encode_media_segments(segments, **kw):
        segs = list(segments)                     # runs the video / audio decoders here (device, every rank)
        if rank == 0:
            pending["item"] = (segs, kw)

    def emit_to(job, k, gen_s):
        with lock:
            state["inflight"] += 1
        segs, kw = pending.pop("item")
        q.put((segs, kw, job, k, gen_s))

    def drain():
        with lock:
            while state["inflight"]:
                lock.wait(5)

    D.encode_media_segments = encode_media_segments
    rt["_ENC"] = {"emit_to": emit_to, "drain": drain}


def _dit_hooks(rt, fps):  # noqa: C901
    """dit role: the chunk generator's decode step ships the chunk (latents) to the VAE pod instead (rank 0, on a
    worker thread, so the next chunk's denoising overlaps the decode); the mp4 the VAE returns goes to the session."""
    import httpx
    D, move, rank = rt["D"], rt["_move"], rt["rank"]
    url = os.environ["VAE_URL"].rstrip("/") + "/decode"
    sendq, state = queue.Queue(), {"job": None, "k": 0, "inflight": 0}
    lock = threading.Condition()

    def sender():
        client = httpx.Client(timeout=httpx.Timeout(10, read=7200))   # the first decodes compile on the VAE pod
        while True:
            body, job, k, gen_s = sendq.get()
            t0 = time.perf_counter()
            try:
                r = client.post(url, content=body)
                r.raise_for_status()
                ev = {"chunk": k, "mp4": base64.b64encode(r.content).decode(), "gen_s": round(gen_s, 3),
                      "vae_s": round(time.perf_counter() - t0, 3)}
            except Exception as e:  # noqa: BLE001
                ev = {"error": f"vae decode failed: {e}"}
            if job is not None:
                job.events.put(ev)
            with lock:
                state["inflight"] -= 1
                lock.notify_all()

    if rank == 0:
        threading.Thread(target=sender, daemon=True).start()
    pending = {}

    def decode_chunks(chunks, video_decoder, audio_decoder, fps, **kw):
        for c in chunks:
            if rank == 0:
                buf = io.BytesIO()
                # only what the decode reads: the latents and the layout (the conditioning factories are local
                # functions, not picklable, and the prompt embeddings are not used by the decoders)
                from dataclasses import replace
                from ltx_pipelines.chunks import state as _CS
                c = replace(c, video_context=None, audio_context=None,
                            make_video_conditionings=_CS._empty_conditionings,
                            make_audio_conditionings=_CS._empty_conditionings)
                torch.save({"chunk": move(c, torch.device("cpu")), "fps": fps}, buf)
                pending["body"] = buf.getvalue()
        return iter(())

    def encode_media_segments(segments, **kw):
        for _ in segments:
            pass

    def emit_to(job, k, gen_s):
        with lock:
            state["inflight"] += 1
        sendq.put((pending.pop("body"), job, k, gen_s))

    def throttle():
        """Backpressure: at most LTX_MAX_INFLIGHT chunks (default 2) sent and not yet decoded. Without it the DiT ran
        hundreds of chunks ahead of a slower VAE (unbounded latency and memory). All ranks call it, same beats as
        drain()."""
        cap = int(os.environ.get("LTX_MAX_INFLIGHT", "2"))
        if rank == 0:
            while True:
                with lock:
                    if state["inflight"] >= cap:
                        lock.wait(30)
                    busy = state["inflight"] >= cap
                _bcast(busy, rank)
                if not busy:
                    return
        else:
            while _bcast(None, rank):
                pass

    def drain():
        """Wait until every sent chunk is decoded. All ranks call it: rank 0 broadcasts a heartbeat every 30 s while it
        waits (the VAE pod's first decodes compile for a long time), then a stop; the other ranks follow the beats."""
        if rank == 0:
            if "body" in pending:          # warmup chunks: decode them too (compiles / warms the VAE pod)
                emit_to(None, -1, 0.0)
            while True:
                with lock:
                    if state["inflight"]:
                        lock.wait(30)
                    busy = state["inflight"] > 0
                _bcast(busy, rank)
                if not busy:
                    return
        else:
            while _bcast(None, rank):
                pass

    D.decode_chunks = decode_chunks
    D.encode_media_segments = encode_media_segments
    rt["_DIT"] = {"emit_to": emit_to, "drain": drain, "throttle": throttle}


def _vae_loop(rt, base, fps, decodes, out):
    """vae role: build the pipeline's video + audio decoders once; every rank decodes each received chunk (the split
    VAE and split vocoder use all ranks); rank 0 encodes the mp4 and answers the request."""
    D, rank, dit_device = rt["D"], rt["rank"], rt["dit_device"]
    sys.argv = ["distilled"] + base + ["--prompt", "x", "--output-path", out]
    params = D.resolve_cli_params(distilled=True)
    parser = D.add_chunk_layout_args(D.add_keyframe_decode_arg(D.add_generated_keyframes_arg(
        D.default_2_stage_distilled_arg_parser(params=params, supports_auto_duration=True))))
    args = parser.parse_args()
    pipe = D.DistilledPipeline(model_paths=args.model_paths, spatial_upsampler_path=args.spatial_upsampler_path,
                               loras=(), quantization=args.quantization, compilation_config=args.compile,
                               offload_mode=args.offload_mode, prompt_enhancer_gemma_root=args.prompt_enhancer_gemma_root,
                               diffvae_optimization=args.diffvae_optimization)
    enc = rt["_enc"]

    def decode(body):
        obj = torch.load(io.BytesIO(body), weights_only=False)
        segs = D.decode_chunks(iter([obj["chunk"]]), pipe.video_decoder, pipe.audio_decoder, fps=obj["fps"],
                               tiling_config=None, generator=None, dtype=torch.bfloat16, keyframes=False)
        enc(segs, fps=obj["fps"], output_path=out, video_chunks_number=None)
        return open(out, "rb").read() if rank == 0 else b""

    # LTX_VAE_ASYNC_ENCODE=1: the device decode on the main thread (every rank), the H.264 / AAC mux on rank-0 worker
    # threads (LTX_VAE_ENC_THREADS, default 2), so the next chunk's decode overlaps this chunk's encode (the encode,
    # ~0.55 s of host time, was most of the VAE pod's 1.2 s round trip). Exact: the same frames go to the same encoder.
    async_enc = os.environ.get("LTX_VAE_ASYNC_ENCODE", "0") == "1"
    encq = queue.Queue()

    def enc_worker(i):
        while True:
            segs, efps, box = encq.get()
            path = f"/tmp/ltx_vae_enc{i}.mp4"
            try:
                enc(iter(segs), fps=efps, output_path=path, video_chunks_number=None)
                box["mp4"] = open(path, "rb").read()
            except Exception as e:  # noqa: BLE001
                box["error"] = f"encode failed: {e}"
            box["done"].set()

    if async_enc and rank == 0:
        for i in range(int(os.environ.get("LTX_VAE_ENC_THREADS", "2"))):
            threading.Thread(target=enc_worker, args=(i,), daemon=True).start()

    stage_log = os.environ.get("LTX_STAGE_LOG", "0") == "1"

    def decode_async(body, box):
        t0 = time.perf_counter()
        rt["T"].clear()
        obj = torch.load(io.BytesIO(body), weights_only=False)
        t1 = time.perf_counter()
        segs = list(D.decode_chunks(iter([obj["chunk"]]), pipe.video_decoder, pipe.audio_decoder, fps=obj["fps"],
                                    tiling_config=None, generator=None, dtype=torch.bfloat16, keyframes=False))
        t2 = time.perf_counter()
        if rank == 0:
            encq.put((segs, obj["fps"], box))
            if stage_log:
                print(f"[VSTAGE] load={1e3 * (t1 - t0):.0f} decode={1e3 * (t2 - t1):.0f} " +
                      " ".join(f"{k}={v * 1e3:.0f}" for k, v in rt["T"].items()) + f" q={decodes.qsize()}", flush=True)

    warm = os.environ.get("LTX_VAE_WARM")          # optional: a pickled chunk to decode at startup
    if warm and os.path.exists(warm):
        for i in range(2):
            t0 = time.perf_counter()
            decode(open(warm, "rb").read())
            if rank == 0:
                print(f"[SERVE] vae warmup {i}: {time.perf_counter() - t0:.1f} s", flush=True)
    if rank == 0:
        READY.set()
        print("[SERVE] vae ready", flush=True)
    while True:
        item = _poll(decodes) if rank == 0 else None
        body = _bcast(None if item is None else item[0], rank)
        if body is None:               # idle heartbeat
            continue
        if async_enc:
            try:
                decode_async(body, item[1] if rank == 0 else None)
            except Exception as e:  # noqa: BLE001
                if rank == 0:
                    item[1]["error"] = str(e)
                    item[1]["done"].set()
                print(f"[SERVE] decode failed on rank {rank}: {e}", flush=True)
            continue
        try:
            mp4 = decode(body)
            if rank == 0:
                item[1]["mp4"] = mp4
        except Exception as e:  # noqa: BLE001
            if rank == 0:
                item[1]["error"] = str(e)
            print(f"[SERVE] decode failed on rank {rank}: {e}", flush=True)
        if rank == 0:
            item[1]["done"].set()
