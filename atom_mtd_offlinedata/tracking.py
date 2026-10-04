"""Online-only W&B logging, with server-side metric readback."""

import os
import time
import netrc
from urllib.parse import urlparse

import requests
import wandb


def configure_auth(c):
    """Forge's W&B gateway accepts the existing W&B cloud credential.

    Reuse it only for this explicitly requested, verified gateway; keep the key
    in process memory/environment, never in config files or command arguments.
    """
    endpoint = urlparse(settings(c)["base_url"])
    if (
        endpoint.scheme == "https"
        and endpoint.netloc == "forge.coreweave.com"
        and endpoint.path.rstrip("/") == "/api/wandb"
        and not os.environ.get("WANDB_API_KEY")
    ):
        credentials = netrc.netrc().authenticators("api.wandb.ai")
        if credentials:
            os.environ["WANDB_API_KEY"] = credentials[2]


def run_url(c, run):
    if settings(c)["base_url"].rstrip("/") == "https://forge.coreweave.com/api/wandb":
        return f"https://forge.coreweave.com/wandb/{run.entity}/{run.project}/runs/{run.id}"
    return run.url


def settings(c):
    return dict(
        base_url=os.environ.get("WANDB_BASE_URL", c["wandb_base_url"]), init_timeout=30
    )


def probe_endpoint(c):
    url = settings(c)["base_url"].rstrip("/") + "/graphql"
    # Auth-free capability probe. Never repurpose a credential from a different host.
    response = requests.post(url, json={"query": "{ __typename }"}, timeout=15)
    if "json" not in response.headers.get("content-type", ""):
        raise RuntimeError(
            f"W&B GraphQL endpoint {url} returned HTTP {response.status_code} non-JSON; configure the actual Forge API URL"
        )
    if response.status_code not in (200, 401, 403):
        raise RuntimeError(f"W&B GraphQL endpoint returned HTTP {response.status_code}")


def initialize(c, name, directory, full_config, run_id=None):
    """Start an online run; run_id continues an earlier run (training resume)."""
    probe_endpoint(c)
    configure_auth(c)
    if os.environ.get("WANDB_MODE", "online") != "online":
        raise RuntimeError(
            "The requested experiment requires verified online W&B metrics"
        )
    run = wandb.init(
        project=c["project"],
        entity=os.environ.get("WANDB_ENTITY") or c["wandb_entity"],
        name=name,
        dir=str(directory),
        config=full_config,
        mode="online",
        settings=wandb.Settings(**settings(c)),
        **({"id": run_id, "resume": "must"} if run_id else {}),
    )
    if run.settings.mode != "online":
        raise RuntimeError("W&B is not online")
    return run


def verify_history(c, run_path, key, attempts=6):
    configure_auth(c)
    api = wandb.Api(overrides={"base_url": settings(c)["base_url"]}, timeout=20)
    for attempt in range(attempts):
        remote = api.run(run_path)
        if any(
            key in row and row[key] is not None
            for row in remote.scan_history(keys=[key], page_size=10)
        ):
            return True
        if attempt + 1 < attempts:
            time.sleep(3)
    raise RuntimeError(f"W&B server has no recorded {key} metric for {run_path}")


def log_evaluation(c, run_path, results):
    configure_auth(c)
    api = wandb.Api(overrides={"base_url": settings(c)["base_url"]}, timeout=20)
    remote = api.run(run_path)
    metrics = {}
    # Specialists are evaluated on their own task only; log whatever is present.
    for task in ("i1", "i5", "x1"):
        if task not in results:
            continue
        metrics[f"offline/{task}/status"] = results[task]["status"]
        for name, value in results[task].get("metrics", {}).items():
            metrics[f"offline/{task}/{name}"] = value
    if "mean_atomic_temporal_residual_error" in results:
        metrics["offline/mean_atomic_temporal_residual_error"] = results[
            "mean_atomic_temporal_residual_error"
        ]
    remote.summary.update(metrics)
