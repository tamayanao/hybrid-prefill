"""Own the local oMLX process, SSH tunnel, and prepared Windows helper."""

from contextlib import ExitStack
import fcntl
import ipaddress
import json
import os
from pathlib import Path
import re
import secrets
import signal
import socket
import subprocess
import sys
import time
from urllib.error import URLError
from urllib.request import Request, urlopen
from uuid import UUID

from ram_plan import GIB, MAX_VALIDATED_LAYERS, RESIDENT_LAYERS, choose_layers, expert_sizes, read_meminfo


PORT = 18880
CONTAINER = "hybrid-prefill-helper"
APP_IMAGE = "hybrid-prefill-helper:0.0.1"
AUTH_SCOPE = "io.hybridprefill.package"


class Stopped(Exception):
    pass


def stop_requested(*_):
    raise Stopped()


def label(text):
    print(text, flush=True)


def checked_pattern(value, pattern, name):
    if not isinstance(value, str) or not re.fullmatch(pattern, value) or value == "CHANGE-ME":
        raise ValueError(f"{name} を設定してください。")
    return value


def config_at(file, *, pairing=False):
    config = json.loads(file.read_text(encoding="utf-8"))
    for field, pattern in {
        "pc_host": r"[a-zA-Z0-9][a-zA-Z0-9.-]{0,252}",
        "pc_user": r"[a-zA-Z0-9_][a-zA-Z0-9_.-]{0,63}",
        "wsl_distribution": r"[a-zA-Z0-9][a-zA-Z0-9_.-]{0,80}",
        "model_id": r"[a-zA-Z0-9][a-zA-Z0-9_.-]{0,191}",
    }.items():
        checked_pattern(config.get(field), pattern, field)
    if config.get("ssh_host_key_alias"):
        checked_pattern(config["ssh_host_key_alias"], r"[a-zA-Z0-9][a-zA-Z0-9.-]{0,252}", "ssh_host_key_alias")
    for name in ("omlx_python", "model_directory"):
        value = config.get(name)
        if not value or value == "CHANGE-ME":
            raise ValueError(f"{name} を設定してください。")
        config[name] = Path(value).expanduser().absolute()
    if not config["omlx_python"].is_file() or not os.access(config["omlx_python"], os.X_OK):
        raise ValueError("oMLXのPython実行ファイルが見つかりません。")
    if not (config["model_directory"] / config["model_id"] / "config.json").is_file():
        raise ValueError("モデル保存先に指定したチェックポイントが見つかりません。")
    if not pairing:
        config["helper_node_id"] = str(UUID(config["helper_node_id"]))
    maximum = config.get("max_layers", MAX_VALIDATED_LAYERS)
    if type(maximum) is not int or not RESIDENT_LAYERS <= maximum <= MAX_VALIDATED_LAYERS:
        raise ValueError("max_layers は3〜9の整数を指定してください。")
    config["max_layers"] = maximum
    if config.get("chunk_tokens") not in (2048, 4096, 8192):
        raise ValueError("chunk_tokens は2048、4096、8192から選んでください。")
    if type(config.get("copy_missing_model_files", False)) is not bool:
        raise ValueError("copy_missing_model_files は true または false を指定してください。")
    if config.get("ssh_key"):
        config["ssh_key"] = Path(config["ssh_key"]).expanduser().resolve(strict=True)
        if not config["ssh_key"].is_file():
            raise ValueError("SSH秘密鍵のファイルを指定してください。")
    return config


def command(args, timeout=20, capture=True):
    return subprocess.run(args, check=True, timeout=timeout, stdin=subprocess.DEVNULL, encoding="utf-8", errors="replace",
                          stdout=subprocess.PIPE if capture else None,
                          stderr=subprocess.PIPE if capture else None)


def request(path, key=None, *, method="GET", data=None, port=PORT, timeout=5):
    headers = {"Content-Type": "application/json"}
    if key:
        headers["Authorization"] = f"Bearer {key}"
    req = Request(f"http://127.0.0.1:{port}{path}",
                  json.dumps(data).encode() if data is not None else None,
                  headers, method=method)
    with urlopen(req, timeout=timeout) as response:
        return json.load(response)


def stop_process(process):
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=65)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


def wait_for(check, label_text, *, timeout=120, processes=()):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if any(p.poll() is not None for p in processes):
            raise RuntimeError(f"{label_text} の途中でプロセスが終了しました。")
        try:
            if check():
                return
        except (URLError, TimeoutError):
            pass
        time.sleep(1)
    raise TimeoutError(f"{label_text} がタイムアウトしました。")


def run(config, config_file, *, pairing=False):
    app_dir = config_file.parent
    ssh = ["/usr/bin/ssh", "-n", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=yes",
           "-o", "ConnectTimeout=8", "-o", "ServerAliveInterval=5", "-o", "ServerAliveCountMax=3"]
    if config.get("ssh_host_key_alias"):
        ssh += ["-o", f'HostKeyAlias={config["ssh_host_key_alias"]}']
    if config.get("ssh_key"):
        ssh += ["-i", str(config["ssh_key"]), "-o", "IdentitiesOnly=yes"]
    target = f'{config["pc_user"]}@{config["pc_host"]}'
    wsl = f'wsl -d {config["wsl_distribution"]} -u root --'
    model = config["model_id"]
    keyfile = app_dir / "state" / "settings.json"
    state = keyfile.parent
    state.mkdir(mode=0o700, exist_ok=True)
    isolated_home = app_dir / "runtime-home"
    isolated_home.mkdir(mode=0o700, exist_ok=True)
    if not keyfile.exists():
        keyfile.write_text(json.dumps({
            "auth": {"api_key": secrets.token_urlsafe(32)},
            "server": {"distributed_inference_enabled": True},
        }))
        keyfile.chmod(0o600)
    key = json.loads(keyfile.read_text())["auth"]["api_key"]

    def remote(text, timeout=20):
        return command([*ssh, target, text], timeout).stdout.strip()

    for port in (PORT, 8850, 8851, 8852):
        with socket.socket() as probe:
            probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                probe.bind(("127.0.0.1", port))
            except OSError as error:
                raise RuntimeError(f"ローカルポート {port} を確保できません。") from error
    label("SSH接続と専用helperを確認中")
    metadata, = json.loads(remote(f"{wsl} docker inspect {CONTAINER}"))
    if metadata["Config"]["Image"] != APP_IMAGE:
        raise RuntimeError("配布パックのhelperをWindows側にセットアップしてください。")
    if (metadata["Config"].get("Labels") or {}).get(AUTH_SCOPE) != "1":
        raise RuntimeError("専用helperコンテナを識別できません。")
    running = "true" if metadata["State"]["Running"] else "false"
    if running != "false":
        raise RuntimeError("専用helperが既に稼働しています。先に停止してください。")
    free = remote("nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits")
    if len(free.splitlines()) != 1 or int(free) < 29 * 1024:
        raise RuntimeError("GPUの空き容量を確保してください。このパックはRTX 5090 1枚を想定しています。")
    physical = int(remote(
        'powershell -NoProfile -Command "(Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory"'
    ))
    memory = read_meminfo(remote(f"{wsl} cat /proc/meminfo"))
    sizes = expert_sizes(config["model_directory"] / model, config["max_layers"])
    plan = choose_layers(physical_bytes=physical, wsl_total_bytes=memory["MemTotal"],
                         wsl_available_bytes=memory["MemAvailable"],
                         layer_bytes=sizes, maximum=config["max_layers"])
    layers = plan.layers
    label(f"搭載RAM {physical/GIB:.1f} GiB / WSL利用可能 {memory['MemAvailable']/GIB:.1f} GiB")
    label(f"担当 {layers} 層 / ピン留め重み {plan.pinned_bytes/GIB:.1f} GiB")
    addresses = remote(f"{wsl} hostname -I").split()
    address = str(ipaddress.IPv4Address(addresses[0]))

    with ExitStack() as cleanup:
        forward = [*ssh, "-o", "ExitOnForwardFailure=yes",
                   "-R", f"127.0.0.1:{PORT}:127.0.0.1:{PORT}"]
        for port in (8850, 8851, 8852):
            forward += ["-L", f"127.0.0.1:{port}:{address}:{port}"]
        tunnel = subprocess.Popen([*forward, "-N", target])
        cleanup.callback(stop_process, tunnel)
        label("開発版oMLXを起動中")
        server = subprocess.Popen(
            [str(config["omlx_python"]), str(Path(__file__).with_name("serve.py")), "serve",
             "--host", "127.0.0.1", "--port", str(PORT), "--base-path", str(state),
             "--model-dir", str(config["model_directory"])],
            stdin=subprocess.DEVNULL,
            env={**os.environ, "HOME": str(isolated_home),
                 "HYBRID_PREFILL_LAYERS": str(layers),
                 "HYBRID_PREFILL_CHUNK_TOKENS": str(config["chunk_tokens"])},
        )
        cleanup.callback(stop_process, server)
        wait_for(lambda: request("/health").get("status") == "healthy",
                 "oMLXの起動", processes=(tunnel, server))
        command([*ssh, target, f"{wsl} docker start {CONTAINER}"], 30)
        cleanup.callback(remote, f"{wsl} docker stop -t 10 {CONTAINER}", 25)
        lease = subprocess.Popen([*ssh, target, f"{wsl} docker wait {CONTAINER}"],
                                 stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        cleanup.callback(stop_process, lease)
        processes = (lease, tunnel, server)

        def identify():
            nonlocal node_id
            node_id = str(UUID(request("/api/cluster/node_id", port=8850)["node_id"]))
            if not pairing and node_id != config["helper_node_id"]:
                raise RuntimeError("helperのIDが設定と一致しません。")
            return True

        node_id = None
        wait_for(identify, "helperとSSHトンネルの起動", processes=processes)
        if pairing:
            label("Windows側の docker logs hybrid-prefill-helper で6桁のコードを確認してください。")
            code = input("Pairing code: ").strip()
            if not re.fullmatch(r"[0-9]{6}", code):
                raise ValueError("6桁のペアリングコードを入力してください。")
            request("/api/cluster/pair/approve", key, method="POST",
                    data={"node_id": node_id, "code": code}, timeout=25)

        def check_helper():
            candidate = request(f"/admin/api/prefill-helper/check?node_id={node_id}", key)
            rows = candidate.get("rows", [])
            return bool(rows) and all(row["status"] != "fail" for row in rows)

        wait_for(check_helper, "helperの認証", timeout=30, processes=processes)
        if pairing:
            saved = json.loads(config_file.read_text(encoding="utf-8"))
            saved["helper_node_id"] = node_id
            temp = config_file.with_suffix(".tmp")
            temp.write_text(json.dumps(saved, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            temp.replace(config_file)
            label("ペアリング完了。Macアプリから起動できます。")
            return
        request(f"/admin/api/models/{model}/settings", key, method="PUT", data={
            "prefill_helper_node_id": config["helper_node_id"],
            "prefill_helper_layers": layers, "max_context_window": 70000,
        })
        label("モデルとGPU担当層をロード中")
        request(f"/v1/models/{model}/load", key, method="POST", timeout=600)
        copy_requested = False

        def configured():
            nonlocal copy_requested
            value = request(f"/admin/api/prefill-helper/models/{model}", key)
            status = value.get("state")
            if status == "needs_files" and config.get("copy_missing_model_files") and not copy_requested:
                label("helperに不足しているモデルファイルを転送中")
                request(f"/admin/api/prefill-helper/models/{model}/copy", key, method="POST", timeout=20)
                copy_requested = True
                return False
            if status == "needs_files" and copy_requested:
                return False
            if status in ("error", "offline", "needs_files"):
                raise RuntimeError(f"helper準備失敗: {status} {value.get('reason', '')}")
            return status == "ready" and value.get("layers") == layers
        wait_for(configured, "GPU層の準備", timeout=1200, processes=processes)
        label(f"API: http://127.0.0.1:{PORT}/v1")
        label("READY")
        while all(p.poll() is None for p in processes):
            time.sleep(2)
        raise RuntimeError("oMLX、SSH、helperのいずれかが終了しました。")


def main():
    os.umask(0o077)
    signal.signal(signal.SIGTERM, stop_requested)
    signal.signal(signal.SIGINT, stop_requested)
    label("Hybrid Prefill v.0.0.1 検証テスト版")
    if len(sys.argv) not in (2, 3) or (len(sys.argv) == 3 and sys.argv[2] != "--pair"):
        raise ValueError("Usage: launch.py CONFIG [--pair]")
    pairing = len(sys.argv) == 3
    file = Path(sys.argv[1]).resolve(strict=True)
    with (file.parent / ".launcher.lock").open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("この設定は既に使用中です。") from None
        run(config_at(file, pairing=pairing), file, pairing=pairing)


if __name__ == "__main__":
    try:
        main()
    except Stopped:
        label("停止しました。")
    except (EOFError, KeyError, TypeError, ValueError, OSError, RuntimeError, subprocess.SubprocessError) as error:
        label(str(error))
        raise SystemExit(1)
