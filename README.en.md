# Hybrid Prefill v.0.0.1 (Verification Test Build)

[日本語](README.md)

Hybrid Prefill splits DeepSeek V4.1 Flash prefill between a Mac Studio (M3 Ultra) and a Windows PC with an RTX 5090. The Mac keeps the full model and KV cache. The RTX 5090 processes the first layers, keeping three layers resident and streaming the expert weights for additional layers from pinned PC RAM.

At startup the launcher reads installed Windows RAM, total WSL2 RAM, and available WSL2 RAM. It reserves 12 GiB from the smallest of the three and uses weight sizes from the checkpoint headers to select 3–9 layers. The default chunk limit is 2,048 tokens. At 8,192 tokens, VRAM pressure reduced the number of layers the GPU could process in testing.

## Measurements on real hardware

| Input tokens | Configuration | Time to first token |
| --- | --- | ---: |
| 8,405 | Mac only, median of three runs after warm-up | 9.920 s |
| 8,405 | Verification launcher, nine layers, median of three runs after warm-up | 6.522 s |
| 32,773 | Nine layers, one preliminary run | 21.714 s |
| 65,541 | Nine layers, one preliminary run | 42.590 s |

Time to first token was 34.3% shorter on the 8,405-token input. Each long-input result is from one run. The 32,773-token output differed from the eight-layer trial. See [benchmark](benchmark) for conditions and data.

## Verification scope

On the test machines, we verified automatic selection of nine layers, first-time pairing, inference, and normal shutdown. The PC had 125.6 GiB of installed RAM and around 60.5 GiB of available WSL2 RAM at startup. The plan pinned 40.3 GiB of weights. Other hardware configurations and model transfers into an empty volume are unverified.

## Setup

1. Prepare Python 3.11, uv, the Xcode command-line tools, and the `DeepSeek-V4.1-Flash-1017-oQ4e-mtp` checkpoint on the Apple Silicon Mac. Apply the included patch to the pinned oMLX commit.

   ```sh
   git clone https://github.com/jundot/omlx.git omlx
   git -C omlx checkout --detach bf28cd6c43541ececeec01ba797377c913da5f8f
   git -C omlx apply ../omlx.patch
   uv venv --python 3.11 omlx/.venv
   uv pip install --python omlx/.venv/bin/python -e ./omlx
   ```

   Use the Mac app in the distribution ZIP or run `bash build-mac.sh` to build it from source.

2. Set up Windows OpenSSH Server, WSL2 (Ubuntu 24.04), and a CUDA-enabled Docker Engine. Verify the Windows SSH host key from the Mac and enable key authentication and TCP forwarding. Free TCP ports 18880 and 8850–8852 on the Mac and at least 29 GiB of VRAM on the RTX 5090.
3. With WSL2 in NAT mode, use `ip route show default` to find the Windows-side gateway IPv4 address. From an elevated PowerShell session, forward TCP 18880 on that gateway address to `127.0.0.1:18880` on Windows. Scope the Windows Firewall rule to the WSL IPv4 address. Run the included `windows-helper/install.sh` inside WSL with the gateway IPv4 address as its argument. The first image build needs internet access.

   ```powershell
   # Substitute the Windows-side gateway and WSL guest addresses
   $gateway = 'WINDOWS_WSL_GATEWAY_IPV4'
   $wslIp = 'WSL_IPV4'
   netsh interface portproxy add v4tov4 listenaddress=$gateway listenport=18880 connectaddress=127.0.0.1 connectport=18880
   New-NetFirewallRule -DisplayName 'Hybrid Prefill WSL pairing' -Direction Inbound -Action Allow -Protocol TCP -LocalAddress $gateway -LocalPort 18880 -RemoteAddress $wslIp
   ```

   In WSL, run `bash windows-helper/install.sh WINDOWS_WSL_GATEWAY_IPV4`. Review the forwarding and firewall addresses if the WSL network is reassigned.
4. Open the app and choose **設定ファイルを開く** (Open Settings File). Fill in `pc_host` and `pc_user`, and optionally `ssh_key` and `ssh_host_key_alias`. Set `wsl_distribution` to the name of the distribution that hosts the helper. Set `omlx_python` to the venv interpreter from step 1, `model_directory` to the parent directory of the checkpoint, and `model_id` to the checkpoint directory name.
5. Run the command below in Mac Terminal. Enter the six-digit code shown by `docker logs hybrid-prefill-helper` in WSL. The launcher saves `helper_node_id`.

   ```sh
   /usr/bin/python3 launcher/launch.py "$HOME/Library/Application Support/HybridPrefill/launcher.json" --pair
   ```

6. Check free disk space on the PC and enable `copy_missing_model_files` if the helper needs to receive the model shards. Click **起動** (Start) and check the selected layer count and running status. Click **停止** (Stop) to shut down. The API is at `http://127.0.0.1:18880/v1` on the Mac, and the API key is stored in `state/settings.json` inside the settings directory.

Set `max_layers` to cap the assignment within the 3–9 layer range. Model weights, SSH keys, API keys, and pairing state are supplied by each installation. The Mac app is an ad hoc-signed verification build.
