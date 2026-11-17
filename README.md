# Hybrid Prefill v0.0.2 検証テスト版

[English](README.en.md) · [配布ZIP](https://github.com/tamayanao/hybrid-prefill/raw/main/dist/Hybrid-Prefill-v0.0.2.zip)

![v0.0.2の8K・32K・64K実測と元図の参考値との比較](docs/hybrid-prefill-comparison-v0.0.2-measured.png)

DeepSeek V4.1 Flashの入力処理を、Mac StudioとRTX 5090搭載Windows PCで分担します。Macがモデル全体とKVキャッシュを保持して応答を生成し、RTXが先頭の層を処理します。最初の3層をGPUに常駐させ、追加層のExpert重みをWSL2のピン留めRAMから転送します。

起動時にWindowsの搭載RAMとWSL2の総量・空き容量を読み取り、専用Helperの上限68 GiBも考慮します。12 GiBを確保し、チェックポイントから算出した重みサイズに応じて3〜11層を自動選択します。検証済みの高速構成は11層、32 Expert Bank、2,048トークンチャンク、全ピン留め、先読みOFFです。

## 実機計測

図のTTFTは同じ65,541トークンのプロンプトを使った別系列の実測です。Mac単体は8,192トークンチャンクで1回、9層と11層は2,048トークンチャンクで各5回の中央値です。[3構成のTTFT実測データ](benchmark/v0.0.2/64k-ttft-comparison.json)

| 65,541トークン・同一11層構成 | Prefill中央値 | TTFT中央値 | Prefill速度 |
| --- | ---: | ---: | ---: |
| BF16一時KV | 34.640秒 | 34.927秒 | 1,892.1 tok/s |
| BF16一時KV＋pooled KV再利用 | **33.811秒** | **34.098秒** | **1,938.4 tok/s** |

Mac Studio（M3 Ultra・512 GB）、RTX 5090（32 GB VRAM）・DDR5 128 GB搭載PC、5GbE、WSL2上限96 GBで測定しました。各モードはウォームアップ1回と計測5回。全12試行で生成が一致し、全396チャンクが11層を維持しました。Windows計測中のコミット余裕は最低14.90 GiB、物理空きRAMは39.75 GiB、WSL2空きRAMは30.20 GiBでした。[計測データ](benchmark/v0.0.2/64k-paired.json)

2026-11-18の再試験では、公開v0.0.2のランチャーとHelperを使用し、8K・32K・64Kを各ウォームアップ1回と本計測5回で完走しました。全18試行で参照生成が一致し、全330チャンクで11層を維持しました。[追加実測とメモリ記録](docs/v0.0.2-measured-20261118.md)

11層を採用しました。12層は生成の数値差が残り、Expert GEMM候補は実入力の単体試験段階です。上記の結果は記載の機材・モデルでの検証値です。

## セットアップ

1. MacにApple Silicon向けのPython 3.11、uv、Xcodeのコマンドラインツールと、モデル `DeepSeek-V4.1-Flash-1017-oQ4e-mtp` を用意します。下記の固定コミットに同梱パッチを適用して開発版oMLXを作ります。

   ```sh
   git clone https://github.com/jundot/omlx.git omlx
   git -C omlx checkout --detach bf28cd6c43541ececeec01ba797377c913da5f8f
   git -C omlx apply ../omlx.patch
   uv venv --python 3.11 omlx/.venv
   uv pip install --python omlx/.venv/bin/python -e ./omlx
   ```

   配布ZIPのMacアプリを使用できます。ソースから再ビルドする場合は `bash build-mac.sh` を実行します。

2. WindowsにOpenSSH Server、WSL2（Ubuntu 24.04）、CUDA対応のDocker Engineを用意します。MacからWindowsへのSSHホスト鍵を確認し、鍵認証とTCP転送を使える状態にします。MacではTCP 18880と8850〜8852を空け、RTX 5090に29 GiB以上の空きVRAMを確保します。
3. WSL2のNAT環境では、`ip route show default` でWindows側ゲートウェイのIPv4を確認します。Windowsの管理者PowerShellで、そのゲートウェイのTCP 18880だけをWindows側 `127.0.0.1:18880` へ転送します。Windowsファイアウォールではアクセス元をWSLのIPv4に限定します。WSLで同梱の `windows-helper/install.sh` を実行し、ゲートウェイIPv4を引数に指定します。初回イメージビルドにネット接続が必要です。

   ```powershell
   # 実環境のWSLゲートウェイとWSL IPv4に置き換える
   $gateway = 'WINDOWS_WSL_GATEWAY_IPV4'
   $wslIp = 'WSL_IPV4'
   netsh interface portproxy add v4tov4 listenaddress=$gateway listenport=18880 connectaddress=127.0.0.1 connectport=18880
   New-NetFirewallRule -DisplayName 'Hybrid Prefill WSL pairing' -Direction Inbound -Action Allow -Protocol TCP -LocalAddress $gateway -LocalPort 18880 -RemoteAddress $wslIp
   ```

   WSL内で `bash windows-helper/install.sh WINDOWS_WSL_GATEWAY_IPV4` を実行します。WSLのアドレスが変わった場合は転送先と接続元の指定を見直してください。
4. Macアプリを開き、「設定ファイルを開く」を選びます。Windowsの `pc_host` と `pc_user`、必要に応じて `ssh_key`、`ssh_host_key_alias` を設定します。`wsl_distribution` は配置先の名前です。`omlx_python` は手順1の `omlx/.venv/bin/python`、`model_directory` はチェックポイントの親ディレクトリ、`model_id` はチェックポイント名です。
5. 新規セットアップ時はMacのTerminalで以下を実行します。WSLの `docker logs hybrid-prefill-helper` に表示される6桁のコードを入力すると、`helper_node_id` が保存されます。

   ```sh
   /usr/bin/python3 launcher/launch.py "$HOME/Library/Application Support/HybridPrefill/launcher.json" --pair
   ```

6. 初回はモデルの必要なシャードを転送するため、Windows側の空きディスク容量を確認し、`copy_missing_model_files` を `true` にします。Macアプリで「起動」を押し、選ばれた層数と稼働中表示を確認します。終了は「停止」です。APIはMacの `http://127.0.0.1:18880/v1`、APIキーは設定ディレクトリ内の `state/settings.json` に保存されます。

`max_layers` で上限を3〜11層の範囲で変更できます。モデル重み、SSH鍵、APIキー、ペアリング情報は利用環境ごとに用意します。Macアプリはローカル署名の検証ビルドです。

`reuse_pooled_kv` を `false` にするとBF16一時KVのみの経路に戻ります。専用Helperのインストール時に68 GiBのコンテナ上限を設定します。WSL2やDockerのグローバル設定は各環境で管理してください。ランチャーはWindowsのコミット余裕・物理空きRAMとWSL2空きRAMを監視します。

v0.0.1から更新する場合は旧Helperを停止し、既存コンテナをリネームしてから同梱インストーラーを実行してください。既存の `hybrid-prefill-helper` データボリュームとMac側の設定を再利用できます。旧コンテナと新コンテナは同時に起動しないでください。
