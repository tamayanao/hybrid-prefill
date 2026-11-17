# Hybrid Prefill v0.0.2

検証テスト版 / Experimental release

- 11層の分担と物理RAMに応じた3〜11層の自動選択 / Up to 11 offloaded layers, with automatic sizing from available RAM.
- BF16一時KVとpooled KVの再利用 / BF16 scratch with pooled KV reuse on the Mac.
- 専用WSL2 Helperの68 GiB上限とWindows・WSLのメモリ監視 / Dedicated 68 GiB helper limit and Windows/WSL memory monitoring.
- 65,541トークンでPrefill **33.811秒・1,938.4 tok/s**、TTFT **34.098秒**。全12試行で生成一致 / 65,541-token paired test, with matching generation in all 12 runs.

[日本語の説明とセットアップ](https://github.com/tamayanao/hybrid-prefill/blob/main/README.md) · [English setup and measurements](https://github.com/tamayanao/hybrid-prefill/blob/main/README.en.md)

配布ZIPはMacアプリ、Windows/WSL Helper、oMLXパッチ、日英の説明図を収録します。モデル重みは利用環境で用意してください。 / The ZIP includes the Mac app, Windows/WSL helper, oMLX patch, and both diagrams. Supply the model weights separately.

配布版の実機試験はペアリング、11層の自動選択、64K初回生成の参照一致まで確認しました。次の試行はSSH切断により安全停止したため、配布版での性能再現は未確認です。 / The packaged smoke test verified pairing, automatic selection of 11 layers, and one matching 64K generation. An SSH disconnect stopped the next run. The performance figures are from the paired development benchmark.
