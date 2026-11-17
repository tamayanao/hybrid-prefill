# Benchmark / 計測データ

Mac Studio (M3 Ultra) / RTX 5090 / 5 GbE / SSH forwarding / DeepSeek V4.1 Flash oQ4e / 2,048-token hybrid chunks. Both 8,405-token conditions use one warm-up and three measured requests. Cache storage is disabled. The long-context samples are single development runs. TTFT is the streaming client time to first nonempty output.

両方の8,405トークン条件はウォームアップ1回と計測3回。キャッシュ保存は無効。長文は開発中の各1回の予備測定です。TTFTはストリーミングAPIで最初の空ではない出力を受け取るまでの時間です。

```sh
python3 benchmark.py --url http://127.0.0.1:18880/v1/completions \
  --model DeepSeek-V4.1-Flash-1017-oQ4e-mtp --revision local-oQ4e-checkpoint \
  --condition hybrid --prompt prompt-8405.txt --output local-results.jsonl \
  --runs 3 --warmups 1 --max-tokens 8 --skip-cache-store
```

Supply the local API key through `OMLX_API_KEY`. / ローカルのAPIキーを環境変数 `OMLX_API_KEY` に設定してください。
