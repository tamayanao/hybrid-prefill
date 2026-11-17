FROM jundot/omlx-cuda-helper@sha256:921c8cf349f690e14c64ccf41780853533e6c760a58d44b2073a7ca648c06adc
COPY worker.py /opt/helper/lib/python3.12/site-packages/omlx_cuda_helper/worker.py
COPY deepseek_v41/ /opt/helper/lib/python3.12/site-packages/omlx_cuda_helper/families/deepseek_v41/
RUN python -c 'from pathlib import Path; p=Path("/opt/helper/lib/python3.12/site-packages/omlx_cuda_helper/supervisor.py"); content=p.read_text(); old="RESERVE_BYTES = 6 << 30"; assert content.count(old)==1; p.write_text(content.replace(old,"RESERVE_BYTES = 1 << 30"))'
ENV OMLX_CUDA_PREFILL_METRICS=1 OMLX_CUDA_STREAM_EXPERTS_THROUGH=11
COPY start.sh /usr/local/bin/hybrid-prefill
ENTRYPOINT ["/bin/sh", "/usr/local/bin/hybrid-prefill"]
