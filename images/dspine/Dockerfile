# SGLang 0.5.22 (cu130, arm64) + Tencent's DSpine plugin at 0cc92d8, with gb10.patch for GPUs without FA3 (GB10 is
# sm_121): the draft attention may use triton (flashinfer gave garbage output), and ragged verify graphs are skipped when the backend
# has none (batches of 8 then verify full 16-token blocks). DSpine's H100 sampling kernels stay off on the GB10.
# No weights inside.
FROM lmsysorg/sglang@sha256:e3a4f06001a29542fc4a8638536d9ffc5eb9c9942b08603c09b19a56eae4a1ae
RUN git clone https://github.com/Tencent/DSpine /opt/dspine && cd /opt/dspine \
 && git checkout 0cc92d857a8f7bf252de67995ad4c19251620143
COPY gb10.patch /opt/dspine/
RUN cd /opt/dspine && git apply gb10.patch && pip install --no-deps -e .
ENTRYPOINT ["sglang", "serve"]
