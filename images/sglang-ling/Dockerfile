# SGLang at 20621aa (the nightly the inclusionAI branch ling_v3_support_mxfp4_humming is based on) + that branch's
# one commit 079d404: the online FP8 lm_head, the MTP draft sharing that lm_head, and the load fix 20621aa needs for
# Ling-3.0. The three files come from GitHub at the pinned commit and are checked against their sha256. No weights inside.
FROM lmsysorg/sglang@sha256:40b02a1d78982c85dda79ade7522b30d8152ee9487d5c075ea7f9134b197d25f
ARG C=079d4046003daf757d8a2daa44a3bb8935174b5e
WORKDIR /sgl-workspace/sglang/python/sglang/srt
RUN b=https://raw.githubusercontent.com/inclusionAI/sglang/$C/python/sglang/srt \
 && curl -sfL -o environ.py $b/environ.py \
 && curl -sfL -o models/bailing_moe_v3.py $b/models/bailing_moe_v3.py \
 && curl -sfL -o models/bailing_moe_nextn.py $b/models/bailing_moe_nextn.py \
 && printf '%s\n' "ccce6143c8fdb18cdefb1ab970d3f7d6c38284f094970701860e5d47482a5a05  environ.py" \
    "ac04ef2eaa1e2a736a1daede8189803e568d486119395cf00978479700d54c2d  models/bailing_moe_v3.py" \
    "31c0daeedd3dcb20ac6ef45ba7a065fe603a645f53af6b8a7446b48e8c138af5  models/bailing_moe_nextn.py" \
    | sha256sum -c -
WORKDIR /
# FP8 lm_head on (the commit's switch); the notebook's DeepGEMM and context-length settings
ENV SGLANG_ENABLE_FP8_LM_HEAD=1 SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1 \
    SGLANG_JIT_DEEPGEMM_PRECOMPILE=0 SGLANG_ENABLE_JIT_DEEPGEMM=0
ENTRYPOINT ["python3", "-m", "sglang.launch_server"]
