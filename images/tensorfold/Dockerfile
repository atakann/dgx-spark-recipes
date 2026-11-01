# TensorFold's release binary on NGC PyTorch, which brings CUDA 13, NCCL and verbs for two-Spark ranks.
FROM nvcr.io/nvidia/pytorch@sha256:2140e699b3beaf7f96a0081fd9c9406bc3832b435cdb60dfa2d261f7d2f34a1c
ARG TF_VERSION=1.1.9
ARG TF_SHA256=e1d36b67f0298ee1cacf4f6c40482bbc3a4f539c771d01006bf7f6e0927c669a
RUN curl -fsSL -o /tmp/tf.tgz \
      https://github.com/ashhart/TensorFold/releases/download/v${TF_VERSION}/tensorfold-${TF_VERSION}-linux-aarch64.tar.gz && \
    echo "${TF_SHA256}  /tmp/tf.tgz" | sha256sum -c - && \
    mkdir /opt/tensorfold && tar xzf /tmp/tf.tgz -C /opt/tensorfold --strip-components=1 && rm /tmp/tf.tgz && \
    cd /opt/tensorfold && sha256sum -c SHA256SUMS
ENTRYPOINT ["/opt/tensorfold/bin/tensorfold-native"]
