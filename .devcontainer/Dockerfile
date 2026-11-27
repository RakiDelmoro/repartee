FROM mcr.microsoft.com/devcontainers/base:resolute

USER root

# Bump to upgrade. Changing this ARG invalidates the omp layer and nothing above it.
ARG OMP_VERSION=v19.2.0

# The presigned CDN resets HTTP/2 streams on the 125 MB binary behind the
# devcontainer proxy (curl exit 92). Shim curl -> HTTP/1.1 + retries.
RUN set -eux; \
    mkdir -p /tmp/curl-shim; \
    printf '#!/bin/sh\nexec /usr/bin/curl --http1.1 --retry 5 --retry-all-errors --retry-delay 2 "$@"\n' \
        > /tmp/curl-shim/curl; \
    chmod +x /tmp/curl-shim/curl

# Official installer, pinned. PI_INSTALL_DIR puts omp in a directory already on
# PATH, so no rc edits and no "omp: command not found".
# bash -o pipefail matters: dash would report success even if the script
# download itself failed.
RUN set -eux; \
    PATH="/tmp/curl-shim:$PATH" \
    PI_INSTALL_DIR="/usr/local/bin" \
    OMP_VERSION="${OMP_VERSION}" \
    bash -o pipefail -c 'curl -fsSL https://omp.sh/install | sh -s -- --binary --ref "$OMP_VERSION"'; \
    omp --version; \
    rm -rf /tmp/curl-shim