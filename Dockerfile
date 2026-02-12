FROM python:3.10-slim-bullseye

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

# uv configuration
ENV UV_LINK_MODE=copy
ENV UV_CACHE_DIR=/root/.cache/uv

RUN apt-get -qq update \
    && DEBIAN_FRONTEND=noninteractive apt-get -qq install -y --no-install-recommends \
    curl \
    python-dev \
    build-essential \
    libopenblas-dev \
    gfortran \
    libboost-thread-dev \
    glpk-utils \
    libglpk-dev \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean -y \
    && rm -rf /var/cache/apt/archives/* /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | env UV_UNMANAGED_INSTALL=/usr/local/bin sh

WORKDIR /app

COPY pyproject.toml uv.lock VERSION.txt /app/
COPY hots /app/hots

# Install dependencies from the lockfile (reproducible build)
# - --frozen: fail if uv.lock doesn't match pyproject.toml
# - --no-editable: install as a normal wheel/site-packages install
# - extras: include optional runtime bits if needed
ARG UV_EXTRAS="kafka"
RUN uv sync --frozen --no-editable --extra "$UV_EXTRAS"

# Run the CLI by default
ENTRYPOINT ["uv", "run", "--", "hots"]
