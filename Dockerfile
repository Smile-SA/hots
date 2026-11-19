FROM python:3.12-slim-bookworm

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

# uv configuration
ENV UV_LINK_MODE=copy
ENV UV_CACHE_DIR=/root/.cache/uv
# Use the image's Python, never download one.
ENV UV_PYTHON_DOWNLOADS=never

RUN apt-get -qq update \
    && DEBIAN_FRONTEND=noninteractive apt-get -qq install -y --no-install-recommends \
    build-essential \
    libopenblas-dev \
    gfortran \
    libboost-thread-dev \
    glpk-utils \
    libglpk-dev \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean -y \
    && rm -rf /var/cache/apt/archives/* /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Install uv from PyPI (the CI runners cannot reach astral.sh).
RUN python -m pip install --no-cache-dir --root-user-action=ignore --disable-pip-version-check uv

WORKDIR /app

COPY pyproject.toml uv.lock VERSION.txt /app/
COPY hots /app/hots

# Install dependencies from the lockfile (reproducible build)
# - --frozen: install exactly what uv.lock pins, without checking it against pyproject.toml
#   (the CI `audit` tox env fails when uv.lock is out of date)
# - --no-editable: install as a normal wheel/site-packages install
# - extras: include optional runtime bits if needed
ARG UV_EXTRAS="kafka"
RUN uv sync --frozen --no-editable --extra "$UV_EXTRAS"

# Run the CLI by default, straight from the venv: `uv run` would re-sync the
# project (rebuilding hots as editable) on every start.
ENV PATH="/app/.venv/bin:$PATH"
ENTRYPOINT ["hots"]
