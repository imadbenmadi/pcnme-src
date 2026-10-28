# One image for every PCNME testbed role (zone, car, cloud, wan-gw, runner); the role is chosen by the command.
# final-design 15.4. Built and used from milestone M9.
FROM python:3.12-slim

# iproute2: tc netem shaping and static routes (final-design 15.4); procps for sysctl checks.
RUN apt-get update \
 && apt-get install -y --no-install-recommends iproute2 procps \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
COPY config ./config
RUN pip install --no-cache-dir -e ".[testbed]"

COPY docker/net-setup.sh /usr/local/bin/net-setup.sh
RUN chmod +x /usr/local/bin/net-setup.sh

ENV PYTHONUNBUFFERED=1 PYTHONUTF8=1
ENTRYPOINT ["/usr/local/bin/net-setup.sh"]
