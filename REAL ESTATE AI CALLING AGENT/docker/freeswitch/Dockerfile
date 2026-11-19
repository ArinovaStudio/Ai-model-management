FROM signalwire/freeswitch:1.10.10

LABEL maintainer="RealEstateAI SRE Team"
LABEL description="Production FreeSWITCH VoIP Telephony PBX container with SIP trunking and WebRTC audio bridging"

# Install system runtime diagnostic tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    procps \
    net-tools \
    iproute2 \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Ensure FreeSWITCH runtime directories exist and install mod_audio_fork if available
RUN mkdir -p /etc/freeswitch \
    /var/log/freeswitch \
    /var/run/freeswitch \
    /var/lib/freeswitch/db \
    /var/lib/freeswitch/recordings \
    /var/lib/freeswitch/sounds \
    /usr/lib/freeswitch/mod && \
    (curl -sSL -f "https://github.com/drachtio/drachtio-freeswitch-modules/releases/download/v1.0.0/mod_audio_fork.so" -o /usr/lib/freeswitch/mod/mod_audio_fork.so || true) && \
    chmod 755 /usr/lib/freeswitch/mod/mod_audio_fork.so 2>/dev/null || true

# Copy custom configurations
COPY conf/ /etc/freeswitch/

# Telephony SIP Signaling & RTP Audio Media Ports
# 5060: SIP UDP/TCP (Standard internal/carrier signaling)
# 5080: External SIP gateway profile (Twilio/Telnyx inbound trunk)
# 8021: Event Socket Inbound (ESL control)
# 16384-16484: RTP Media stream port range (UDP audio)
EXPOSE 5060/tcp 5060/udp 5080/tcp 5080/udp 8021/tcp 16384-16484/udp

# Set working directory
WORKDIR /etc/freeswitch

# Health check verifies FreeSWITCH process responsiveness
HEALTHCHECK --interval=15s --timeout=5s --retries=3 \
    CMD ps aux | grep freeswitch | grep -v grep || exit 1

ENTRYPOINT ["freeswitch", "-nc", "-nonat"]
