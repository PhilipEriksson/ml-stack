FROM ghcr.io/open-webui/open-webui:main

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip \
    && rm -rf /var/lib/apt/lists/*

COPY ./requirements.txt /tmp/api-requirements.txt
RUN pip install --no-cache-dir -r /tmp/api-requirements.txt

COPY ./app.py /opt/api/app.py
COPY ./start.sh /start-services.sh
RUN chmod +x /start-services.sh

EXPOSE 8000 8080

ENTRYPOINT ["/start-services.sh"]
