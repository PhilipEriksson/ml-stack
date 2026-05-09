FROM vllm/vllm-openai:v0.20.1

# Use start.sh as the entrypoint — it builds the full vllm serve command.
# The base image's ENV vars (CUDA_VERSION, LD_LIBRARY_PATH, PATH, etc.)
# are inherited automatically, so CUDA initialization still works.

WORKDIR /opt

COPY start.sh start.sh
RUN chmod +x start.sh

EXPOSE 8000

ENTRYPOINT ["./start.sh"]
CMD []
