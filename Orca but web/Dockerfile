FROM python:3.11-slim
WORKDIR /app
RUN pip install flask
COPY app_minimal.py .
CMD ["sh", "-c", "ls -la && python app_minimal.py"]
