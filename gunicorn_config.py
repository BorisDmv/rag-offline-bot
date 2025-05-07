# gunicorn_config.py
bind = "0.0.0.0:9090"
workers = 1
timeout = 300  # Increase timeout if LLM inference takes long