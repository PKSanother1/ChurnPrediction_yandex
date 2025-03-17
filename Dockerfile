FROM python:3.12-slim



RUN pip install -r requirements.txt

WORKDIR /app

COPY [ "model.bin", "predict.py", "./"]

EXPOSE 9696

ENTRYPOINT [ "gunicorn", "--bind", "0.0.0.0:9696", "predict:app" ]

# docker build -t predict_app .
# docker run predict_app
# docker login
# docker push [название образа]