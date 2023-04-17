FROM python:3.8

WORKDIR /app

COPY . .

RUN ["apt-get", "update"]
RUN ["apt-get", "install", "-y", "vim"]
RUN pip3 install -r requirements.txt

EXPOSE 8000

ENTRYPOINT ["uvicorn",  "app:app", "--reload"]
