FROM python:3.8

WORKDIR /app

ADD requirements.txt requirements.txt
RUN python -m pip install --upgrade pip
RUN pip3 install -r requirements.txt

COPY . .

EXPOSE 8000
EXPOSE 8001

CMD ["sh", "/app/run.sh"]