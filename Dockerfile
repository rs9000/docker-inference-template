FROM python:3.8

WORKDIR /app

COPY . .

RUN python -m pip install --upgrade pip
RUN pip3 install -r requirements.txt

EXPOSE 8000
EXPOSE 8001

CMD ["sh", "run.sh"]