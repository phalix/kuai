FROM python:3.8.6
RUN apt-get update \
    && apt-get install default-jre -y\
    && apt-get install default-jdk -y
#RUN apk --no-cache add\
#    openjdk11-jre
ENV PYTHONUNBUFFERED=1
WORKDIR /code
COPY requirements.txt /code/
RUN pip install -r requirements.txt
COPY . /code/
RUN python manage.py makemigrations
RUN python manage.py migrate
EXPOSE 8000