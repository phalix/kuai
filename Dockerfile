# For more information, please refer to https://aka.ms/vscode-docker-python
FROM python:3.8.6

EXPOSE 8000

# Keeps Python from generating .pyc files in the container
#ENV PYTHONDONTWRITEBYTECODE=1

# Turns off buffering for easier container logging
#ENV PYTHONUNBUFFERED=1

RUN apt-get update \
    && apt-get install default-jre -y\
    && apt-get install default-jdk -y
#    && apt-get install g++ -y\
#    && apt-get install python3-pandas -y\
#    && apt-get install python3-numpy -y
#RUN apk add openjdk11 # for alpine
#RUN apk add g++ # for alpine
RUN /usr/local/bin/python -m pip install --upgrade pip
WORKDIR /code
COPY requirements.txt /code/
RUN pip install -r requirements.txt
COPY . /code/

RUN python manage.py makemigrations
RUN python manage.py migrate
#RUN wget https://github.com/dessa-oss/atlas/releases/download/0.1.1/atlas_installer.py
#RUN python atlas_installer.py -y -i -s -a -C
#RUN python atlas_installer.py -y -N -s


# During debugging, this entry point will be overridden. For more information, please refer to https://aka.ms/vscode-docker-python-debug
# File wsgi.py was not found in subfolder: 'kuai'. Please enter the Python path to wsgi file.
#CMD ["gunicorn", "--bind", "0.0.0.0:8000", "pythonPath.to.wsgi"]
