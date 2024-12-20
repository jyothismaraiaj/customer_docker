FROM python:3.13
WORKDIR /code
COPY ./requirements.txt /code/requirements.txt
RUN pip install -r /code/requirements.txt
COPY ./Train.csv /code/
COPY ./generate_model.py /code/generate_model.py
COPY ./app /code/app
RUN python generate_model.py
RUN mv logr_v1.pkl app/logr_v1.pkl
RUN rm /code/Train.csv
CMD ["fastapi", "run", "app/main.py", "--port", "80"]