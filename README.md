# Snake Brain

## Description

This is a back end part to a snake game variant Feed the Snake (https://github.com/taavi247/feed-the-snake). Snake Brain is built on Django and it stores Feed the Snake states into PostgreSQL database and sends back corresponding data to control the snake. Game states in the database are used through reinforcement learning algorithm to teach neural network.

This is a part of learning project on full stack (React, Django) and machine learning (Pytorch). Django was chosen for back end because language compatibility with Pytorch.

## Used packages:

Python 3.8.10<br/>

## Installation

Step 1:

Create directory and clone the project. Python and pip are required. Also virtual environment recommended. You can create and activate one using commands

```
python -m venv .your-venv
source .your-vent/bin/activate
```

Step 2: Inside the folder __snake-brain__ run following to install dependencies

```
pip install -r requirements.txt
```

Step 3: Install PostgreSQL

```
sudo apt install postgresql postgresql-contrib
```

Step 4: Configure PostgreSQL database for Django

```
sudo -u postgresql psql

CREATE DATABASE db_snakebrain;
CREATE USER snakefeeder WITH PASSWORD djangoproject;

ALTER ROLE snakefeeder SET client_encoding TO 'utf8';
ALTER ROLE snakefeeder SET default_transaction_isolation TO 'read committed';
ALTER ROLE snakefeeder SET timezone TO 'UTC';

GRANT ALL PRIVILEGES ON DATABASE db_snakebrain TO snakefeeder;

\q
```

Step 5: Create and apply migrations to the database and create your Django superuser
```
python manage.py makemigrations
python manage.py migrate

python manage.py createsuperuser
```

Step 6: Run the server
```
python manage.py runserver
```