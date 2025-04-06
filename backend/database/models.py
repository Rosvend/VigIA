#!/usr/bin/env python3
from peewee import *
from flask_login import UserMixin

psql_db = PostgresqlDatabase('route_patrol', user='postgres')

class BaseModel(Model):
    class Meta:
        database = psql_db

class Manager(BaseModel, UserMixin):
    cedula = CharField()
    password_hash = CharField()
    cai_id = IntegerField()

    def get_id():
        return cedula

if __name__ == "__main__":
    psql_db.connect()
    psql_db.create_tables([Manager])
