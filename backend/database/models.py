#!/usr/bin/env python3
from peewee import *

psql_db = PostgresqlDatabase('route_patrol', user='postgres')

class BaseModel(Model):
    class Meta:
        database = psql_db

class Manager(BaseModel):
    cedula = CharField()
    password_hash = CharField()
    cai_id = IntegerField()

if __name__ == "__main__":
    psql_db.connect()
    psql_db.create_tables([Manager])
