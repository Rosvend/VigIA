#!/usr/bin/env python3
from playhouse.postgres_ext import *
from flask_login import UserMixin

psql_db = PostgresqlExtDatabase('route_patrol', user='postgres')

class BaseModel(Model):
    class Meta:
        database = psql_db

class Manager(BaseModel, UserMixin):
    cedula = CharField()
    password_hash = CharField()
    cai_id = IntegerField()

    def get_id():
        return cedula

class Route(BaseModel):
    geometry = JSONField()
    date = DateField()
    cai_id = IntegerField()
    assigned_to = IntegerField()
    assigned_by = ForeignKeyField(Manager, backref='issued_routes')

    class Meta:
        primary_key = CompositeKey('date', 'cai_id', 'assigned_to')

if __name__ == "__main__":
    psql_db.connect()
    psql_db.create_tables([Manager, Route])
