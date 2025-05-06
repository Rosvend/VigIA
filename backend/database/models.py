#!/usr/bin/env python3
from playhouse.postgres_ext import *
from playhouse.shortcuts import model_to_dict
from flask_login import UserMixin

def init_db(app):
    return PostgresqlExtDatabase(app.config['DB_NAME']
                        , host=app.config['DB_HOST']
                        , user=app.config['DB_USER']
                        , password=app.config['DB_PASSWORD'])

class BaseModel(Model):
    EXCLUDED_FIELDS = []

    def toDict(self):
        return model_to_dict(self, exclude=self.EXCLUDED_FIELDS)

    @classmethod
    def setDatabase(cls, db):
        cls._meta.database = db

class Manager(BaseModel, UserMixin):
    cedula = CharField()
    password_hash = CharField()
    cai_id = IntegerField()

    EXCLUDED_FIELDS = [password_hash]

    def get_id():
        return cedula

class Route(BaseModel):
    geometry = JSONField()
    date = DateField()
    cai_id = IntegerField()
    assigned_to = IntegerField()
    assigned_by = ForeignKeyField(Manager, backref='issued_routes')

    EXCLUDED_FIELDS = [date, assigned_by]

    class Meta:
        primary_key = CompositeKey('date', 'cai_id', 'assigned_to')

if __name__ == "__main__":
    psql_db.connect()
    psql_db.create_tables([Manager, Route])
