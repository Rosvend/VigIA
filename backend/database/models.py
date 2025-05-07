#!/usr/bin/env python3
from playhouse.postgres_ext import *
from playhouse.shortcuts import model_to_dict
from .populate import init_data
from flask_login import UserMixin

def init_db(app, *models):
    app.config['DATABASE'] = PostgresqlExtDatabase(
        app.config['DB_NAME'],
        host=app.config['DB_HOST'],
        user=app.config['DB_USER'],
        password=app.config['DB_PASSWORD'])
    
    for model in models:
        model.setDatabase(app.config['DATABASE'])
        if not model.table_exists():
            model.create_table()
            if (app.config['DB_AUTOPOPULATE']
               and model._meta.table_name in init_data):
                app.logger.info('Populating table ' + model._meta.table_name)
                model.insert_many(init_data[model._meta.table_name]).execute()
                    

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
