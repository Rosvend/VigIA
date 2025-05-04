import flask_unittest
from database.models import init_db, Manager, Route
from app import create_app
import flask.globals

class TestLogin(flask_unittest.ClientTestCase):
    app = create_app("config_test.py")

    def setUp(self, client):
        self.app.config['DATABASE'].create_tables([Manager, Route], safe=False)
