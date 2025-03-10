from flask import Flask, request
from RoutePlanner.default_router import PoliceRouter
from markupsafe import escape
from flask_restful import Resource, Api
from flask_restful.reqparse import RequestParser
from flasgger import Swagger, swag_from

__version__ = "0.0.2"

app = Flask(__name__)
api = Api(app)
app.config['SWAGGER'] = {
    'title': 'Patrol Routes API',
    'uiversion': 3,
}
swagger = Swagger(app)


class RouteSuggestions(Resource):
    route_computer = PoliceRouter()
    parser: RequestParser
    def __init__(self):
        self.parser = RequestParser()
        self.parser.add_argument('cai',
                                 required=True,
                                 type=int,
                                 location='args')
        self.parser.add_argument('n',
                                 type=int,
                                 location='args')

    
    @swag_from("doc/RouteSuggestions_get.yml")
    def get(self):
        args = self.parser.parse_args()
        # Todo: use real values here
        return self.route_computer.compute_routes(
            args['cai'], args['n'] if args['n'] is not None else 1
        )

api.add_resource(RouteSuggestions, '/api/routes')
if __name__ == '__main__':
    app.run(debug=True)
