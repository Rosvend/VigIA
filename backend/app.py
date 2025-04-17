from flask import Flask, request
from flask_cors import CORS
from RoutePlanner.default_router import PoliceRouter, ORS_PROFILES, DEFAULT_ORS_PROFILE
from RoutePlanner.prediction.simple_wrapper import SimpleModelWrapper
from markupsafe import escape
from flask_restful import Resource, Api
from flask_restful.reqparse import RequestParser
from flasgger import Swagger, swag_from
import json

__version__ = "0.0.2"

app = Flask(__name__)
CORS(app)
api = Api(app)
app.config['SWAGGER'] = {
    'title': 'Patrol Routes API',
    'uiversion': 3,
}
swagger = Swagger(app)


class RouteSuggestions(Resource):
    route_computer = PoliceRouter(SimpleModelWrapper(
        model_data_path = "crime_model_simple.pkl",
        grid_features_path = "crime_dataset_quick_sample.csv"
    ))
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
        self.parser.add_argument('profile',
                                 type=str,
                                 choices=ORS_PROFILES,
                                 location='args')
        self.parser.add_argument('exclude_station',
                                 type=bool,
                                 location='args')
        self.parser.add_argument('threshold',
                                 type=float,
                                 location='args')
        self.parser.add_argument('hotspots',
                                 type=bool,
                                 location='args')
        self.parser.add_argument('requested_spots',
                                 type=str,
                                 location='args')
    
    @swag_from("doc/RouteSuggestions_get.yml")
    def get(self):
        args = self.parser.parse_args()
        requested_spots = json.loads(args['requested_spots']) \
            if args['requested_spots'] is not None \
            else []
        return self.route_computer.compute_routes(
            args['cai'],
            args['n'] if args['n'] is not None else 1,
            args['profile'] if args['profile'] is not None
                else DEFAULT_ORS_PROFILE,
            args['exclude_station'] is None,
            args['threshold'] if args['threshold'] is not None
                else 0.0,
            args['hotspots'] is not None,
            requested_spots
        )

api.add_resource(RouteSuggestions, '/api/routes')
if __name__ == '__main__':
    app.run(debug=True)
