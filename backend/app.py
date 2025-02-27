from flask import Flask, request
from RoutePlanner.default_router import PoliceRouter
from markupsafe import escape
app = Flask(__name__)
__version__ = "0.0.1"

route_computer = PoliceRouter()

@app.route("/")
def index():
    return f"Running crime backend version {__version__}</p>"

@app.route("/routes/<int:area>")
def getRoutes(area):
    route_qty = request.args.get("n", "")
    return route_computer.compute_routes()
