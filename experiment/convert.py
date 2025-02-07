import pandas as pd

data = pd.read_csv("../hurto_a_persona.csv")
data.to_json("hurto_a_persona.json", orient='records')
