import json

js_file = "data2.json"

with open(js_file) as f:
    data = json.load(f)

formated = "formatted_" + js_file
with open(formated, "w") as g:
    json.dump(data, g, indent=2)

