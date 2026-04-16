import json
test_json = '{"test": "This string has "quotes" in it"}'
try:
    json.loads(test_json)
except Exception as e:
    print(e)
