import json
test_json = """{
  "cv_quality": {
    "overall_score": 91.0,
    "subscores": [
      {
        "dimension": "ats_compatibility"
      }"""
try:
    json.loads(test_json)
except Exception as e:
    print(e)
