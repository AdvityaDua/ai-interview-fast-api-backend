import json
test_json = """{
  "cv_quality": {
    "overall_score": 91.0,
    "subscores": [
      {
        "dimension": "ats_compatibility"
      }
    ]
  },
  "jd_match": {
    "score": 8.0
  }"""
try:
    json.loads(test_json)
except Exception as e:
    print(e)
