import json
test_json = """{
  "cv_quality": {
    "overall_score": 91.0,
    "subscores": [
      {
        "dimension": "ats_compatibility",
        "score": 9.5,
        "max_score": 10.0,
        "evidence": [
          "This candidate's code """
try:
    json.loads(test_json)
except Exception as e:
    print(e)
