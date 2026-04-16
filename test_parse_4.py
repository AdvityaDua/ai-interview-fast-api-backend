import json

test_json = """{
  "cv_quality": {
    "overall_score": 91.0,
    "subscores": [
      {
        "dimension": "ats_
"""

try:
    json.loads(test_json)
except json.JSONDecodeError as jde:
    print(jde)
    print(f"Expectation matches? {jde.msg}")
