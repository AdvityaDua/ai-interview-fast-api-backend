def _extract_json_from_response(text: str) -> str:
    if "```json" in text:
        s = text.find("```json") + 7
        e = text.find("```", s)
        text = text[s:e].strip()
    elif "```" in text:
        s = text.find("```") + 3
        e = text.find("```", s)
        text = text[s:e].strip()
    
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1:
        text = text[start:end+1]
    import re
    text = re.sub(r',\s*([\]}])', r'\1', text)
    print(f"New len: {len(text)}, text: {text}")
    return text.strip()

test_json = """{
  "cv_quality": {
    "overall_score": 91.0,
    "subscores": [
      {
        "dimension": "ats_compatibility",
        "score": 9.5,
        "max_score": 10.0,
        "evidence": [
          "Formatting is clean and doesn't use complex structures like ```tables or columns"
        ]
      }
    ]
  }
}"""
_extract_json_from_response(test_json)
