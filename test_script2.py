import requests, os, jwt
from datetime import datetime, timedelta

secret = "mBocwPzdc1d05KpjxqDV97jQ860F1OmjUBw-madaTPxjOv9vc5BGwsZztIQ5u9iJwo9qt2fbMcxubFi40JkxiQ"

token = jwt.encode(
    {
        "sub": "user_id_123",
        "email": "test@test.com",
        "exp": datetime.now() + timedelta(hours=1)
    },
    secret,
    algorithm="HS256"
)

req_data = {
    "message": "test",
    "resume": {
        "id": "123",
        "filename": "mock.pdf",
        "url": "http://example.com/mock.pdf",
        "analytics": {
            "cv_quality": {"overall_score": 80, "subscores": []},
            "jd_match": {"overall_score": 80, "subscores": []}
        },
        "enhancement": {
            "tailored_resume": {
                "summary": "M",
                "experience": [],
                "skills": [],
                "projects": []
            }
        }
    }
}

resp = requests.post(
    "http://localhost:8001/api/v1/resume/final-enhanced",
    json=req_data,
    headers={"Authorization": f"Bearer {token}"}
)

print(resp.status_code)
