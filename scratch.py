import httpx
try:
    raise httpx.ReadTimeout("")
except Exception as e:
    print(f"Error is: ({e})")
