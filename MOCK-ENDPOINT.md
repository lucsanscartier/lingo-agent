# Mock API Endpoint

This document defines a mock API endpoint to validate the `routeAI()` integration. The mock is a local HTTP server providing placeholder model completions.

## Endpoint Details
- **URL:** `http://localhost:5000/api/mock`
- **Method:** POST
- **Authentication:** Fake bearer token (`Token-AbCdEf123456`)

## Example cURL Command
```bash
curl -X POST "http://localhost:5000/api/mock" \
     -H "Authorization: Bearer Token-AbCdEf123456" \
     -H "Content-Type: application/json" \
     -d '{"prompt": "Write a poem about the stars."}'
```

### Expected Output
```json
{
  "response": "The stars shine bright in the evening light,\ntelling stories of infinite delight."
}
```