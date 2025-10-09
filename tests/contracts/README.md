# Contract Tests

These tests validate the API contracts between this service (provider)
and its consumers (other services, frontends, or clients).

Typical setup:
- Use **pact-python** to define consumer expectations (requests/responses).
- Run provider verification to ensure compatibility.
