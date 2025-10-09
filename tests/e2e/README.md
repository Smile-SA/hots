# End-to-End (E2E) Tests

E2E tests validate that the entire system works together as expected.

They often:
- Spin up the full stack (API, database, message broker) using Docker.
- Send realistic scheduling requests.
- Verify persistence, emitted events, and results.

Run with:
```bash
pytest -m e2e
