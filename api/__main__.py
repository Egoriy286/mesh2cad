"""Run the local API and its serialized background worker."""

import uvicorn

if __name__ == "__main__":
    uvicorn.run("api.app:create_app", factory=True, host="127.0.0.1", port=8765, workers=1)
