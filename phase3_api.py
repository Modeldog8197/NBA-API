"""Launch the NBA API dashboard. See README.md for configuration."""
import logging

from nba_app.api import create_app
from nba_app.config import Settings

app = create_app(Settings.from_env(local_default=True))

if __name__ == "__main__":
    import uvicorn
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    uvicorn.run(app, host="127.0.0.1", port=8000)
