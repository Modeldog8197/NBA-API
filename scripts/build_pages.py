"""Package only the dashboard for GitHub Pages; Python stays on its API host."""
import argparse
import json
import os
from pathlib import Path
import shutil
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parent.parent


def normalize_api_url(value: str) -> str:
    value = value.strip().rstrip('/')
    if not value:
        return ''
    parsed = urlsplit(value)
    if (parsed.scheme != 'https' or not parsed.hostname or parsed.username or parsed.password
            or parsed.query or parsed.fragment
            or parsed.hostname in {'localhost', '127.0.0.1', '::1'}):
        raise ValueError('NBA_API_URL must be a public HTTPS backend URL, without credentials, query, or fragment.')
    return value


def build(output: Path, api_url: str = ''):
    api_url = normalize_api_url(api_url)
    output.mkdir(parents=True, exist_ok=True)
    assets = output / 'static'
    assets.mkdir(exist_ok=True)
    shutil.copyfile(ROOT / 'static/index.html', output / 'index.html')
    for name in ('styles.css', 'app.js'):
        shutil.copyfile(ROOT / 'static' / name, assets / name)
    config = {'mode': 'pages', 'apiBaseUrl': api_url}
    (assets / 'runtime-config.js').write_text(
        'window.NBA_DEPLOYMENT = ' + json.dumps(config) + ';\n', encoding='utf-8')
    (output / '.nojekyll').write_text('', encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'artifacts/pages')
    args = parser.parse_args()
    build(args.output, os.getenv('NBA_API_URL', ''))
