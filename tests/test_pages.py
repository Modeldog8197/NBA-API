import json

import pytest

from scripts.build_pages import build, normalize_api_url


def test_pages_package_contains_dashboard_and_no_repository_files(tmp_path):
    build(tmp_path)
    assert sorted(p.relative_to(tmp_path).as_posix() for p in tmp_path.rglob('*') if p.is_file()) == [
        '.nojekyll', 'index.html', 'static/app.js', 'static/runtime-config.js', 'static/styles.css']
    html = (tmp_path / 'index.html').read_text(encoding='utf-8')
    assert 'id="court"' in html
    assert 'href="./static/styles.css' in html
    assert 'href="./"' in html
    assert 'data-api-path="/docs"' in html
    assert '"mode": "pages"' in (tmp_path / 'static/runtime-config.js').read_text()


def test_pages_backend_config_is_explicit_and_serialized(tmp_path):
    build(tmp_path, 'https://api.example.org/nba/')
    source = (tmp_path / 'static/runtime-config.js').read_text()
    config = json.loads(source.removeprefix('window.NBA_DEPLOYMENT = ').rstrip(';\n'))
    assert config == {'mode': 'pages', 'apiBaseUrl': 'https://api.example.org/nba'}


@pytest.mark.parametrize('url', ['http://api.example.org', 'https://localhost',
                                 'https://127.0.0.1:8000', 'https://u:p@api.example.org',
                                 'https://api.example.org?secret=x', 'https://api.example.org#x'])
def test_public_pages_does_not_point_at_a_local_or_credentialed_backend(url):
    with pytest.raises(ValueError):
        normalize_api_url(url)
