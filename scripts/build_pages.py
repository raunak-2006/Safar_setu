import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app import app


response = app.test_client().get("/")
if response.status_code != 200:
    raise SystemExit(f"Flask page rendering failed with status {response.status_code}.")
html = response.get_data(as_text=True)
html = html.replace('/static/style.css', 'style.css')
html = html.replace('/static/app.js', 'app.js')
html = html.replace('/static/travel-background.jpg', './travel-background.jpg')
html = html.replace('href="/"', 'href="./"')
html = html.replace('<body>', '<body data-static-demo="true">', 1)
html = html.replace('40+ languages', '40+ language options')
html = html.replace('On-device translation', 'Hindi phrasebook')
(ROOT / "index.html").write_text(html, encoding="utf-8")

for asset_name in ("app.js", "style.css", "travel-background.jpg"):
    asset = ROOT / "static" / asset_name
    contents = asset.read_bytes()
    if asset_name == "style.css":
        contents = contents.replace(
            b'url("/static/travel-background.jpg")',
            b'url("./travel-background.jpg")',
        )
    (ROOT / asset_name).write_bytes(contents)