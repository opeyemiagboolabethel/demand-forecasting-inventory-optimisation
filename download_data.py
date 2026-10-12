"""Retrieve the pinned, licensed UCI transaction archive."""
import hashlib
import json
import urllib.request
from pathlib import Path
ROOT = Path(__file__).resolve().parent
if __name__ == '__main__':
    m = json.loads((ROOT / 'data/source_manifest.json').read_text())
    p = ROOT / 'data/raw/online_retail_ii.zip'
    p.parent.mkdir(parents=True, exist_ok=True)
    if not p.exists():
        b = urllib.request.urlopen(m['url'], timeout=180).read()
        if hashlib.sha256(b).hexdigest() != m['sha256']:
            raise ValueError('Source archive differs from the pinned snapshot')
        p.write_bytes(b)
    if hashlib.sha256(p.read_bytes()).hexdigest() != m['sha256']:
        raise ValueError('Source checksum mismatch')
    print('UCI archive verified')
