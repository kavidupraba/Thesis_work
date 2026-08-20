import sys, io, json, urllib.request
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
req = urllib.request.Request("https://api.github.com/repos/MiKTeX/miktex/releases/latest", headers={"User-Agent": "Mozilla/5.0"})
data = json.loads(urllib.request.urlopen(req, timeout=15).read())
print("Release:", data.get("tag_name", "?"))
for a in data.get("assets", []):
    name = a.get("name", "")
    if ".exe" in name.lower():
        size_mb = round(a["size"] / 1024 / 1024, 1)
        print(f"  {name} ({size_mb} MB)")
        print(f"  URL: {a['browser_download_url']}")
