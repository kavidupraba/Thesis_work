import os, sys, json
sys.path.insert(0, r"C:\Users\Admin\Documents\thesis_sending\PythonProject")
import app as A

c = A.app.test_client()

r = c.get("/")
assert r.status_code == 200 and b"Explanation Browser" in r.data
print("GET /            -> 200, HTML ok")

r = c.get("/api/dates?stock=AAPL")
d = r.get_json()
ok_date = d["dates"][len(d["dates"]) // 2]
print("GET /api/dates   ->", len(d["dates"]), "dates, sample", ok_date)

r = c.get(f"/api/case?stock=AAPL&model=RandomForest&date={ok_date}")
case = r.get_json()
assert "error" not in case, case
assert len(case["top_features"]) == 5
print("GET /api/case    ->", case["predicted"], "prob", case["prob_up"],
      "| top1:", case["top_features"][0]["feature"])

r = c.get(f"/api/waterfall?stock=AAPL&model=RandomForest&date={ok_date}")
assert r.status_code == 200 and r.mimetype == "image/png"
print("GET /api/waterfall -> 200, png,", len(r.data), "bytes")

r = c.get(f"/api/case?stock=AAPL&model=RandomForest&date=2000-01-01")
assert "error" in r.get_json()
print("GET /api/case (bad date) -> error ok")

r = c.get("/api/case?stock=NOPE&model=XGBoost&date=2025-01-01")
assert "error" in r.get_json()
print("GET /api/case (bad stock) -> error ok")

ex = None
for k in A.llm_cache:
    ex = k
    break
r = c.get(f"/api/explain?stock={ex[0]}&model={ex[1]}&date={ex[2]}")
out = r.get_json()
assert out.get("source") == "cached" and len(out["text"]) > 50
print(f"GET /api/explain (cached {ex[0]} {ex[1]} {ex[2]}) -> {out['source']}")

if not A.API_KEY:
    r = c.get(f"/api/explain?stock=AAPL&model=RandomForest&date={ok_date}")
    assert "GEMINI_API_KEY" in r.get_json()["error"]
    print("GET /api/explain (uncached, no key) -> graceful message ok")
else:
    print("Gemini key present (env or ~/.gemini_api_key); skipping no-key branch")

print("\nALL SMOKE TESTS PASSED")