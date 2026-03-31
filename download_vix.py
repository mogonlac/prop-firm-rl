"""Download VIX daily close from Yahoo Finance and save to data/vix.csv"""
import urllib.request, json, csv, os, time

_HERE = os.path.dirname(os.path.abspath(__file__))
OUT   = os.path.join(_HERE, 'data', 'vix.csv')

url = (
    "https://query1.finance.yahoo.com/v8/finance/chart/%5EVIX"
    "?interval=1d&period1=1577836800&period2=1800000000"
    # period1 = 2020-01-01 UTC,  period2 = far future
)

req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
r   = urllib.request.urlopen(req, timeout=15)
data = json.loads(r.read())

chart     = data["chart"]["result"][0]
timestamps = chart["timestamp"]
closes     = chart["indicators"]["quote"][0]["close"]

rows = []
for ts, c in zip(timestamps, closes):
    if c is None:
        continue
    import datetime
    date = datetime.datetime.utcfromtimestamp(ts).strftime("%Y-%m-%d")
    rows.append((date, round(c, 4)))

os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["date", "vix_close"])
    w.writerows(rows)

print(f"Saved {len(rows)} rows -> {OUT}")
print(f"  Date range: {rows[0][0]} to {rows[-1][0]}")
print(f"  VIX range:  {min(c for _,c in rows):.1f} to {max(c for _,c in rows):.1f}")
