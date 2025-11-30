import requests
import time
import pandas as pd
import json

def test_yahoo(symbol):
    print(f"Testing {symbol}...")
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    end_time = int(time.time())
    start_time = end_time - (365 * 24 * 60 * 60)
    
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
    params = {
        'period1': start_time,
        'period2': end_time,
        'interval': '1d',
        'events': 'history'
    }
    
    try:
        response = requests.get(url, headers=headers, params=params, timeout=15)
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            try:
                data = response.json()
                # print(json.dumps(data, indent=2)[:500]) # Print first 500 chars
                
                if 'chart' in data and 'result' in data['chart']:
                    result = data['chart']['result']
                    if result:
                        print("Has result")
                        if 'timestamp' in result[0]:
                            print(f"Has {len(result[0]['timestamp'])} timestamps")
                        else:
                            print("No timestamps in result")
                            print(result[0].keys())
                    else:
                        print("Result is empty or None")
                        print(data)
                else:
                    print("Invalid JSON structure")
                    print(data.keys())
            except Exception as e:
                print(f"JSON Decode Error: {e}")
                print(response.text[:500])
        else:
            print("Request failed")
            print(response.text[:500])
            
    except Exception as e:
        print(f"Exception: {e}")

test_yahoo("AAPL")
