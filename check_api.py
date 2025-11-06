"""
🔍 NASA API Rate Limit Checker
Checks current rate limit status for NeoWs and SBDB APIs
"""

import requests
import time
import json

def check_nasa_rate_limits():
    """Check current rate limit status for NASA APIs"""
    
    print("🔍 Checking NASA API Rate Limits...")
    print("=" * 50)
    
    # NASA API Key (use DEMO_KEY if none provided)
    api_key = "DEMO_KEY"  # Using demo key for testing
    
    # Test NeoWs API
    print("\n📡 Testing NeoWs API...")
    neows_url = "https://api.nasa.gov/neo/rest/v1/neo/browse"
    params = {"api_key": api_key, "page": 0, "size": 1}
    
    try:
        start_time = time.time()
        response = requests.get(neows_url, params=params, timeout=10)
        response_time = (time.time() - start_time) * 1000
        
        print(f"   Status Code: {response.status_code}")
        print(f"   Response Time: {response_time:.0f}ms")
        
        if response.status_code == 200:
            print("   ✅ NeoWs API: AVAILABLE")
            # Check rate limit headers
            headers = dict(response.headers)
            print("   Response Headers:")
            for key, value in headers.items():
                if any(term in key.lower() for term in ['limit', 'remaining', 'rate']):
                    print(f"     {key}: {value}")
        elif response.status_code == 429:
            print("   🚨 NeoWs API: RATE LIMITED")
            retry_after = response.headers.get('Retry-After', 'Unknown')
            print(f"   Retry After: {retry_after} seconds")
        elif response.status_code == 403:
            print("   🔒 NeoWs API: FORBIDDEN (API Key issue)")
        else:
            print(f"   ❌ NeoWs API: ERROR {response.status_code}")
            
    except requests.exceptions.Timeout:
        print("   ⏰ NeoWs API: TIMEOUT")
    except requests.exceptions.ConnectionError:
        print("   🔌 NeoWs API: CONNECTION ERROR")
    except Exception as e:
        print(f"   💥 NeoWs API: UNKNOWN ERROR - {e}")

    # Test SBDB API
    print("\n📡 Testing JPL SBDB API...")
    sbdb_url = "https://ssd-api.jpl.nasa.gov/sbdb.api"
    params = {"des": "433"}  # Using asteroid 433 Eros as test
    
    try:
        start_time = time.time()
        response = requests.get(sbdb_url, params=params, timeout=10)
        response_time = (time.time() - start_time) * 1000
        
        print(f"   Status Code: {response.status_code}")
        print(f"   Response Time: {response_time:.0f}ms")
        
        if response.status_code == 200:
            print("   ✅ SBDB API: AVAILABLE")
            data = response.json()
            if 'object' in data:
                print(f"   Test Object: {data['object'].get('fullname', 'Unknown')}")
        elif response.status_code == 429:
            print("   🚨 SBDB API: RATE LIMITED")
        elif response.status_code == 404:
            print("   ✅ SBDB API: AVAILABLE (404 is normal for some objects)")
        else:
            print(f"   ❌ SBDB API: ERROR {response.status_code}")
            
    except requests.exceptions.Timeout:
        print("   ⏰ SBDB API: TIMEOUT")
    except requests.exceptions.ConnectionError:
        print("   🔌 SBDB API: CONNECTION ERROR")
    except Exception as e:
        print(f"   💥 SBDB API: UNKNOWN ERROR - {e}")

    # Test with your actual API key if available
    actual_api_key = "SEsoH1p8IBnZg44ePhFNPLKtcIHXgIQy3uriPjrc"  # Replace with your key
    if actual_api_key and actual_api_key != "YOUR_ACTUAL_API_KEY_HERE":
        print(f"\n🔑 Testing with your API key: {actual_api_key[:8]}...")
        
        # Test NeoWs with your key
        params = {"api_key": actual_api_key, "page": 0, "size": 1}
        try:
            response = requests.get(neows_url, params=params, timeout=10)
            print(f"   Your Key - NeoWs: {'✅ OK' if response.status_code == 200 else '❌ FAILED'}")
        except:
            print("   Your Key - NeoWs: ❌ FAILED")
        
        # Note: SBDB doesn't use API keys

def continuous_monitor(interval_seconds=60):
    """Continuously monitor API status"""
    print(f"\n🔄 Starting continuous monitoring (every {interval_seconds}s)")
    print("Press Ctrl+C to stop...")
    
    try:
        while True:
            check_nasa_rate_limits()
            print(f"\n⏰ Waiting {interval_seconds} seconds...")
            print("=" * 50)
            time.sleep(interval_seconds)
    except KeyboardInterrupt:
        print("\n🛑 Monitoring stopped")

if __name__ == "__main__":
    # Single check
    check_nasa_rate_limits()
    continuous_monitor(interval_seconds=60)