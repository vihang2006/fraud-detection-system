import requests
import random
import time

API_URL = "http://localhost:8000/predict"

def generate_transaction():
    return {
        "features": [random.uniform(-3, 3) for _ in range(30)]
    }

def run_simulator():
    print("Starting Real-Time Fraud Simulator...")
    while True:
        data = generate_transaction()
        try:
            response = requests.post(API_URL, json=data)
            if response.status_code == 200:
                print("Transaction Sent:", response.json())
            else:
                print("Error:", response.status_code)
        except Exception as e:
            print("Connection error:", e)

        time.sleep(2)

if __name__ == "__main__":
    run_simulator()
