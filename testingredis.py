import redis
from dotenv import load_dotenv
import os

load_dotenv()

try:
    redis_client = redis.from_url(os.getenv("REDIS_URL"), decode_responses=True)
    redis_client.ping()
    print("✅ Redis connection successful!")
    
    # Test basic operations
    redis_client.set("test_key", "Hello from Redis Cloud!")
    result = redis_client.get("test_key")
    print(f"✅ Test write/read: {result}")
    
    # Clean up
    redis_client.delete("test_key")
    print("✅ Redis ready for memory management!")
    
except Exception as e:
    print(f"❌ Redis connection failed: {e}")