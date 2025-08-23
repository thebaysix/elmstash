"""
Test script to verify the threading fix for SQLite works correctly.
"""

import sys
import os
import threading
import time

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from observer.core import ModelObserver

def test_threading_fix():
    """Test that the observer works correctly in a multi-threaded environment."""
    
    print("🧵 Testing Threading Fix")
    print("=" * 40)
    
    # Create observer
    observer = ModelObserver("test_threading.sqlite")
    
    # Test function to run in different threads
    def thread_worker(thread_id):
        try:
            session_id = f"thread_{thread_id}"
            
            # Record interaction
            observer.record_interaction(
                session_id=session_id,
                step=1,
                input_str=f"Test input from thread {thread_id}",
                output_str=f"Test output from thread {thread_id}",
                action="thread_test"
            )
            
            # Get data back
            interactions = observer.get_session_data(session_id)
            
            print(f"✅ Thread {thread_id}: Recorded and retrieved {len(interactions)} interactions")
            return True
            
        except Exception as e:
            print(f"❌ Thread {thread_id}: Error - {e}")
            return False
    
    # Create and start multiple threads
    threads = []
    results = {}
    
    for i in range(5):
        def worker(tid=i):
            results[tid] = thread_worker(tid)
        
        thread = threading.Thread(target=worker)
        threads.append(thread)
        thread.start()
    
    # Wait for all threads to complete
    for thread in threads:
        thread.join()
    
    # Check results
    success_count = sum(1 for success in results.values() if success)
    total_count = len(results)
    
    print("=" * 40)
    if success_count == total_count:
        print(f"🎉 Threading test passed! {success_count}/{total_count} threads successful")
        print("✅ SQLite threading issue is fixed")
        return True
    else:
        print(f"❌ Threading test failed! Only {success_count}/{total_count} threads successful")
        return False

if __name__ == "__main__":
    success = test_threading_fix()
    if success:
        print("\n🚀 Demo should now work without threading errors!")
    else:
        print("\n🔧 Threading issues still exist - please check the implementation.")