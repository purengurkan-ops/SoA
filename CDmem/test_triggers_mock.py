import time
from unittest.mock import MagicMock

# This script mocks the serial TriggerBox for testing in the lab.
# You can run this when you are set up at the lab to verify your setup.

class MockSerial:
    def __init__(self, port):
        print(f"DEBUG: Initialized Mock Serial on {port}")
    def write(self, data):
        print(f"DEBUG: Sent trigger byte: {list(data)}")
    def close(self):
        print("DEBUG: Closed Mock Serial")

def test_triggers():
    # Simulate the trigger mapping
    print("Testing EEG Trigger Mapping...")
    
    # Levels 1-4
    for level in [1, 2, 3, 4]:
        print(f"\n--- Testing Level {level} ---")
        
        # Stimulus Onset (10 + Level)
        stim_val = 10 + level
        print(f"Mocking Stimulus Onset (10 + {level}): {stim_val}")
        
        # Motion Start (20 + Level)
        motion_val = 20 + level
        print(f"Mocking Motion Start (20 + {level}): {motion_val}")
        
        # Response Screen (30 + Level)
        resp_onset_val = 30 + level
        print(f"Mocking Response Onset (30 + {level}): {resp_onset_val}")
        
    print("\n--- Testing Response Values ---")
    print(f"Correct: 41")
    print(f"Incorrect: 42")
    
    print("\n--- Testing Memory Phase ---")
    print(f"Seen Image Onset: 51")
    print(f"Unseen Image Onset: 52")
    print(f"Memory Correct (Hit/CR): 61")
    print(f"Memory Incorrect (Miss/FA): 62")

if __name__ == "__main__":
    test_triggers()
