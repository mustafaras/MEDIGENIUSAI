import sys
import os

# Print Python executable path
print("Python Executable:", sys.executable)
print("Python Version:", sys.version)
print("Python Path:")
for path in sys.path:
    print("  ", path)

# Check if key packages are available
packages_to_check = [
    'openai', 'anthropic', 'google.generativeai', 
    'ollama', 'streamlit', 'PIL', 'docx'
]

print("\nPackage Availability:")
for package in packages_to_check:
    try:
        __import__(package)
        print(f"  ✅ {package}: Available")
    except ImportError:
        print(f"  ❌ {package}: Not available")

# Create a simple test to verify everything works
print("\nPython Built-ins Test:")
try:
    result = len([1, 2, 3])
    print(f"  ✅ len() works: {result}")
    result = str(123)
    print(f"  ✅ str() works: {result}")
    result = int("456")
    print(f"  ✅ int() works: {result}")
    print("  ✅ All Python built-ins working correctly!")
except Exception as e:
    print(f"  ❌ Built-in functions error: {e}")
