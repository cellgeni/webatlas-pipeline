#!/usr/bin/env python3
"""
Wrapper for router.py that sets Numba environment variables
BEFORE any imports that might trigger Numba compilation.
"""
import os
import sys

# Set Numba environment variables BEFORE any other imports
os.environ['NUMBA_DISABLE_JIT'] = '1'
os.environ['NUMBA_DISABLE_CACHING'] = '1'
os.environ['NUMBA_CACHE_DIR'] = '/tmp/numba_cache'
os.environ['PYTHONNOUSERSITE'] = '1'

# Ensure cache directory exists
os.makedirs('/tmp/numba_cache', exist_ok=True)

# Now we can safely import and run the actual router
# Get the directory where this wrapper is located
wrapper_dir = os.path.dirname(os.path.abspath(__file__))

# Add the bin directory to the path if it's not already there
if wrapper_dir not in sys.path:
    sys.path.insert(0, wrapper_dir)

# Import the actual router module
import router

# Run the router's main function
if __name__ == '__main__':
    # The router.py script should have been designed to run as main
    # Since we can't directly exec it, we need to import and call it
    # If router.py has a main() function, call it
    # Otherwise, we need to use exec
    
    # Read and execute the router.py script
    router_path = os.path.join(wrapper_dir, 'router.py')
    with open(router_path, 'r') as f:
        code = f.read()
    
    # Execute in the current namespace
    exec(code, {'__name__': '__main__', '__file__': router_path})