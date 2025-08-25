import sys
import os

try:
    # Add the root directory to sys.path
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

    # Try to import the required modules
    from banking_agent.agents.spendings_agent import SpendingAgent
    from banking_agent.agents.agent_router import EnhancedPersonalFinanceRouter

    print("Modules imported successfully!")

except ImportError as e:
    print(f"Error importing modules: {e}")
    sys.exit(1)
