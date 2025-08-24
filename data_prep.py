import sys
import os

sys.path.insert(
    0,
    os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            os.pardir
        )
    )
)
try:
    from agents.agent_router import EnhancedPersonalFinanceRouter
except ImportError:
    print("Could not import the Router")
