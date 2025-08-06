#!/usr/bin/env python3
"""
Simple step-by-step test script for the personal finance system
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_imports():
    """Test that all imports work"""
    print("🔍 Testing imports...")
    
    try:
        from langgraph.graph import StateGraph, END
        from langgraph.checkpoint.memory import MemorySaver
        print("✅ LangGraph imports working")
    except ImportError as e:
        print(f"❌ LangGraph import failed: {e}")
        return False
    
    try:
        from langchain_core.messages import AIMessage, HumanMessage
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_openai import ChatOpenAI
        print("✅ LangChain imports working")
    except ImportError as e:
        print(f"❌ LangChain import failed: {e}")
        return False
    
    try:
        from pydantic import BaseModel, Field
        print("✅ Pydantic import working")
    except ImportError as e:
        print(f"❌ Pydantic import failed: {e}")
        return False
    
    return True

def test_data_store():
    """Test data store initialization"""
    print("\n📊 Testing data store...")
    
    try:
        from data_store.data_store import DataStore
        
        client_csv = "Banking_Data.csv"
        overall_csv = "overall_data.csv"
        
        if not os.path.exists(client_csv):
            print(f"❌ Client CSV not found: {client_csv}")
            return False
            
        if not os.path.exists(overall_csv):
            print(f"❌ Overall CSV not found: {overall_csv}")
            return False
        
        ds = DataStore(
            client_csv_path=client_csv,
            overall_csv_path=overall_csv
        )
        
        print("✅ DataStore initialized successfully")
        
        # Test a simple query
        rows, cols = ds.execute_sql_query("SELECT COUNT(*) FROM client_transactions")
        print(f"✅ Database query works: {rows[0][0]} transactions")
        
        ds.close()
        return True
        
    except Exception as e:
        print(f"❌ DataStore test failed: {e}")
        return False

def test_spending_agent():
    """Test spending agent individually"""
    print("\n📊 Testing spending agent...")
    
    try:
        # Import with path handling
        sys.path.append('agents')
        from agents.spendings_agent import SpendingAgent
        
        agent = SpendingAgent(
            client_csv_path="Banking_Data.csv",
            overall_csv_path="overall_data.csv",
            memory=False
        )
        
        print("✅ SpendingAgent initialized")
        
        # Test a simple query
        result = agent.process_query(
            client_id=430,
            user_query="How much did I spend last month?"
        )
        
        print(f"✅ Query processed: Success = {result['success']}")
        if result['success']:
            print(f"📝 Response preview: {result['response'][:100]}...")
        else:
            print(f"❌ Error: {result.get('error', 'Unknown')}")
            
        return result['success']
        
    except Exception as e:
        print(f"❌ SpendingAgent test failed: {e}")
        return False

def test_budget_agent():
    """Test budget agent individually"""
    print("\n💰 Testing budget agent...")
    
    try:
        from agents.budget_agent import BudgetAgent
        
        agent = BudgetAgent(
            client_csv_path="Banking_Data.csv",
            overall_csv_path="overall_data.csv",
            memory=False
        )
        
        print("✅ BudgetAgent initialized")
        
        # Test a simple query
        result = agent.process_query(
            client_id=430,
            user_query="Create a $800 budget for groceries"
        )
        
        print(f"✅ Query processed: Success = {result['success']}")
        if result['success']:
            print(f"📝 Response preview: {result['response'][:100]}...")
        else:
            print(f"❌ Error: {result.get('error', 'Unknown')}")
            
        return result['success']
        
    except Exception as e:
        print(f"❌ BudgetAgent test failed: {e}")
        return False

def test_router():
    """Test the router (if individual agents work)"""
    print("\n🤖 Testing router...")
    
    try:
        from agents.agent_router import PersonalFinanceRouter
        
        router = PersonalFinanceRouter(
            client_csv_path="Banking_Data.csv",
            overall_csv_path="overall_data.csv",
            enable_memory=False  # Disable memory for testing
        )
        
        print("✅ Router initialized")
        
        # Test a simple query
        result = router.chat(
            client_id=430,
            user_query="How much did I spend last month?"
        )
        
        print(f"✅ Router query processed: Success = {result['success']}")
        print(f"🎯 Routed to: {result['agent_used']}")
        
        if result['success']:
            print(f"📝 Response preview: {result['response'][:100]}...")
        else:
            print(f"❌ Error: {result.get('error', 'Unknown')}")
            
        return result['success']
        
    except Exception as e:
        print(f"❌ Router test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 PERSONAL FINANCE SYSTEM TESTS")
    print("=" * 50)
    
    os.chdir('/Users/mohibalikhan/Desktop/banking-agent/banking_agent')
    print(f"📁 Working directory: {os.getcwd()}")
    
    tests = [
        ("Imports", test_imports),
        ("Data Store", test_data_store),
        ("Spending Agent", test_spending_agent),
        ("Budget Agent", test_budget_agent),
        ("Router", test_router)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
            if not result:
                print(f"\n⚠️ {test_name} test failed - stopping here")
                break
        except Exception as e:
            print(f"\n💥 {test_name} test crashed: {e}")
            results.append((test_name, False))
            break
    
    print("\n📊 TEST SUMMARY")
    print("=" * 30)
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
    
    if all(result for _, result in results):
        print("\n🎉 All tests passed! System is ready to use.")
        print("▶️ Run: python agents/agent_router.py")
    else:
        print("\n🔧 Some tests failed. Check the errors above.")

if __name__ == "__main__":
    main()