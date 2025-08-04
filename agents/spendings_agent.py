import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, TypedDict, Annotated
from datetime import datetime
import json
import operator
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver

from banking_agent.data_store.data_store import (
    DataStore,
    generate_sql_for_client_analysis,
    generate_sql_for_benchmark_analysis,
    execute_generated_sql
)


class SpendingAgentState(TypedDict):
    """Enhanced state for SQL-first Spending Agent workflow"""
    client_id: int
    user_query: str
    intent: Optional[Dict[str, Any]]
    sql_queries: Optional[List[Dict[str, Any]]]      
    raw_data: Optional[List[Dict[str, Any]]]           # NEW: Raw SQL results
    analysis_result: Optional[List[Dict[str, Any]]]    
    response: Optional[str]
    messages: Annotated[List[BaseMessage], operator.add]
    error: Optional[str]
    execution_path: List[str]
    analysis_type: Optional[str]                       


class SpendingAgent:
    """SQL-first LangGraph-based Spending Agent with benchmark capabilities"""

    def __init__(
        self,
        client_csv_path: str,
        overall_csv_path: str,
        model_name: str = "gpt-4o",
        memory: bool = True
    ):
        print("🚀 Initializing SpendingAgent with SQL-first approach...")
        print(f"📥 Client data: {client_csv_path}")
        print(f"📊 Overall data: {overall_csv_path}")

        self.data_store = DataStore(
            client_csv_path=client_csv_path,
            overall_csv_path=overall_csv_path
        )

        self.llm = ChatOpenAI(model=model_name, temperature=0)

        self.sql_tools = [
            generate_sql_for_client_analysis,
            generate_sql_for_benchmark_analysis,
            execute_generated_sql
        ]

        self.memory = SqliteSaver.from_conn_string(":memory:") if memory else None

        self.graph = self._build_graph()
        print("✅ SpendingAgent initialized with SQL-first capabilities!")

    def _build_graph(self) -> StateGraph:
        """Build the SQL-first LangGraph workflow"""
        workflow = StateGraph(SpendingAgentState)

        workflow.add_node("intent_classifier", self._intent_classifier_node)
        workflow.add_node("sql_generator",    self._sql_generator_node)
        workflow.add_node("sql_executor",     self._sql_executor_node)
        workflow.add_node("data_analyzer",    self._data_analyzer_node)
        workflow.add_node("response_generator", self._response_generator_node)
        workflow.add_node("error_handler",    self._error_handler_node)

        workflow.set_entry_point("intent_classifier")

        workflow.add_conditional_edges(
            "intent_classifier",
            self._route_after_intent,
            {"generate_sql": "sql_generator", "error": "error_handler"}
        )

        workflow.add_edge("sql_generator",      "sql_executor")
        workflow.add_edge("sql_executor",       "data_analyzer")
        workflow.add_edge("data_analyzer",      "response_generator")
        workflow.add_edge("response_generator", END)
        workflow.add_edge("error_handler",      END)

        return workflow.compile(checkpointer=self.memory)

    def _intent_classifier_node(self, state: SpendingAgentState) -> SpendingAgentState:
        """Enhanced intent classification with analysis type detection"""
        classification_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an advanced AI assistant for banking analysis. Classify user queries and determine the analysis approach.

                ANALYSIS TYPES:
                - "personal": Focus only on client’s personal spending patterns
                - "comparative": Compare client to market benchmarks/demographics
                - "hybrid": Both personal analysis AND market comparison

                Respond in JSON:
                {
                "analysis_type": "personal|comparative|hybrid",
                "requires_client_data": true|false,
                "requires_benchmark_data": true|false,
                "query_focus": "spending_summary|category_analysis|time_patterns|comparison",
                "time_period": "last_month|last_quarter|last_year|specific_dates|all_time",
                "confidence": 0.9
                }

                Classify this query:"""),
            ("human", "{user_query}")
        ])

        try:
            print("🧠 [DEBUG] Classifying:", state['user_query'])
            resp = self.llm.invoke(
                classification_prompt.format_messages(user_query=state['user_query'])
            )
            print("[DEBUG] LLM intent response:", resp.content)
            intent_data = json.loads(resp.content.strip())

            state['intent'] = intent_data
            state['analysis_type'] = intent_data.get('analysis_type', 'personal')
            state['execution_path'].append("intent_classifier")

            state['messages'].append(AIMessage(
                content=f"Classified as {state['analysis_type']} analysis. Generating SQL queries..."
            ))

        except json.JSONDecodeError as e:
            state['error'] = f"Intent classification JSON error: {e}"
        except Exception as e:
            state['error'] = f"Intent classification error: {e}"

        return state

    def _sql_generator_node(self, state: SpendingAgentState) -> SpendingAgentState:
        """Generate appropriate SQL queries based on intent"""
        try:
            print("🔧 [DEBUG] Generating SQL queries...")
            intent = state.get('intent', {}) or {}
            to_gen: List[str] = []
            if intent.get('requires_client_data', True):
                to_gen.append('client_analysis')
            if intent.get('requires_benchmark_data', False):
                to_gen.append('benchmark_analysis')

            sql_queries: List[Dict[str, Any]] = []
            for qtype in to_gen:
                if qtype == 'client_analysis':
                    client_sql = generate_sql_for_client_analysis.invoke({
                        'user_query': state['user_query'],
                        'client_id': state['client_id']
                    })
                    sql_queries.append(client_sql)
                else:
                    benchmark_sql = generate_sql_for_benchmark_analysis.invoke({
                        'user_query': state['user_query'],
                        'demographic_filters': self._get_client_demographics(state['client_id'])
                    })
                    sql_queries.append(benchmark_sql)

            state['sql_queries'] = sql_queries
            state['execution_path'].append("sql_generator")
            print(f"✅ Generated {len(sql_queries)} SQL queries")

        except Exception as e:
            state['error'] = f"SQL generation failed: {e}"
            print(f"❌ SQL generation error: {e}")

        return state

    def _sql_executor_node(self, state: SpendingAgentState) -> SpendingAgentState:
        """Execute generated SQL queries and collect raw data"""
        try:
            print("⚡ [DEBUG] Executing SQL queries...")
            raw = []
            for info in state.get('sql_queries', []) or []:
                sql = info.get('sql_query')
                if not sql:
                    continue
                exec_res = execute_generated_sql.invoke({
                    'sql_query': sql,
                    'query_type': info.get('query_type', 'unknown')
                })
                raw.append({
                    'query_type':    info.get('query_type'),
                    'original_query': info.get('original_query'),
                    'sql_executed':   sql,
                    'results':        exec_res.get('results', []),
                    'row_count':      exec_res.get('row_count', 0),
                    'error':          exec_res.get('error')
                })
                print(f" ✅ Executed {info.get('query_type')}: {exec_res.get('row_count', 0)} rows")

            state['raw_data'] = raw
            state['execution_path'].append("sql_executor")

        except Exception as e:
            state['error'] = f"SQL execution failed: {e}"
            print(f"❌ SQL execution error: {e}")

        return state

    def _data_analyzer_node(self, state: SpendingAgentState) -> SpendingAgentState:
        """Analyze raw SQL results using local processing (hybrid)"""
        try:
            print("📊 [DEBUG] Analyzing raw data...")
            analysis: List[Dict[str, Any]] = []
            for chunk in state.get('raw_data', []) or []:
                df = pd.DataFrame(chunk.get('results', []))
                if chunk.get('query_type') == 'client_analysis':
                    analysis.append({
                        'type': 'personal_analysis',
                        'data': self._analyze_personal_spending(df)
                    })
                else:
                    analysis.append({
                        'type': 'benchmark_analysis',
                        'data': self._analyze_benchmark_data(df)
                    })

            state['analysis_result'] = analysis
            state['execution_path'].append("data_analyzer")
            print(f"✅ Completed analysis: {len(analysis)} result sets")

        except Exception as e:
            state['error'] = f"Data analysis failed: {e}"
            print(f"❌ Analysis error: {e}")

        return state

    def _analyze_personal_spending(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze personal spending data with precise calculations"""
        if df.empty:
            return {"error": "No personal spending data available"}

        out: Dict[str, Any] = {}
        if 'amount' in df:
            out['spending_summary'] = {
                'total_amount': df['amount'].sum(),
                'transaction_count': df.shape[0],
                'average_transaction': df['amount'].mean(),
                'median_transaction': df['amount'].median(),
                'max_transaction': df['amount'].max(),
                'min_transaction': df['amount'].min()
            }
        if {'mcc_category', 'amount'}.issubset(df.columns):
            cat = (df
                   .groupby('mcc_category')['amount']
                   .agg(['sum', 'count', 'mean'])
                   .round(2)
                   .sort_values('sum', ascending=False))
            out['category_breakdown'] = {
                'top_categories': cat.head(5).to_dict('index'),
                'total_categories': len(cat)
            }
        if {'is_weekend', 'amount'}.issubset(df.columns):
            w = df.groupby('is_weekend')['amount'].agg(['sum', 'count', 'mean']).to_dict()
            out['time_patterns'] = {'weekend_vs_weekday': w}

        return out

    def _analyze_benchmark_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze benchmark data for comparisons"""
        if df.empty:
            return {"error": "No benchmark data available"}

        out: Dict[str, Any] = {}
        if 'amount' in df:
            out['market_benchmarks'] = {
                'market_average': df['amount'].mean(),
                'market_median': df['amount'].median(),
                'sample_size': df.shape[0]
            }
        if {'current_age', 'amount'}.issubset(df.columns):
            age_map = df.groupby('current_age')['amount'].mean().to_dict()
            out['demographic_patterns'] = {'spending_by_age': age_map}

        return out

    def _response_generator_node(self, state: SpendingAgentState) -> SpendingAgentState:
        """Generate comprehensive response combining personal and benchmark insights"""
        response_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert financial advisor. Create a comprehensive, actionable response based on the analysis results.

            RESPONSE STRUCTURE:
            1. **Direct Answer**: Address the user’s question first
            2. **Key Insights**: Highlight top findings with numbers
            3. **Comparisons**: Offer meaningful benchmarks if available
            4. **Recommendations**: 2–3 actionable tips

            Analysis Type: {analysis_type}"""),
                        ("human", """
            Original Query: {query}

            Analysis Results:
            {results}

            Please craft the final user-facing reply.
            """)
        ])

        try:
            results = state.get('analysis_result', []) or []
            results_json = json.dumps(results, indent=2, default=str)

            resp = self.llm.invoke(
                response_prompt.format_messages(
                    analysis_type=state.get('analysis_type', 'personal'),
                    query=state['user_query'],
                    results=results_json
                )
            )
            state['response'] = resp.content
            state['execution_path'].append("response_generator")

        except Exception as e:
            state['error'] = f"Response generation error: {e}"
            state['response'] = (
                "I analyzed your data but ran into a response error. "
                "Please try rephrasing or ask about: spending summary, "
                "category breakdown, or market comparison."
            )
        return state

    def _error_handler_node(self, state: SpendingAgentState) -> SpendingAgentState:
        """Enhanced error handling with helpful suggestions"""
        msg = state.get('error', 'Unknown error occurred')
        state['response'] = (
            f"I encountered an error: {msg}\n\n"
            "🔧 Try:\n"
            "- “Show me my spending summary for last month”\n"
            "- “How does my restaurant spending compare?”\n"
            "- “Break down my spending by category”"
        )
        state['execution_path'].append("error_handler")
        return state

    def _route_after_intent(self, state: SpendingAgentState) -> str:
        """Enhanced routing logic"""
        if state.get('error') or not state.get('intent'):
            return "error"
        return "generate_sql"

    def _get_client_demographics(self, client_id: int) -> Dict[str, Any]:
        """Get client demographics for benchmark filtering"""
        try:
            df = self.data_store.get_client_data(client_id)
            if not df.empty:
                row = df.iloc[0]
                return {
                    'age_min': max(18, row['current_age'] - 5),
                    'age_max': min(80, row['current_age'] + 5),
                    'gender': row['gender'],
                    'income_min': row['yearly_income'] * 0.8
                }
        except Exception:
            pass
        return {}

    def process_query(
        self,
        client_id: int,
        user_query: str,
        config: Dict = None
    ) -> Dict[str, Any]:
        """Process a spending query with SQL-first approach"""
        initial = SpendingAgentState(
            client_id=client_id,
            user_query=user_query,
            intent=None,
            sql_queries=None,
            raw_data=None,
            analysis_result=None,
            response=None,
            messages=[HumanMessage(content=user_query)],
            error=None,
            execution_path=[],
            analysis_type=None
        )

        try:
            final = self.graph.invoke(initial, config=config or {})
            return {
                "client_id": client_id,
                "query": user_query,
                "response": final.get('response'),
                "analysis_type": final.get('analysis_type'),
                "sql_queries": final.get('sql_queries'),
                "analysis_result": final.get('analysis_result'),
                "execution_path": final.get('execution_path'),
                "error": final.get('error'),
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            return {
                "client_id": client_id,
                "query": user_query,
                "response": f"Error processing request: {e}",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }

    def demo_spending_agent(self):
        """Demonstrate the enhanced SpendingAgent capabilities"""
        print("\n" + "="*60)
        print("🚀 SQL-FIRST SPENDING AGENT DEMO")
        print("="*60)

        examples = [
            (430, 'How much did I spend last month?')
            # (430, 'Show me my top spending categories'),
            # (430, 'How does my restaurant spending compare to others my age?'),
            # (430, 'Am I spending more than average on groceries?')
        ]

        for cid, qry in examples:
            print(f"\n🔍 Query: '{qry}' for client {cid}")
            print("-" * 50)
            try:
                res = self.process_query(client_id=cid, user_query=qry)
                print(f"✅ Analysis Type: {res.get('analysis_type')}")
                print(f"🔧 SQL Queries: {len(res.get('sql_queries') or [])}")
                print(f"🛤️ Path: {' → '.join(res.get('execution_path') or [])}")
                print(f"\n💬 Response:\n{res.get('response')}")
                if res.get('error'):
                    print(f"❌ Error: {res['error']}")
            except Exception as e:
                print(f"❌ Error processing '{qry}': {e}")
            print("\n" + "."*50)


if __name__ == "__main__":
    print("🚀 SQL-First SpendingAgent Demo")
    client_csv = "C:/Users/mohib.alikhan/Desktop/Banking-Agent/Banking_Data.csv"
    overall_csv = "C:/Users/mohib.alikhan/Desktop/Banking-Agent/overall_data.csv"

    try:
        agent = SpendingAgent(
            client_csv_path=client_csv,
            overall_csv_path=overall_csv,
            memory=False
        )
        agent.demo_spending_agent()
    except Exception as e:
        print(f"❌ Failed to initialize agent: {e}")
