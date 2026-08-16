import streamlit as st
from agent import run_agent
from verification import verify_grounding

st.set_page_config(page_title="Grounded Data Analyst", layout="wide")

st.title("Grounded Data Analyst Agent")
st.markdown("Ask natural language business questions. The agent writes SQL, executes it against the database, self corrects on errors, and grounds its final answer.")

question = st.text_input("Enter your business question:", placeholder="e.g. How many total orders are there?")

if st.button("Run Agent") and question:
    with st.spinner("Agent is working..."):
        result = run_agent(question)
        
        st.subheader("Final Answer")
        if "Error:" in result["final_answer"]:
            st.error(result["final_answer"])
        else:
            st.success(result["final_answer"])
            
        is_grounded, unsupported = verify_grounding(result['final_answer'], result['steps'])
        
        if is_grounded:
            st.success("Hallucination Guard: PASSED (all numbers verified against tool output)")
        else:
            st.error(f"Hallucination Guard: TRIGGERED. Unsupported numbers: {unsupported}")
            
        st.subheader("Reasoning Trace")
        st.text(f"Latency: {result['latency_seconds']}s | Tool Calls: {result['tool_calls_count']} | Retries: {result['retry_count']}")
        
        for i, step in enumerate(result["steps"]):
            step_type = step["type"]
            with st.expander(f"Step {i+1}: {step_type.upper()}", expanded=(step_type != "tool_result")):
                if step_type == "thought":
                    st.markdown(step["content"])
                elif step_type == "tool_call":
                    if "query" in step:
                        st.code(step["query"], language="sql")
                    elif "tool_name" in step:
                        st.text(f"Called: {step['tool_name']}()")
                elif step_type == "tool_result":
                    st.text(step["result"])
                elif step_type == "error":
                    st.error(step["error"])
