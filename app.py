"""Streamlit UI.  Run: streamlit run app.py"""
import streamlit as st

from src import pipeline

st.set_page_config(page_title="Ask-the-Policy", page_icon="📄", layout="wide")
st.title("📄 Ask-the-Policy")
st.caption("Question answering over Indian government policy documents. "
           "Answers cite their sources and say so when the answer isn't in the documents.")

with st.sidebar:
    st.header("Pipeline settings")
    mode = st.selectbox("Retrieval", ["hybrid_rerank", "hybrid", "dense", "bm25"])
    chunking = st.selectbox("Chunking", ["semantic", "fixed"])
    k = st.slider("Passages to use (k)", 1, 10, 5)
    strict = st.checkbox("Refuse if unsure (strict prompt)", value=True)

question = st.text_input("Your question", placeholder="Ask a question about the documents…")

if st.button("Ask", type="primary") and question.strip():
    try:
        with st.spinner("Searching documents…"):
            result = pipeline.ask(question, mode=mode, chunking=chunking, k=k, strict=strict)
    except FileNotFoundError as e:
        st.error(f"Index not built yet. {e}")
        st.stop()
    except RuntimeError as e:
        st.error(str(e))
        st.stop()
    except Exception as e:  # e.g. wrong model name or API/network problem
        st.error(f"Something went wrong calling the LLM or the pipeline: {e}")
        st.stop()

    st.subheader("Answer")
    (st.warning if result["refused"] else st.success)(result["answer"])

    st.subheader("Sources")
    for i, src in enumerate(result["sources"], start=1):
        cited = "  ✅ cited" if i in result["cited"] else ""
        with st.expander(f"[{i}] {src['doc']} — p.{src['page_start']}  (score {src['score']:.3f}){cited}"):
            st.write(src["text"])
