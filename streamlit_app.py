import numpy as np
import streamlit as st
from llama_index.llms.openai import OpenAI
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.core.chat_engine import CondenseQuestionChatEngine
from trulens_eval import TruLlama

from trulens_eval.feedback.provider import OpenAI as tru_OpenAI
from trulens_eval import Feedback


st.set_page_config(page_title="Chat with the Streamlit docs, powered by LlamaIndex", page_icon="🦙", menu_items=None)
from trulens_eval import streamlit as trulens_st
import tru_st

st.title("Chat with the Streamlit docs, powered by LlamaIndex 💬🦙")

# Hide running man due to feedback reruns
st.html(
    """
    <style>
    [data-testid="stStatusWidget"] {
            visibility: hidden;
            height: 0%;
            position: fixed;
        }
    </style>
    """,
)

if "messages" not in st.session_state:  # Initialize the chat messages history
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Ask me a question about Streamlit's open-source Python library!",
        }
    ]
messages = st.session_state.messages

@st.cache_resource(show_spinner="Setting up assistant")
def build_query_engine():
    reader = SimpleDirectoryReader(input_dir="./data", recursive=True)
    docs = reader.load_data()
    Settings.llm = OpenAI(
        model="gpt-4o-mini",
        temperature=0.2,
        system_prompt="""You are an expert on 
        the Streamlit Python library and your 
        job is to answer technical questions. 
        Assume that all questions are related 
        to the Streamlit Python library. Keep 
        your answers technical and based on 
        facts – do not hallucinate features.""",
    )
    index = VectorStoreIndex.from_documents(docs)
    return index.as_query_engine()

# Query engine can be shared across all sessions
query_engine = build_query_engine()

# New chat_engine per session for unique history
if "chat_engine" not in st.session_state:
    st.session_state.chat_engine = CondenseQuestionChatEngine.from_defaults(
        query_engine=query_engine,
    )
chat_engine = st.session_state.chat_engine


@st.cache_resource(show_spinner="Configuring eval")
def build_recorder():
    # Initialize provider class
    provider = tru_OpenAI(model_engine="gpt-4o-mini")

    # select context to be used in feedback. the location of context is app specific.
    from trulens_eval.app import App
    context = App.select_context(query_engine)

    # Define a groundedness feedback function
    f_groundedness = (
        Feedback(provider.groundedness_measure_with_cot_reasons, name = "Groundedness")
        .on(context.collect()) # collect context chunks into a list
        .on_output()
    )

    # Question/answer relevance between overall question and answer.
    f_answer_relevance = (
        Feedback(provider.relevance_with_cot_reasons, name = "Answer Relevance")
        .on_input_output()
    )
    # Question/statement relevance between question and each context chunk.
    f_context_relevance = (
        Feedback(provider.context_relevance_with_cot_reasons, name = "Context Relevance")
        .on_input()
        .on(context)
        .aggregate(np.mean)
    )
    # Create the final recorder
    return TruLlama(query_engine,
            app_id='LlamaIndex_App1',
            feedbacks=[f_groundedness, f_answer_relevance, f_context_relevance],
        )

tru_recorder = build_recorder()

def draw_eval(record):
    with st.expander("View eval and trace"):
        tru_st.trulens_feedback(record=record, key_suffix=record.main_input)
        trulens_st.trulens_trace(record=record)

for message in messages:  # Write message history to UI
    with st.chat_message(message["role"]):
        st.write(message["content"])
        if "record" in message:
            draw_eval(message["record"])


if prompt := st.chat_input("Ask a question"):
    messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    with st.chat_message("assistant"):
        with tru_recorder as recording:
            response = chat_engine.chat(prompt)
        record = recording.records[-1]
        message = {
                "role": "assistant",
                "content": response.response,
                "record": record,
            }
        messages.append(message)
        st.write(response.response)
        draw_eval(record)
