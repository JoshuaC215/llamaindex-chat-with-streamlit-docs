import numpy as np
import streamlit as st
from llama_index.llms.openai import OpenAI
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from trulens_eval import TruLlama
from trulens_eval import streamlit as trulens_st
import tru_st
from trulens_eval.feedback.provider import OpenAI as tru_OpenAI
from trulens_eval import Feedback


st.set_page_config(page_title="Chat with the Streamlit docs, powered by LlamaIndex", page_icon="🦙", menu_items=None)
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
def build_engine():
    reader = SimpleDirectoryReader(input_dir="./data", recursive=True)
    docs = reader.load_data()
    Settings.llm = OpenAI(
        model="gpt-3.5-turbo",
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

chat_engine = build_engine()

@st.cache_resource(show_spinner="Configuring eval")
def build_recorder():
    # Initialize provider class
    provider = tru_OpenAI()

    # select context to be used in feedback. the location of context is app specific.
    from trulens_eval.app import App
    context = App.select_context(chat_engine)

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
    return TruLlama(chat_engine,
            app_id='LlamaIndex_App1',
            feedbacks=[f_groundedness, f_answer_relevance, f_context_relevance]
        )

tru_recorder = build_recorder()

def draw_eval(record):
    with st.expander("View eval and trace"):
        tru_st.trulens_feedback(record=record, key_suffix=record.main_input)
        trulens_st.trulens_trace(record=record)

for message in st.session_state.messages:  # Write message history to UI
    with st.chat_message(message["role"]):
        st.write(message["content"])
        if "record" in message:
            draw_eval(message["record"])


if prompt := st.chat_input("Ask a question"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            with tru_recorder as recording:
                response = chat_engine.query(prompt)
                st.write(response.response)
                record = recording.get()
        draw_eval(record)
        message = {"role": "assistant", "content": response.response, "record": record}
        st.session_state.messages.append(message)
