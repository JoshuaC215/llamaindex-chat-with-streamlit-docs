import asyncio

# https://github.com/jerryjliu/llama_index/issues/7244:
asyncio.set_event_loop(asyncio.new_event_loop())

import streamlit as st
from streamlit_pills import pills
from trulens_eval.schema.record import Record
from trulens_eval.utils import display
from trulens_eval.schema.feedback import FeedbackCall
from pydantic import BaseModel
from typing import List


class FeedbackDisplay(BaseModel):
    score: float = 0
    calls: List[FeedbackCall]
    icon: str


@st.experimental_fragment(run_every=2)
def trulens_feedback(record: Record, key_suffix=""):
    """
    Render clickable feedback pills for a given record.
    Args:
        record (Record): A trulens record.
    !!! example
        ```python
        from trulens_eval import streamlit as trulens_st
        with tru_llm as recording:
            response = llm.invoke(input_text)
        record, response = recording.get()
        trulens_st.trulens_feedback(record=record)
        ```
    """
    feedback_cols = []
    feedbacks = {}
    icons = []
    for feedback, feedback_result in record.wait_for_feedback_results().items():
        call_data = {
            'feedback_definition': feedback,
            'feedback_name': feedback.name,
            'result': feedback_result.result
        }
        feedback_cols.append(call_data['feedback_name'])
        feedbacks[call_data['feedback_name']] = FeedbackDisplay(
            score=call_data['result'],
            calls=[],
            icon=display.get_icon(fdef=feedback, result=feedback_result.result)
        )
        icons.append(feedbacks[call_data['feedback_name']].icon)

    st.write('**Feedback functions**')
    selected_feedback = pills(
        "Feedback functions",
        feedback_cols,
        index=None,
        format_func=lambda fcol: f"{fcol} {feedbacks[fcol].score:.4f}",
        label_visibility=
        "collapsed",  # Hiding because we can't format the label here.
        icons=icons,
        key=
        f"{call_data['feedback_name']}_{len(feedbacks)}_{key_suffix}"  # Important! Otherwise streamlit sometimes lazily skips update even with st.experimental_fragment
    )

    if selected_feedback is not None:
        st.dataframe(
            display.get_feedback_result(
                record, feedback_name=selected_feedback
            ),
            use_container_width=True,
            hide_index=True
        )