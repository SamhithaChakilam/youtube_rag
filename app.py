import streamlit as st
from rag import (
    extract_video_id,
    build_video_index,
    retrieve_relevant_chunks,
    generate_answer
)


# =========================================================
# STREAMLIT PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="YouTube RAG Assistant",
    page_icon="🎥",
    layout="centered"
)

st.title("🎥 YouTube RAG Assistant")
st.write("Ask questions about a YouTube video without watching the whole thing.")


# =========================================================
# USER INPUTS
# =========================================================

youtube_url = st.text_input(
    "🔗 YouTube Video URL",
    placeholder="Paste YouTube URL here..."
)

question = st.text_input(
    "Ask a question",
    placeholder="Ask anything about the video..."
)

ask_button = st.button(
    "Ask Question",
    use_container_width=True
)


# =========================================================
# MAIN APPLICATION LOGIC
# =========================================================

if ask_button:

    # Validate inputs
    if not youtube_url:
        st.warning("Please enter a YouTube video URL.")
        st.stop()

    if not question:
        st.warning("Please enter a question.")
        st.stop()

    # Extract video ID
    video_id = extract_video_id(youtube_url)

    if not video_id:
        st.error("Please enter a valid YouTube URL.")
        st.stop()

    try:
        # Build index
        with st.spinner("Reading the video..."):
            video_index = build_video_index(video_id)

        # Retrieve relevant chunks
        ranked_documents = retrieve_relevant_chunks(
            question,
            video_index
        )

        # Generate answer
        with st.spinner("🤖 Thinking..."):
            answer = generate_answer(
                question,
                ranked_documents
            )

        # Display answer
        st.markdown("### 💬 Answer")
        st.write(answer)

    except Exception as e:
        st.error(f"Something went wrong: {str(e)}")