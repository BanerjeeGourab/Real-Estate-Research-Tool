import streamlit as st
from rag import process_urls, generate_answer

st.title("Real Estate Research Tool")

# Sidebar input for URLs
url1 = st.sidebar.text_input("URL 1")
url2 = st.sidebar.text_input("URL 2")
url3 = st.sidebar.text_input("URL 3")

urls = [url for url in (url1, url2, url3) if url]

# Track if URLs have been processed
if "urls_processed" not in st.session_state:
    st.session_state.urls_processed = False

# Placeholder for status updates
status_placeholder = st.empty()

# Button to process URLs
if st.sidebar.button("Process URLs"):
    if not urls:
        st.warning("You must provide at least one valid URL")
    else:
        st.session_state.urls_processed = False
        status_text = ""
        for status in process_urls(urls):
            status_text += status + "\n"
            status_placeholder.text(status_text)  # update in real-time
        st.session_state.urls_processed = True

# Question input
query = st.text_input(
    "Ask a Question:",
    disabled=not st.session_state.urls_processed
)

if not st.session_state.urls_processed:
    st.info("Please process URLs first to enable the question input.")

# Generate answer
if query:
    try:
        answer, sources = generate_answer(query)
        st.header("Answer:")
        st.write(answer)

        if sources:
            st.subheader("Sources:")
            for source in sources.split("\n"):
                st.write(source)
    except Exception as e:
        st.error(f"An error occurred: {e}")

