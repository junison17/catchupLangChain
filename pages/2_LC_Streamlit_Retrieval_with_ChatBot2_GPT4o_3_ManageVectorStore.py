# manage_topics.py
import streamlit as st
import os
import pickle

TOPICS_DIR = "faiss_indices"

def load_metadata(topic):
    meta_file = os.path.join(TOPICS_DIR, f"{topic}_meta.pkl")
    if os.path.exists(meta_file):
        try:
            with open(meta_file, "rb") as f:
                data = pickle.load(f)
                if len(data) == 5:
                    metadata = data[-1]
                    creation_date = metadata.get("creation_date", "N/A")
                    content_size = metadata.get("content_size", "N/A")
                    file_names = metadata.get("file_names", [])
                    file_names_str = ", ".join(file_names)
                    return f"\ncreation_date : {creation_date}, content_size : {content_size}\n\nfile_names : {file_names_str}"
                elif len(data) == 4:
                    return "No additional metadata"
        except (EOFError, pickle.UnpicklingError):
            return "Metadata file is corrupted"
    return "No metadata available"

def delete_topic(topic):
    index_file = os.path.join(TOPICS_DIR, f"{topic}.index")
    meta_file = os.path.join(TOPICS_DIR, f"{topic}_meta.pkl")
    if os.path.exists(index_file):
        os.remove(index_file)
    if os.path.exists(meta_file):
        os.remove(meta_file)

def delete_all_topics():
    for file in os.listdir(TOPICS_DIR):
        file_path = os.path.join(TOPICS_DIR, file)
        if os.path.exists(file_path):
            os.remove(file_path)

def main():
    st.title('Manage FAISS Topics')

    if not os.path.exists(TOPICS_DIR):
        os.makedirs(TOPICS_DIR)
    
    topics = [f.split('.')[0] for f in os.listdir(TOPICS_DIR) if f.endswith('.index')]

    st.header("Available Topics")
    if not topics:
        st.write("No topics available.")
    else:
        selected_topics = []
        for topic in topics:
            metadata = load_metadata(topic)
            checkbox = st.checkbox(f"Topic : {topic}\n{metadata}", key=f"checkbox_{topic}")
            if checkbox:
                selected_topics.append(topic)

        if st.button("Delete Selected Topics"):
            for topic in selected_topics:
                delete_topic(topic)
            st.experimental_rerun()

    if st.button("Delete All Topics"):
        delete_all_topics()
        st.experimental_rerun()

if __name__ == "__main__":
    main()
