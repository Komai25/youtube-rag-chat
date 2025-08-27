import streamlit as st
import os
import tempfile
import whisper
import yt_dlp
from langchain_core.output_parsers import StrOutputParser
from langchain.llms import Ollama
from langchain.prompts import ChatPromptTemplate
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from langchain.vectorstores import FAISS
from langchain.docstore.document import Document
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_experimental.text_splitter import SemanticChunker
from langchain_core.documents import Document
import hashlib
import pickle
import shutil
import json
from pathlib import Path
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="YouTube RAG Chat",
    page_icon="🎥",
    layout="wide"
)

# Constants
STORAGE_DIR = Path("processed_videos")
STORAGE_DIR.mkdir(exist_ok=True)

def load_existing_videos():
    """Load previously processed videos from disk"""
    processed_videos = {}
    
    if STORAGE_DIR.exists():
        for video_folder in STORAGE_DIR.iterdir():
            if video_folder.is_dir():
                metadata_file = video_folder / "metadata.json"
                chat_file = video_folder / "chat_history.json"
                
                if metadata_file.exists():
                    try:
                        with open(metadata_file, 'r', encoding='utf-8') as f:
                            metadata = json.load(f)
                        
                        video_id = video_folder.name
                        processed_videos[video_id] = {
                            'title': metadata.get('title', 'Unknown Title'),
                            'url': metadata.get('url', ''),
                            'transcription': metadata.get('transcription', ''),
                            'db_path': str(video_folder / "faiss_index"),
                            'processed_date': metadata.get('processed_date', ''),
                            'duration': metadata.get('duration', ''),
                            'db': None,  # Will be loaded when needed
                            'chain': None  # Will be created when needed
                        }
                        
                        # Load chat history if exists
                        if chat_file.exists():
                            with open(chat_file, 'r', encoding='utf-8') as f:
                                chat_history = json.load(f)
                                if 'chat_history' not in st.session_state:
                                    st.session_state.chat_history = {}
                                st.session_state.chat_history[video_id] = chat_history
                        
                    except Exception as e:
                        st.warning(f"Could not load video {video_id}: {str(e)}")
                        continue
    
    return processed_videos

def save_chat_history(video_id, chat_history):
    """Save chat history to disk"""
    video_folder = STORAGE_DIR / video_id
    video_folder.mkdir(exist_ok=True)
    chat_file = video_folder / "chat_history.json"
    
    try:
        with open(chat_file, 'w', encoding='utf-8') as f:
            json.dump(chat_history, f, ensure_ascii=False, indent=2)
    except Exception as e:
        st.error(f"Failed to save chat history: {str(e)}")

# Initialize session state
if 'processed_videos' not in st.session_state:
    st.session_state.processed_videos = load_existing_videos()
if 'current_video_id' not in st.session_state:
    st.session_state.current_video_id = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = {}

class YouTubeRAGProcessor:
    def __init__(self):
        self.embedder = None
        self.whisper_model = None
        self.llm = None
        self.setup_models()
    
    def setup_models(self):
        """Initialize all required models"""
        try:
            # Initialize embedder
            with st.spinner("Loading embedding model..."):
                self.embedder = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en-v1.5")
            
            # Initialize Whisper model
            with st.spinner("Loading Whisper model..."):
                self.whisper_model = whisper.load_model("medium")
            
            # Initialize LLM
            with st.spinner("Connecting to Ollama..."):
                self.llm = Ollama(model="hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF")
            
            st.success("All models loaded successfully!")
        except Exception as e:
            st.error(f"Error loading models: {str(e)}")
            st.stop()
    
    def get_video_id(self, youtube_url):
        """Extract video ID from YouTube URL"""
        import re
        video_id_match = re.search(r'(?:youtube\.com/watch\?v=|youtu\.be/)([^&\n?#]+)', youtube_url)
        return video_id_match.group(1) if video_id_match else None
    
    def get_video_metadata(self, youtube_url):
        """Extract video metadata including title and duration"""
        ydl_opts = {
            'quiet': True,
            'no_warnings': True,
        }
        
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(youtube_url, download=False)
                return {
                    'title': info.get('title', 'Unknown Title'),
                    'duration': info.get('duration', 0),
                    'duration_string': info.get('duration_string', 'Unknown'),
                    'uploader': info.get('uploader', 'Unknown'),
                    'upload_date': info.get('upload_date', 'Unknown'),
                    'view_count': info.get('view_count', 0)
                }
        except Exception as e:
            st.warning(f"Could not extract video metadata: {str(e)}")
            return {
                'title': 'Unknown Title',
                'duration': 0,
                'duration_string': 'Unknown',
                'uploader': 'Unknown',
                'upload_date': 'Unknown',
                'view_count': 0
            }
    
    def download_youtube_audio(self, youtube_url, video_id):
        """Download audio from YouTube video"""
        output_path = f"temp_audio_{video_id}"
        
        ydl_opts = {
            'format': 'bestaudio/best',
            'outtmpl': f'{output_path}.%(ext)s',
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'mp3',
                'preferredquality': '192',
            }],
        }
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([youtube_url])
        
        return f"{output_path}.mp3"
    
    def transcribe_audio(self, audio_path):
        """Transcribe audio using Whisper"""
        result = self.whisper_model.transcribe(audio_path)
        return result["text"]
    
    def create_vector_store(self, text, video_id):
        """Create FAISS vector store from text"""
        # Create semantic chunks
        chunker = SemanticChunker(
            embeddings=self.embedder,
            buffer_size=1,
            breakpoint_threshold_type="percentile",
            breakpoint_threshold_amount=95.0,
            min_chunk_size=50
        )
        
        semantic_chunks = chunker.create_documents([text])
        
        # Create FAISS vector store
        db = FAISS.from_documents(semantic_chunks, embedding=self.embedder)
        
        # Save vector store in persistent storage
        video_folder = STORAGE_DIR / video_id
        video_folder.mkdir(exist_ok=True)
        db_path = video_folder / "faiss_index"
        db.save_local(str(db_path))
        
        return db, str(db_path)
    
    def load_vector_store(self, db_path):
        """Load existing FAISS vector store"""
        try:
            db = FAISS.load_local(db_path, self.embedder, allow_dangerous_deserialization=True)
            return db
        except Exception as e:
            st.error(f"Failed to load vector store: {str(e)}")
            return None
    
    def save_video_metadata(self, video_id, metadata, transcription):
        """Save video metadata to disk"""
        video_folder = STORAGE_DIR / video_id
        video_folder.mkdir(exist_ok=True)
        metadata_file = video_folder / "metadata.json"
        
        video_data = {
            'title': metadata['title'],
            'url': metadata.get('url', ''),
            'transcription': transcription,
            'duration': metadata['duration'],
            'duration_string': metadata['duration_string'],
            'uploader': metadata['uploader'],
            'upload_date': metadata['upload_date'],
            'view_count': metadata['view_count'],
            'processed_date': datetime.now().isoformat()
        }
        
        try:
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(video_data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            st.error(f"Failed to save video metadata: {str(e)}")
    
    def setup_rag_chain(self, db):
        """Setup RAG chain for question answering"""
        # Create template
        template = """
        You are a helpful assistant. Use the provided context to answer the question as accurately as possible.
        If the answer cannot be found in the context, respond with "I don't know".
        
        Context:
        {context}
        
        Question:
        {question}
        
        Answer:"""
        
        prompt = ChatPromptTemplate.from_template(template)
        parser = StrOutputParser()
        
        def fetch_context(question):
            return "\n\n".join([
                doc.page_content for doc in db.similarity_search(question, k=4)
            ])
        
        setup = RunnableParallel({
            "question": RunnablePassthrough(),
            "context": fetch_context
        })
        
        chain = setup | prompt | self.llm | parser
        return chain
    
    def process_video(self, youtube_url):
        """Complete video processing pipeline"""
        video_id = self.get_video_id(youtube_url)
        if not video_id:
            raise ValueError("Invalid YouTube URL")
        
        # Check if already processed
        if video_id in st.session_state.processed_videos:
            st.info(f"Video already processed! Loading from cache...")
            video_data = st.session_state.processed_videos[video_id]
            
            # Load vector store and create chain if not already loaded
            if video_data.get('db') is None and video_data.get('db_path'):
                with st.spinner("Loading vector store..."):
                    video_data['db'] = self.load_vector_store(video_data['db_path'])
                    if video_data['db']:
                        video_data['chain'] = self.setup_rag_chain(video_data['db'])
            
            return video_id, video_data
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            # Step 0: Get video metadata
            status_text.text("Extracting video metadata...")
            progress_bar.progress(5)
            metadata = self.get_video_metadata(youtube_url)
            metadata['url'] = youtube_url
            
            # Step 1: Download audio
            status_text.text("Downloading video audio...")
            progress_bar.progress(20)
            audio_path = self.download_youtube_audio(youtube_url, video_id)
            
            # Step 2: Transcribe
            status_text.text("Transcribing audio...")
            progress_bar.progress(50)
            transcription = self.transcribe_audio(audio_path)
            
            # Step 3: Create vector store
            status_text.text("Creating vector embeddings...")
            progress_bar.progress(75)
            db, db_path = self.create_vector_store(transcription, video_id)
            
            # Step 4: Setup RAG chain
            status_text.text("Setting up RAG chain...")
            progress_bar.progress(90)
            chain = self.setup_rag_chain(db)
            
            # Step 5: Save metadata and transcription
            status_text.text("Saving to persistent storage...")
            progress_bar.progress(95)
            self.save_video_metadata(video_id, metadata, transcription)
            
            # Store processed video data
            video_data = {
                'title': metadata['title'],
                'url': youtube_url,
                'transcription': transcription,
                'duration': metadata['duration'],
                'duration_string': metadata['duration_string'],
                'uploader': metadata['uploader'],
                'processed_date': datetime.now().isoformat(),
                'db': db,
                'db_path': db_path,
                'chain': chain
            }
            
            st.session_state.processed_videos[video_id] = video_data
            
            # Initialize chat history for this video
            if video_id not in st.session_state.chat_history:
                st.session_state.chat_history[video_id] = []
            
            # Cleanup temporary files
            try:
                os.remove(audio_path)
            except:
                pass
            
            progress_bar.progress(100)
            status_text.text("Video processed and saved successfully!")
            
            return video_id, video_data
            
        except Exception as e:
            status_text.text(f"Error: {str(e)}")
            raise e
        finally:
            progress_bar.empty()
            status_text.empty()

def main():
    st.title("🎥 YouTube RAG Chat")
    st.markdown("Upload a YouTube video and chat with its content using RAG!")
    
    # Initialize processor
    if 'processor' not in st.session_state:
        st.session_state.processor = YouTubeRAGProcessor()
    
    # Sidebar for video management
    with st.sidebar:
        st.header("📹 Video Management")
        
        # YouTube URL input
        youtube_url = st.text_input(
            "Enter YouTube URL:",
            placeholder="https://youtu.be/example"
        )
        
        # Process button
        if st.button("Process Video", type="primary"):
            if youtube_url:
                try:
                    video_id, video_data = st.session_state.processor.process_video(youtube_url)
                    st.session_state.current_video_id = video_id
                    if video_id not in st.session_state.chat_history:
                        st.session_state.chat_history[video_id] = []
                    st.success(f"Video processed! ID: {video_id}")
                except Exception as e:
                    st.error(f"Error processing video: {str(e)}")
            else:
                st.warning("Please enter a YouTube URL")
        
        # Video selector
        if st.session_state.processed_videos:
            st.subheader("📚 Processed Videos")
            
            # Create a more informative display
            video_options = {}
            for vid, data in st.session_state.processed_videos.items():
                title = data.get('title', 'Unknown Title')
                duration = data.get('duration_string', 'Unknown duration')
                uploader = data.get('uploader', 'Unknown uploader')
                processed_date = data.get('processed_date', '')
                
                if processed_date:
                    try:
                        date_obj = datetime.fromisoformat(processed_date)
                        processed_date = date_obj.strftime('%m/%d/%Y')
                    except:
                        processed_date = 'Unknown date'
                
                display_name = f"🎬 {title[:40]}{'...' if len(title) > 40 else ''}\n📺 {uploader} • {duration}\n📅 Processed: {processed_date}"
                video_options[vid] = display_name
            
            selected_video = st.selectbox(
                "Select a video to chat with:",
                options=list(video_options.keys()),
                format_func=lambda x: video_options[x],
                index=0 if st.session_state.current_video_id is None else (
                    list(video_options.keys()).index(st.session_state.current_video_id) 
                    if st.session_state.current_video_id in video_options 
                    else 0
                )
            )
            
            if selected_video != st.session_state.current_video_id:
                st.session_state.current_video_id = selected_video
                if selected_video not in st.session_state.chat_history:
                    st.session_state.chat_history[selected_video] = []
            
            # Show video statistics
            if selected_video and selected_video in st.session_state.processed_videos:
                video_data = st.session_state.processed_videos[selected_video]
                st.markdown("---")
                st.markdown("**Video Info:**")
                st.caption(f"**Title:** {video_data.get('title', 'N/A')}")
                st.caption(f"**Duration:** {video_data.get('duration_string', 'N/A')}")
                st.caption(f"**Uploader:** {video_data.get('uploader', 'N/A')}")
                
                # Show chat history count
                chat_count = len(st.session_state.chat_history.get(selected_video, []))
                st.caption(f"**Chat Messages:** {chat_count}")
                
                # Delete video option
                if st.button("🗑️ Delete Video", type="secondary", key="delete_video"):
                    if st.session_state.current_video_id == selected_video:
                        st.session_state.current_video_id = None
                    
                    # Delete from session state
                    del st.session_state.processed_videos[selected_video]
                    if selected_video in st.session_state.chat_history:
                        del st.session_state.chat_history[selected_video]
                    
                    # Delete from disk
                    video_folder = STORAGE_DIR / selected_video
                    if video_folder.exists():
                        try:
                            shutil.rmtree(video_folder)
                            st.success("Video deleted successfully!")
                        except Exception as e:
                            st.error(f"Error deleting video files: {str(e)}")
                    st.rerun()
    
    # Main chat interface
    if st.session_state.current_video_id:
        video_id = st.session_state.current_video_id
        video_data = st.session_state.processed_videos[video_id]
        
        # Ensure vector store and chain are loaded
        if video_data.get('db') is None or video_data.get('chain') is None:
            with st.spinner("Loading video data..."):
                if video_data.get('db_path') and st.session_state.processor:
                    video_data['db'] = st.session_state.processor.load_vector_store(video_data['db_path'])
                    if video_data['db']:
                        video_data['chain'] = st.session_state.processor.setup_rag_chain(video_data['db'])
        
        video_title = video_data.get('title', video_id)
        st.header(f"💬 Chat with: {video_title}")
        
        # Video info bar
        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            st.caption(f"📺 {video_data.get('uploader', 'Unknown')}")
        with col2:
            st.caption(f"⏱️ {video_data.get('duration_string', 'Unknown')}")
        with col3:
            chat_count = len(st.session_state.chat_history.get(video_id, []))
            st.caption(f"💬 {chat_count} messages")
        
        # Display transcription in expander
        with st.expander("📄 View Full Transcription"):
            st.text_area(
                "Video Transcription:",
                value=video_data.get('transcription', 'No transcription available'),
                height=200,
                disabled=True
            )
        
        # Chat interface
        chat_container = st.container()
        
        # Display chat history
        with chat_container:
            for message in st.session_state.chat_history[video_id]:
                if message['role'] == 'user':
                    st.chat_message("user").write(message['content'])
                else:
                    st.chat_message("assistant").write(message['content'])
        
        # Chat input
        if prompt := st.chat_input("Ask a question about the video..."):
            # Ensure we have a working chain
            if not video_data.get('chain'):
                st.error("Video processing not complete. Please wait or reprocess the video.")
                st.stop()
            
            # Add user message to history
            st.session_state.chat_history[video_id].append({
                'role': 'user',
                'content': prompt,
                'timestamp': datetime.now().isoformat()
            })
            
            # Display user message
            st.chat_message("user").write(prompt)
            
            # Get response from RAG chain
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        response = video_data['chain'].invoke(prompt)
                        st.write(response)
                        
                        # Add assistant response to history
                        st.session_state.chat_history[video_id].append({
                            'role': 'assistant',
                            'content': response,
                            'timestamp': datetime.now().isoformat()
                        })
                        
                        # Auto-save chat history
                        save_chat_history(video_id, st.session_state.chat_history[video_id])
                        
                    except Exception as e:
                        error_msg = f"Error generating response: {str(e)}"
                        st.error(error_msg)
                        st.session_state.chat_history[video_id].append({
                            'role': 'assistant',
                            'content': error_msg,
                            'timestamp': datetime.now().isoformat()
                        })
                        
                        # Save error to history too
                        save_chat_history(video_id, st.session_state.chat_history[video_id])
    
    else:
        st.info("👆 Please process a YouTube video to start chatting!")
        
        # Display sample and instructions
        st.markdown("""
        ### 🚀 How it works:
        1. **Enter a YouTube URL** in the sidebar
        2. **Click 'Process Video'** to download, transcribe, and create embeddings
        3. **Ask questions** about the video content  
        4. **Switch between videos** using the dropdown in the sidebar
        5. **Your conversations are automatically saved** and restored when you return!
        
        ### 💡 Features:
        - **🔄 Persistent Storage**: Videos stay processed between app sessions
        - **💬 Chat History**: All conversations are saved and restored
        - **📋 Video Library**: Manage multiple processed videos
        - **🗑️ Easy Cleanup**: Delete videos you no longer need
        - **📊 Smart Info**: See video titles, duration, and chat counts
        
        ### 🤔 Example questions you can ask:
        - "What is the main topic of this video?"
        - "Can you summarize the key points?"  
        - "What does the speaker say about [specific topic]?"
        - "Who is mentioned in this video?"
        - "What are the important timestamps?"
        
        ### 📁 Storage:
        Processed videos and chat histories are stored in the `processed_videos/` folder.
        Each video gets its own folder with transcription, embeddings, and chat history.
        """)

if __name__ == "__main__":
    main()
