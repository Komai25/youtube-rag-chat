# YouTube RAG Chat 🎥💬

An interactive Streamlit application that transforms YouTube videos into conversational AI experiences. Simply paste a YouTube URL, and start having intelligent conversations about the video content using Retrieval Augmented Generation (RAG) technology.

# Demo

[![Demo](./demo.gif)](./demo.mp4)

## 🌟 What Makes This Special

This application combines several cutting-edge AI technologies to create a seamless experience:
- **YouTube Integration**: Direct video processing from URLs
- **Whisper Transcription**: High-quality audio-to-text conversion
- **Semantic Chunking**: Intelligent text segmentation for better context understanding
- **Vector Search**: Fast and relevant content retrieval using FAISS
- **Local LLM Integration**: Privacy-focused AI conversations using Ollama
- **Persistent Storage**: Your processed videos and conversations are saved across sessions

## Features

- 🎥 **YouTube Video Processing**: Download and process any YouTube video
- 🎤 **Audio Transcription**: Automatic transcription using OpenAI Whisper
- 🧠 **Semantic Chunking**: Intelligent text segmentation for better context retrieval
- 🔍 **Vector Search**: FAISS-based similarity search for relevant content
- 💬 **Interactive Chat**: Ask questions about video content with conversation history
- 📁 **Multi-Video Support**: Process and switch between multiple videos
- 🔄 **Persistent Storage**: Videos and chat history persist across app restarts
- 📊 **Smart Video Library**: See video titles, durations, upload dates, and chat counts
- 🗑️ **Easy Management**: Delete processed videos you no longer need
- ⚡ **Smart Caching**: Already processed videos load instantly

## 🚀 Quick Start

### Option 1: Automated Setup (Recommended)
```bash
python setup.py
```
This will check and install all dependencies automatically.

### Option 2: Manual Setup
1. **Install dependencies**: `pip install -r requirements.txt`
2. **Install FFmpeg** (required for audio processing)
3. **Install and start Ollama**: `ollama serve` then `ollama pull hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF`

### Running the Application
```bash
# Option 1: Direct run
streamlit run app.py

# Option 2: With dependency checks
python run_app.py
```

Then open your browser to the URL shown (usually `http://localhost:8501`) and start chatting with YouTube videos!

## Prerequisites

Before running the application, make sure you have:

1. **Python 3.8+** installed
2. **FFmpeg** installed (required for audio processing)
   - Windows: Download from [FFmpeg website](https://ffmpeg.org/download.html)
   - Mac: `brew install ffmpeg`
   - Ubuntu: `sudo apt install ffmpeg`
3. **Ollama** installed and running with the required model:
   ```bash
   # Install Ollama from https://ollama.ai
   # Then pull the required model:
   ollama pull hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF
   ```

## Installation

1. **Clone or download the project files**

2. **Create a virtual environment**:
   ```bash
   python -m venv venv
   
   # Activate the virtual environment
   # Windows:
   venv\Scripts\activate
   # Mac/Linux:
   source venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements_app.txt
   ```

## Usage

1. **Start the Ollama service** (if not already running):
   ```bash
   ollama serve
   ```

2. **Run the Streamlit application**:
   ```bash
   streamlit run app.py
   ```

3. **Open your browser** and navigate to the URL shown in the terminal (usually `http://localhost:8501`)

4. **Use the application**:
   - Enter a YouTube URL in the sidebar
   - Click "Process Video" and wait for processing to complete
   - Start chatting with the video content!

## How It Works

1. **Video Download**: Uses `yt-dlp` to download audio from YouTube videos
2. **Transcription**: OpenAI Whisper converts audio to text
3. **Semantic Chunking**: Text is intelligently segmented using semantic similarity
4. **Embedding Creation**: Text chunks are converted to vector embeddings using HuggingFace models
5. **Vector Storage**: FAISS creates a searchable index of embeddings
6. **Question Answering**: User questions retrieve relevant context and generate answers using Ollama

## Configuration

### Model Settings

You can modify the models used in the `YouTubeRAGProcessor` class:

- **Whisper Model**: Change `medium` to `base`, `small`, or `large` based on your needs
- **Embedding Model**: Currently uses `BAAI/bge-base-en-v1.5`
- **LLM Model**: Configure in Ollama settings

### Chunking Parameters

Adjust semantic chunking in the `create_vector_store` method:

```python
chunker = SemanticChunker(
    embeddings=self.embedder,
    buffer_size=1,
    breakpoint_threshold_type="percentile",
    breakpoint_threshold_amount=95.0,  # Adjust this value
    min_chunk_size=50  # Minimum chunk size
)
```

## Troubleshooting

### Common Issues

1. **FFmpeg not found**:
   - Make sure FFmpeg is installed and in your PATH
   - Restart your terminal after installation

2. **Ollama connection error**:
   - Ensure Ollama is running: `ollama serve`
   - Check if the model is pulled: `ollama list`

3. **Memory issues**:
   - Use a smaller Whisper model (`base` instead of `medium`)
   - Reduce batch size in processing

4. **YouTube download fails**:
   - Check if the video is accessible
   - Some videos may be region-restricted or private

### Performance Tips

- **First run will be slower** due to model downloads
- **Use smaller Whisper models** for faster processing
- **GPU acceleration** will significantly improve performance if available

## File Structure

```
youtube-rag-chat/
├── app.py                           # Main Streamlit application
├── setup.py                        # Automated setup script
├── run_app.py                       # Launcher script with dependency checks
├── requirements.txt                 # Python dependencies
├── README.md                       # This documentation
├── .gitignore                      # Git ignore file
├── rag-implementation-reference.ipynb  # Original notebook implementation
└── processed_videos/               # Persistent storage (auto-created)
    ├── [video_id_1]/
    │   ├── metadata.json           # Video info and settings
    │   ├── chat_history.json       # Saved conversations
    │   └── faiss_index/            # Vector embeddings
    │       ├── index.faiss
    │       └── index.pkl
    └── [video_id_2]/
        └── ...
```

### Storage Details

- **`processed_videos/`**: Main storage directory for all processed videos
- **`metadata.json`**: Contains video title, duration, uploader, transcription, and processing date
- **`chat_history.json`**: Stores all conversation messages with timestamps
- **`faiss_index/`**: Vector database files for semantic search
- Videos are organized by their YouTube video ID for easy management

### Reference Implementation

The `rag-implementation-reference.ipynb` notebook contains the original implementation that this Streamlit app is based on. It's useful for:
- Understanding the underlying RAG pipeline
- Experimenting with different models or parameters
- Learning how the components work together

## Dependencies

- **Streamlit**: Web app framework
- **LangChain**: RAG framework and LLM integration
- **OpenAI Whisper**: Audio transcription
- **FAISS**: Vector similarity search
- **HuggingFace Transformers**: Text embeddings
- **yt-dlp**: YouTube video download
- **Ollama**: Local LLM inference

## License

This project is provided as-is for educational and research purposes.

## Contributing

Feel free to submit issues and enhancement requests!
