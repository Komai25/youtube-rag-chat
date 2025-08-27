#!/usr/bin/env python3
"""
Setup script for YouTube RAG Chat application
"""

import subprocess
import sys
import os
from pathlib import Path

def install_requirements():
    """Install Python requirements"""
    print("📦 Installing Python dependencies...")
    try:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'])
        print("✅ Python dependencies installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install Python dependencies: {e}")
        return False

def check_ffmpeg():
    """Check if FFmpeg is installed"""
    try:
        result = subprocess.run(['ffmpeg', '-version'], capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ FFmpeg is already installed")
            return True
        else:
            print("❌ FFmpeg check failed")
            return False
    except FileNotFoundError:
        print("❌ FFmpeg is not installed. Please install it:")
        print("   - Windows: Download from https://ffmpeg.org/download.html")
        print("   - Mac: brew install ffmpeg")
        print("   - Ubuntu: sudo apt install ffmpeg")
        return False

def check_ollama():
    """Check Ollama installation"""
    try:
        result = subprocess.run(['ollama', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ Ollama is installed")
            print("🔄 To complete setup, please run:")
            print("   1. ollama serve  (in a separate terminal)")
            print("   2. ollama pull hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF")
            return True
        else:
            print("❌ Ollama check failed")
            return False
    except FileNotFoundError:
        print("❌ Ollama is not installed. Please install it from:")
        print("   https://ollama.ai")
        print("   Then run the model: ollama pull hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF")
        return False

def create_directories():
    """Create necessary directories"""
    directories = ['processed_videos']
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
    print("📁 Created necessary directories")

def main():
    print("🎥 YouTube RAG Chat - Setup Script")
    print("=" * 50)
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ is required. You have:", sys.version)
        sys.exit(1)
    else:
        print(f"✅ Python version: {sys.version.split()[0]}")
    
    setup_steps = [
        ("Installing Python dependencies", install_requirements),
        ("Checking FFmpeg installation", check_ffmpeg),
        ("Checking Ollama installation", check_ollama),
        ("Creating directories", create_directories),
    ]
    
    failed_steps = []
    
    for step_name, step_function in setup_steps:
        print(f"\n🔄 {step_name}...")
        try:
            if not step_function():
                failed_steps.append(step_name)
        except Exception as e:
            print(f"❌ Error in {step_name}: {e}")
            failed_steps.append(step_name)
    
    print("\n" + "=" * 50)
    print("📋 Setup Summary:")
    
    if not failed_steps:
        print("🎉 All setup steps completed successfully!")
        print("\n🚀 Next steps:")
        print("1. Make sure Ollama is running: ollama serve")
        print("2. Start the application: streamlit run app.py")
        print("   Or use the launcher: python run_app.py")
    else:
        print(f"⚠️  Setup completed with {len(failed_steps)} issue(s):")
        for step in failed_steps:
            print(f"   - {step}")
        print("\nPlease resolve the above issues before running the application.")

if __name__ == "__main__":
    main()
