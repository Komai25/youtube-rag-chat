#!/usr/bin/env python3
"""
Simple launcher script for the YouTube RAG Chat application.
This script checks dependencies and starts the Streamlit app.
"""

import subprocess
import sys
import os

def check_ollama():
    """Check if Ollama is running and has the required model"""
    try:
        result = subprocess.run(['ollama', 'list'], capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ Ollama is running")
            
            # Check if the required model is available
            if 'hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF' in result.stdout:
                print("✅ Required Ollama model is available")
            else:
                print("⚠️  Required model not found. Please run:")
                print("   ollama pull hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF")
            
            return True
        else:
            print("❌ Ollama is not running. Please start it with 'ollama serve'")
            return False
    except FileNotFoundError:
        print("❌ Ollama is not installed. Please install it from https://ollama.ai")
        print("   Then run: ollama pull hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF")
        return False

def check_ffmpeg():
    """Check if FFmpeg is installed"""
    try:
        result = subprocess.run(['ffmpeg', '-version'], capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ FFmpeg is installed")
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

def main():
    print("🎥 YouTube RAG Chat - Startup Check")
    print("=" * 40)
    
    # Check dependencies
    checks_passed = 0
    total_checks = 2
    
    if check_ollama():
        checks_passed += 1
    
    if check_ffmpeg():
        checks_passed += 1
    
    print("=" * 40)
    
    if checks_passed == total_checks:
        print("🚀 All checks passed! Starting the application...")
        print("📱 The app will open in your browser automatically")
        print("🛑 Press Ctrl+C to stop the application")
        print("-" * 40)
        
        # Start Streamlit app
        try:
            subprocess.run([sys.executable, '-m', 'streamlit', 'run', 'app.py'])
        except KeyboardInterrupt:
            print("\n👋 Application stopped by user")
    else:
        print(f"❌ {total_checks - checks_passed} check(s) failed. Please fix the issues above.")
        sys.exit(1)

if __name__ == "__main__":
    main()
