"""
Setup script for LangSmith integration with ITSM Classification System
"""

import subprocess
import sys
import os

def install_langsmith():
    """Install LangSmith package"""
    print("📦 Installing LangSmith...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "langsmith"])
        print("✅ LangSmith installed successfully")
        return True
    except subprocess.CalledProcessError:
        print("❌ Failed to install LangSmith")
        return False

def setup_environment():
    """Setup environment variables for LangSmith"""
    print("\n🔧 Setting up LangSmith environment...")
    
    # Check current environment
    env_vars = {
        "LANGCHAIN_TRACING_V2": "true",
        "LANGCHAIN_ENDPOINT": "https://api.smith.langchain.com",
        "LANGCHAIN_PROJECT": "itsm-classification-demo"
    }
    
    for var, value in env_vars.items():
        os.environ[var] = value
        print(f"✅ Set {var}={value}")
    
    # Check for API key
    api_key = os.environ.get("LANGCHAIN_API_KEY")
    if not api_key:
        print("\n⚠️  LANGCHAIN_API_KEY not found!")
        print("To enable LangSmith tracing:")
        print("1. Go to https://smith.langchain.com/")
        print("2. Create an account and get your API key")
        print("3. Set the environment variable:")
        print("   export LANGCHAIN_API_KEY=your-api-key")
        print("\nFor now, continuing without LangSmith tracing...")
    else:
        print("✅ LANGCHAIN_API_KEY found")
    
    return True

def main():
    print("🚀 LangSmith Setup for ITSM Classification")
    print("=" * 50)
    
    # Install LangSmith
    if not install_langsmith():
        return
    
    # Setup environment
    if not setup_environment():
        return
    
    print("\n🎉 Setup completed!")
    print("\nNext steps:")
    print("1. Run: python demo_comprehensive.py (for detailed JSON output)")
    print("2. Run: python demo_langsmith.py (for LangSmith tracing)")
    print("\nNote: For full LangSmith tracing, set your LANGCHAIN_API_KEY")

if __name__ == "__main__":
    main()
