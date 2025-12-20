#!/usr/bin/env python3
"""
Test script to verify UltraRAG installation and configuration.
Run this after setup to check if everything is working.
"""
import sys
from pathlib import Path


def check_python_version():
    """Check Python version."""
    print("1️⃣  Checking Python version...")
    version = sys.version_info
    if version.major >= 3 and version.minor >= 10:
        print(f"   ✅ Python {version.major}.{version.minor}.{version.micro}")
        return True
    else:
        print(f"   ❌ Python {version.major}.{version.minor} (need 3.10+)")
        return False


def check_dependencies():
    """Check if required packages are installed."""
    print("\n2️⃣  Checking dependencies...")
    
    required = [
        "llama_index",
        "lancedb",
        "dotenv",
        "pandas",
        "tqdm",
        "frontmatter",
        "pydantic"
    ]
    
    missing = []
    for package in required:
        try:
            __import__(package)
            print(f"   ✅ {package}")
        except ImportError:
            print(f"   ❌ {package}")
            missing.append(package)
    
    if missing:
        print(f"\n   Missing packages: {', '.join(missing)}")
        print("   Run: pip install -r requirements.txt")
        return False
    
    return True


def check_env_file():
    """Check if .env file exists and has required settings."""
    print("\n3️⃣  Checking .env configuration...")
    
    env_path = Path(".env")
    if not env_path.exists():
        print("   ❌ .env file not found")
        print("   Run: cp .env.example .env")
        return False
    
    print("   ✅ .env file exists")
    
    # Check for required settings
    with open(env_path, 'r') as f:
        content = f.read()
    
    required_vars = {
        "OBSIDIAN_VAULT_PATH": "Vault path",
        "GOOGLE_API_KEY": "Google API (for Gemini)",
    }
    
    missing = []
    for var, desc in required_vars.items():
        if var in content:
            # Check if it has a value (not just the template)
            for line in content.split('\n'):
                if line.startswith(var):
                    value = line.split('=', 1)[1].strip()
                    if value and not value.startswith('your_') and not value.startswith('/path/'):
                        print(f"   ✅ {desc} configured")
                    else:
                        print(f"   ⚠️  {desc} needs to be set")
                        missing.append(var)
                    break
        else:
            print(f"   ❌ {desc} missing")
            missing.append(var)
    
    if missing:
        print(f"\n   Please configure: {', '.join(missing)}")
        return False
    
    return True


def check_vault_path():
    """Check if vault path exists."""
    print("\n4️⃣  Checking Obsidian vault...")
    
    try:
        from dotenv import load_dotenv
        import os
        
        load_dotenv()
        vault_path = os.getenv("OBSIDIAN_VAULT_PATH", "")
        
        if not vault_path:
            print("   ⚠️  OBSIDIAN_VAULT_PATH not set in .env")
            return False
        
        path = Path(vault_path)
        if not path.exists():
            print(f"   ❌ Vault not found: {vault_path}")
            return False
        
        # Count markdown files
        md_files = list(path.rglob("*.md"))
        print(f"   ✅ Vault found: {vault_path}")
        print(f"   📝 Found {len(md_files)} markdown files")
        return True
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False


def check_api_connectivity():
    """Test API connectivity (optional)."""
    print("\n5️⃣  Testing API connectivity (optional)...")
    
    try:
        from dotenv import load_dotenv
        import os
        
        load_dotenv()
        
        # Test Google API
        google_key = os.getenv("GOOGLE_API_KEY", "")
        if google_key and not google_key.startswith("your_"):
            try:
                import google.generativeai as genai
                genai.configure(api_key=google_key)
                print("   ✅ Google API key valid")
            except Exception as e:
                print(f"   ⚠️  Google API: {str(e)[:50]}...")
        else:
            print("   ⚠️  Google API key not configured")
        
        # Test Voyage API (optional)
        voyage_key = os.getenv("VOYAGE_API_KEY", "")
        if voyage_key and not voyage_key.startswith("your_"):
            print("   ✅ Voyage API key configured")
        else:
            print("   ℹ️  Voyage API key not configured (optional)")
        
        return True
        
    except Exception as e:
        print(f"   ⚠️  Could not test APIs: {e}")
        return True  # Non-critical


def main():
    """Run all checks."""
    print("=" * 60)
    print("🔍 UltraRAG Installation Check")
    print("=" * 60)
    
    checks = [
        check_python_version(),
        check_dependencies(),
        check_env_file(),
        check_vault_path(),
        check_api_connectivity()
    ]
    
    print("\n" + "=" * 60)
    
    if all(checks[:4]):  # First 4 are critical
        print("✅ All critical checks passed!")
        print("\n🚀 You're ready to run:")
        print("   python main.py          (CLI interface)")
        print("   streamlit run app.py    (Web interface)")
    else:
        print("❌ Some checks failed. Please fix the issues above.")
        print("\n📖 See QUICKSTART.md for setup instructions")
    
    print("=" * 60)


if __name__ == "__main__":
    main()
