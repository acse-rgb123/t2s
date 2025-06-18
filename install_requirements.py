#!/usr/bin/env python3
"""
Installation script for T2S project dependencies.
This script installs packages one by one to avoid dependency conflicts.
"""

import subprocess
import sys
import time

def run_pip_install(package, description=""):
    """Install a package using pip with error handling."""
    print(f"\n{'='*60}")
    print(f"Installing: {package}")
    if description:
        print(f"Description: {description}")
    print('='*60)
    
    try:
        result = subprocess.run([
            sys.executable, '-m', 'pip', 'install', package
        ], capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0:
            print(f"✅ Successfully installed: {package}")
            return True
        else:
            print(f"❌ Failed to install: {package}")
            print(f"Error: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print(f"⏰ Installation timeout for: {package}")
        return False
    except Exception as e:
        print(f"💥 Exception during installation of {package}: {e}")
        return False

def main():
    """Main installation function."""
    print("🚀 Starting T2S Project Dependencies Installation")
    print("This may take several minutes...")
    
    # Core packages (install first)
    core_packages = [
        ("python-dotenv", "Environment variable management"),
        ("pydantic", "Data validation"),
        ("fastapi", "Web framework"),
        ("uvicorn[standard]", "ASGI server"),
    ]
    
    # Google Cloud packages
    google_packages = [
        ("google-auth", "Google authentication"),
        ("google-cloud-core", "Google Cloud core libraries"),
        ("google-api-core", "Google API core"),
        ("google-cloud-bigquery", "BigQuery client"),
        ("google-generativeai", "Gemini API client"),
    ]
    
    # Database packages
    db_packages = [
        ("sqlalchemy", "Database ORM"),
        ("alembic", "Database migrations"),
        ("asyncpg", "PostgreSQL async driver"),
        ("db-dtypes", "BigQuery data types"),
    ]
    
    # Utility packages
    utility_packages = [
        ("redis", "Redis client"),
        ("httpx", "HTTP client"),
        ("python-multipart", "Multipart form support"),
    ]
    
    # AI/ML packages
    ai_packages = [
        ("openai", "OpenAI API client"),
        ("anthropic", "Anthropic API client"),
        ("langchain", "LangChain framework"),
        ("langchain-openai", "LangChain OpenAI integration"),
        ("langchain-anthropic", "LangChain Anthropic integration"),
    ]
    
    # Testing packages
    test_packages = [
        ("pytest", "Testing framework"),
        ("pytest-asyncio", "Async testing support"),
    ]
    
    # Install packages in groups
    package_groups = [
        ("Core Packages", core_packages),
        ("Google Cloud Packages", google_packages),
        ("Database Packages", db_packages),
        ("Utility Packages", utility_packages),
        ("AI/ML Packages", ai_packages),
        ("Testing Packages", test_packages),
    ]
    
    total_packages = sum(len(packages) for _, packages in package_groups)
    installed_count = 0
    failed_packages = []
    
    for group_name, packages in package_groups:
        print(f"\n🔧 Installing {group_name}")
        print("-" * 40)
        
        for package, description in packages:
            if run_pip_install(package, description):
                installed_count += 1
            else:
                failed_packages.append(package)
            
            # Small delay to avoid overwhelming the system
            time.sleep(1)
    
    # Summary
    print(f"\n{'='*60}")
    print("📊 INSTALLATION SUMMARY")
    print('='*60)
    print(f"Total packages: {total_packages}")
    print(f"Successfully installed: {installed_count}")
    print(f"Failed installations: {len(failed_packages)}")
    
    if failed_packages:
        print(f"\n❌ Failed packages:")
        for package in failed_packages:
            print(f"  - {package}")
        print(f"\nTo retry failed packages manually:")
        for package in failed_packages:
            print(f"  pip install {package}")
    else:
        print(f"\n🎉 All packages installed successfully!")
    
    print(f"\n💡 Next steps:")
    print(f"1. Set up your .env file with API keys")
    print(f"2. Configure Google Cloud authentication")
    print(f"3. Test the installation with the notebooks")

if __name__ == "__main__":
    main()