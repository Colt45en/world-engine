# 🔧 RAG System Debug Protocol - World Engine Integration

## 🚨 ISSUE IDENTIFIED: Python/PowerShell Context Mismatch

### **Root Cause Analysis:**
- **Problem**: Attempting to run Python imports directly in PowerShell prompt
- **Context**: PowerShell ≠ Python interpreter
- **Impact**: Import errors blocking RAG functionality integration

### **Environment Diagnostics Checklist:**

## Step 1: Environment Verification

### **PowerShell Diagnostic Commands:**
```powershell
# Check Python installation
python --version

# Check pip availability
pip --version

# List installed packages (check for LangChain)
pip list | findstr langchain
```

### **Expected Output:**
```
Python 3.9.0+
pip 21.0+
langchain                     0.1.0+
langchain-community          0.0.20+
langchain-huggingface        0.0.1+
```

## Step 2: Correct Import Testing

### **Enter Python Environment:**
```powershell
# In PowerShell, enter Python interpreter
python
```

### **Test Core Imports (in Python):**
```python
# Test 1: Core LangChain imports
try:
    from langchain_community.document_loaders import PyPDFLoader, UnstructuredWordDocumentLoader
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_community.vectorstores import Chroma
    print("✅ SUCCESS: All LangChain imports working")
except ImportError as e:
    print(f"❌ IMPORT ERROR: {e}")

# Test 2: World Engine RAG integration
try:
    from pathlib import Path
    import re
    print("✅ SUCCESS: Python standard library imports")
except ImportError as e:
    print(f"❌ STANDARD LIBRARY ERROR: {e}")

# Exit Python
exit()
```

## Step 3: Dependency Installation (if needed)

### **Core LangChain Stack:**
```powershell
# Install/upgrade core dependencies
pip install --upgrade langchain
pip install --upgrade langchain-community
pip install --upgrade langchain-huggingface
pip install --upgrade chromadb
pip install --upgrade pypdf
pip install --upgrade unstructured
```

### **Additional Dependencies for World Engine:**
```powershell
# For advanced document processing
pip install --upgrade python-docx
pip install --upgrade markdown
pip install --upgrade beautifulsoup4

# For embeddings and vector storage
pip install --upgrade sentence-transformers
pip install --upgrade faiss-cpu
```

## Step 4: World Engine RAG Integration Test

### **Create Test Script: `test_rag.py`**
```python
#!/usr/bin/env python3
"""
World Engine RAG Integration Test
"""

import os
from pathlib import Path

def test_world_engine_rag():
    """Test RAG functionality for World Engine documentation"""

    print("🧪 Testing World Engine RAG Integration...")

    # Test 1: Import verification
    try:
        from langchain_community.document_loaders import TextLoader
        from langchain_huggingface import HuggingFaceEmbeddings
        from langchain_community.vectorstores import Chroma
        from langchain.text_splitter import CharacterTextSplitter
        print("✅ LangChain imports successful")
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False

    # Test 2: Load World Engine documentation
    try:
        # Look for World Engine docs
        docs_path = Path("./src")
        if not docs_path.exists():
            print("⚠️  World Engine src directory not found")
            return False

        # Find markdown files
        md_files = list(docs_path.glob("**/*.md"))
        if md_files:
            print(f"✅ Found {len(md_files)} documentation files")

            # Test loading first file
            loader = TextLoader(str(md_files[0]))
            documents = loader.load()
            print(f"✅ Successfully loaded {len(documents)} documents")

        else:
            print("⚠️  No markdown files found in src directory")

    except Exception as e:
        print(f"❌ Document loading failed: {e}")
        return False

    # Test 3: Embeddings
    try:
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        print("✅ Embeddings model loaded successfully")
    except Exception as e:
        print(f"❌ Embeddings failed: {e}")
        return False

    print("🎉 World Engine RAG integration test PASSED!")
    return True

if __name__ == "__main__":
    success = test_world_engine_rag()
    exit(0 if success else 1)
```

### **Run Test:**
```powershell
# Save the test script and run it
python test_rag.py
```

## Step 5: World Engine RAG Enhancement

### **Enhanced RAG for World Engine Components:**
```python
"""
Enhanced World Engine RAG with component-specific knowledge
"""

class WorldEngineRAG:
    def __init__(self):
        self.component_docs = {
            'visual-bleedway': self.load_visual_bleedway_docs(),
            'nexus-room': self.load_nexus_room_docs(),
            'crypto-dashboard': self.load_crypto_dashboard_docs(),
            'sensory-framework': self.load_sensory_docs()
        }

    def load_visual_bleedway_docs(self):
        """Load Visual Bleedway specific documentation"""
        return [
            "Visual Bleedway converts PNG silhouettes to 3D meshes",
            "Uses advanced C++ algorithms: Otsu thresholding, morphological operations",
            "Integrates with sensory framework for contextual overlays",
            "Supports GLB export for external applications"
        ]

    def load_nexus_room_docs(self):
        """Load NEXUS Room specific documentation"""
        return [
            "NEXUS Room provides immersive 3D iframe environment",
            "Features 12 configurable panels across 4 walls",
            "Uses WebGL + CSS3D hybrid rendering",
            "Supports real-time content switching and management"
        ]

    def query_component(self, query: str, component: str = None):
        """Query specific component or all components"""
        if component and component in self.component_docs:
            docs = self.component_docs[component]
        else:
            docs = []
            for comp_docs in self.component_docs.values():
                docs.extend(comp_docs)

        # Simple keyword matching (enhance with embeddings)
        results = [doc for doc in docs if any(word in doc.lower() for word in query.lower().split())]
        return results[:3]  # Top 3 results
```

## 🚀 Integration with Your World Engine

### **Add to your existing RAG component:**
```typescript
// In src/ai/WorldEngineRAG.tsx - enhance with Python backend
export class WorldEngineRAG {
  private pythonBackend: string = 'http://localhost:8000'; // FastAPI server

  async queryWithPython(question: string): Promise<string> {
    try {
      const response = await fetch(`${this.pythonBackend}/query`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: question })
      });

      if (response.ok) {
        const result = await response.json();
        return result.answer;
      }
    } catch (error) {
      console.log('Python backend unavailable, using fallback');
    }

    // Fallback to existing RAG system
    return this.generateContextualResponse(question);
  }
}
```

## 📋 Troubleshooting Checklist

### **Common Issues & Solutions:**

| Issue | Cause | Solution |
|-------|--------|----------|
| `ModuleNotFoundError: langchain` | Package not installed | `pip install langchain` |
| `ImportError: langchain_community` | Wrong import path | Use `langchain_community` not `langchain.community` |
| `Python command not found` | Python not in PATH | Reinstall Python with "Add to PATH" option |
| `Permission denied` | Admin rights needed | Run PowerShell as Administrator |

### **Verification Commands:**
```powershell
# Quick health check
python -c "import langchain_community; print('✅ LangChain working')"
python -c "import chromadb; print('✅ ChromaDB working')"
python -c "from sentence_transformers import SentenceTransformer; print('✅ Embeddings working')"
```

## 🎯 Next Steps After Fix:

1. **Test Import Success**: Run the import commands in Python interpreter
2. **Document Loading**: Test loading your World Engine documentation files
3. **Vector Storage**: Set up ChromaDB with your component documentation
4. **Integration**: Connect Python RAG backend with your TypeScript frontend

**Ready to proceed with the diagnostic commands above?** Let me know the results and we'll fix any remaining issues! 🔧✨
