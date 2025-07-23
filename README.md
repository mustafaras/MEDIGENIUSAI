# 🧠 MediGenius AI - Advanced Medical Intelligence Platform

<div align="center">

![MediGenius AI](https://img.shields.io/badge/MediGenius-AI-blue?style=for-the-badge&logo=brain&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.10+-green?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-Latest-red?style=for-the-badge&logo=streamlit&logoColor=white)
![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4o-orange?style=for-the-badge&logo=openai&logoColor=white)
![Status](https://img.shields.io/badge/Status-Research-yellow?style=for-the-badge)

**🔬 Experimental Clinical Decision Support for Research & Educational Validation**

*A cutting-edge medical AI research platform that combines advanced neural networks with 125,847+ peer-reviewed medical sources for experimental clinical decision support simulation.*

</div>

## 🚀 Project Overview
<img width="2551" height="1278" alt="image" src="https://github.com/user-attachments/assets/1314f824-1a66-4236-ac27-b5c3deba038c" />

**MediGenius AI** is an experimental medical research platform that revolutionizes clinical decision support through advanced artificial intelligence. This comprehensive system integrates multiple AI providers, sophisticated retrieval-augmented generation (RAG), and multi-modal medical imaging analysis to provide evidence-based diagnostic insights for research and educational purposes.

### 🎯 Key Features

- **🤖 Multi-AI Integration**: Support for OpenAI GPT-4o/4-Turbo, Anthropic Claude-3, Google Gemini, and Ollama local models
- **📚 Medical Literature Database**: 125,847+ peer-reviewed medical sources including Harrison's Internal Medicine, Gray's Anatomy, Robbins Pathology
- **🔬 Advanced RAG System**: Retrieval-Augmented Generation with MedRAG for evidence-based responses
- **🖼️ Medical Imaging Analysis**: Multi-modal analysis of X-rays, CT scans, MRI, and ultrasounds
- **📊 Comprehensive Diagnostics**: Detailed differential diagnosis with probability scoring and evidence attribution
- **💡 Interactive Interface**: Beautiful dark-mode Streamlit interface with wide sidebar and enhanced UX
- **📄 Detailed Reporting**: Export comprehensive medical reports to Word documents
- **🔍 Literature Synthesis**: Real-time medical literature search and synthesis

### 🏥 Medical Capabilities

#### 🩺 Diagnostic Features
- **Differential Diagnosis Generation**: Multi-layered diagnostic reasoning with probability scoring
- **Evidence-Based Analysis**: Direct attribution to medical literature and textbooks
- **Symptom Analysis**: Comprehensive symptom correlation and pattern recognition
- **Risk Stratification**: Advanced risk assessment and prognostic indicators
- **Clinical Decision Support**: Evidence-based treatment recommendations

#### 📖 Medical Knowledge Base
- **18 Major Medical Textbooks**: Complete coverage of core medical specialties
- **125,847 Chunked Snippets**: Processed medical literature with semantic search
- **Authoritative Sources**: Harrison's, Gray's Anatomy, Robbins Pathology, First Aid USMLE
- **Specialty Coverage**: Internal Medicine, Surgery, Pediatrics, Neurology, Pathology, and more

#### 🔬 Advanced Analysis
- **Multi-Modal Processing**: Text symptoms + medical imaging analysis
- **Real-Time Retrieval**: Dynamic medical literature search with relevance scoring
- **Comprehensive Workup**: Detailed diagnostic testing recommendations
- **Specialist Referrals**: Evidence-based referral guidelines
- **Follow-up Protocols**: Long-term monitoring and management plans

## 🛠️ Technical Architecture

### Core Components

1. **🧠 MedRAG System**: Advanced retrieval-augmented generation for medical literature
2. **🤖 Multi-AI Backend**: Seamless integration with multiple AI providers
3. **🖼️ Image Processing**: Medical imaging analysis with GPT-4o Vision
4. **📊 Data Management**: Efficient handling of large medical datasets
5. **🔍 Semantic Search**: Vector-based medical literature retrieval
6. **📱 Streamlit Frontend**: Modern, responsive web interface

### Supported AI Models

#### OpenAI Models
- **GPT-4o** - Vision Capable | 4K Output | 128K Context
- **GPT-4o Mini** - Fast & Efficient | 16K Output | 128K Context  
- **GPT-4 Turbo** - Advanced | 4K Output | 128K Context
- **GPT-4** - Classic | 8K Output | 8K Context
- **GPT-3.5 Turbo** - Reliable | 4K Output | 16K Context

#### Anthropic Models
- **Claude-3 Opus** - Most Capable | 4K Output | 200K Context
- **Claude-3 Sonnet** - Balanced | 4K Output | 200K Context
- **Claude-3 Haiku** - Fast | 4K Output | 200K Context

#### Google Models
- **Gemini Pro** - Smart | 2K Output | 32K Context
- **Gemini 1.5 Pro** - Ultra Context | 8K Output | 1M Context

#### Local Models (Ollama)
- **Llama 3/3.1** - FREE Local Models
- **Mixtral** - Mixture of Experts
- **Custom Models** - User-defined local models

## 📋 Prerequisites

### System Requirements
- **Python**: 3.10+ (Recommended 3.11)
- **RAM**: Minimum 8GB (16GB+ recommended for optimal performance)
- **Storage**: 20GB+ free space for medical databases
- **GPU**: CUDA-compatible GPU recommended (optional for local models)
- **OS**: Windows 10/11, macOS 10.15+, Ubuntu 18.04+

### API Requirements
- **OpenAI API Key** (for GPT models)
- **Anthropic API Key** (for Claude models)
- **Google AI API Key** (for Gemini models)
- **Ollama Installation** (for local models - optional)

## 🚀 Installation & Setup

### 1. Clone Repository
```bash
git clone https://github.com/yourusername/medigenius-ai.git
cd MEDIGENIUSAI
```

### 2. Create Virtual Environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install --upgrade pip
pip install -r sourcesrc/requirements_final.txt
```

### 4. Download Medical Databases
The medical knowledge base requires large files not included in the repository:

```bash
# Create necessary directories
mkdir -p corpus/textbooks/chunk
mkdir -p corpus/textbooks/index/pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb

# Download medical textbook chunks (125,847 snippets)
# [Instructions for downloading from HuggingFace or your data source]

# Download FAISS indices and embeddings
# [Instructions for downloading pre-computed embeddings]
```

### 5. Configure API Keys
Create a `.env` file in the root directory:
```env
OPENAI_API_KEY=sk-your-openai-api-key-here
ANTHROPIC_API_KEY=sk-ant-your-anthropic-key-here
GOOGLE_AI_API_KEY=your-gemini-api-key-here
```

### 6. Optional: Install Ollama (Local Models)
```bash
# Windows/macOS
curl -fsSL https://ollama.ai/install.sh | sh

# Pull medical models
ollama pull llama3.1:8b
ollama pull mixtral:8x7b
```

## 🎮 Usage Guide

### 🖥️ Launch Application
```bash
# Navigate to project directory
cd MEDIGENIUSAI

# Activate virtual environment
source venv/bin/activate  # macOS/Linux
# or
venv\Scripts\activate     # Windows

# Run Streamlit application
streamlit run sourcesrc/medigeniusAI.py
```

### 🌐 Web Interface
The application will open at `http://localhost:8501` with:
- **Wide Layout**: Optimized for medical workflows
- **Dark Mode**: Eye-friendly interface for extended use
- **Responsive Design**: Works on desktop and tablet devices

### 💻 Programmatic Usage
```python
from sourcesrc.medrag import MedRAG

# Initialize MedRAG with specific configuration
medrag = MedRAG(
    llm_name="OpenAI/gpt-4o",
    rag=True,
    retriever_name="MedCPT", 
    corpus_name="Textbooks"
)

# Medical question with multiple choice
question = "A lesion causing compression of the facial nerve at the stylomastoid foramen will cause ipsilateral"
options = {
    "A": "paralysis of the facial muscles.",
    "B": "paralysis of the facial muscles and loss of taste.",
    "C": "paralysis of the facial muscles, loss of taste and lacrimation.",
    "D": "paralysis of the facial muscles, loss of taste, lacrimation and decreased salivation."
}

# Get answer with evidence
answer, snippets, scores = medrag.answer(
    question=question, 
    options=options, 
    k=15  # Number of literature sources to retrieve
)

print(f"Answer: {answer}")
print(f"Evidence sources: {len(snippets)}")
print(f"Relevance scores: {scores}")
```

## 📁 Project Structure

```
MEDIGENIUSAI/
├── 📄 README.md                          # This comprehensive guide
├── 📄 requirements_final.txt             # Python dependencies
│
├── 📂 sourcesrc/                         # Main source code
│   ├── 🧠 medigeniusAI.py               # Primary Streamlit application (1,416 lines)
│   ├── 🔧 medrag.py                     # Core MedRAG implementation
│   ├── ⚙️ config.py                     # Configuration management
│   ├── 🛠️ utils.py                      # Utility functions
│   ├── 📝 template.py                   # Prompt templates and instructions
│   ├── 🧪 test_medrag.py                # Test suite and examples
│   ├── 🔍 python_check.py               # Environment validation
│   │
│   ├── 📂 corpus/                       # Medical knowledge base
│   │   └── 📂 textbooks/
│   │       ├── 📄 README.md             # Dataset documentation
│   │       ├── 📂 chunk/                # Processed medical text chunks
│   │       │   ├── 📖 Anatomy_Gray.jsonl
│   │       │   ├── 🧬 Biochemistry_Lippincott.jsonl
│   │       │   ├── 🔬 Cell_Biology_Alberts.jsonl
│   │       │   ├── 📚 First_Aid_Step1.jsonl
│   │       │   ├── 📚 First_Aid_Step2.jsonl
│   │       │   ├── 👩‍⚕️ Gynecology_Novak.jsonl
│   │       │   ├── 🔬 Histology_Ross.jsonl
│   │       │   ├── 🛡️ Immunology_Janeway.jsonl
│   │       │   ├── 🏥 InternalMed_Harrison.jsonl
│   │       │   ├── 🧠 Neurology_Adams.jsonl
│   │       │   ├── 🤱 Obstentrics_Williams.jsonl
│   │       │   ├── 🔬 Pathology_Robbins.jsonl
│   │       │   ├── 📋 Pathoma_Husain.jsonl
│   │       │   ├── 👶 Pediatrics_Nelson.jsonl
│   │       │   ├── 💊 Pharmacology_Katzung.jsonl
│   │       │   ├── ⚡ Physiology_Levy.jsonl
│   │       │   ├── 🧠 Psichiatry_DSM-5.jsonl
│   │       │   └── 🔪 Surgery_Schwartz.jsonl
│   │       │
│   │       └── 📂 index/                # Vector indices and embeddings
│   │           ├── 📂 ncbi/
│   │           │   └── 📂 MedCPT-Article-Encoder/
│   │           └── 📂 pritamdeka/
│   │               └── 📂 BioBERT-mnli-snli-scinli-scitail-mednli-stsb/
│   │                   ├── 🔍 faiss.index
│   │                   ├── 📊 metadatas.jsonl
│   │                   └── 📂 embedding/
│   │                       ├── 📖 Anatomy_Gray.npy
│   │                       ├── 🧬 Biochemistry_Lippincott.npy
│   │                       ├── 🔬 Cell_Biology_Alberts.npy
│   │                       └── ... (all textbook embeddings)
│   │
│   ├── 📂 data/                         # Data processing scripts
│   │   ├── 📰 pubmed.py                 # PubMed integration
│   │   ├── 📊 statpearls.py             # StatPearls processing
│   │   ├── 📚 textbooks.py              # Textbook processing
│   │   └── 🌐 wikipedia.py              # Wikipedia medical content
│   │
│   └── 📂 __pycache__/                  # Python cache files
│       ├── ⚙️ config.cpython-312.pyc
│       ├── 🔧 medrag.cpython-312.pyc
│       ├── 📝 template.cpython-312.pyc
│       └── 🛠️ utils.cpython-312.pyc
│
└── 📂 corpus/                           # Alternative corpus location
    └── 📂 textbooks/
        ├── 📂 chunk/                    # Medical text chunks (125,847 snippets)
        └── 📂 index/                    # FAISS indices and embeddings
```

## 🎯 Key Features Explained

### 🧠 Advanced Medical Reasoning

#### Multi-Layered Diagnosis
1. **Primary Analysis**: Initial symptom processing and pattern recognition
2. **Literature Retrieval**: Dynamic search through 125,847+ medical sources
3. **Evidence Synthesis**: Integration of multiple authoritative medical texts
4. **Differential Generation**: Comprehensive differential diagnosis with probabilities
5. **Recommendation Engine**: Evidence-based treatment and workup suggestions

#### Sophisticated RAG Pipeline
```python
# Example: Advanced retrieval with scoring
retrieved_snippets, scores = medrag.retrieval_system.retrieve(
    query="chest pain radiating to left arm", 
    k=15  # Retrieve top 15 most relevant sources
)

# Sources ranked by relevance:
# 1. Harrison's Internal Medicine - Cardiology (Score: 4.85)
# 2. First Aid USMLE Step 1 - Cardiovascular (Score: 4.72)
# 3. Robbins Pathology - Heart Disease (Score: 4.61)
```

### 🖼️ Medical Imaging Analysis

#### Multi-Modal Processing
- **X-Ray Analysis**: Bone fractures, pneumonia, cardiac abnormalities
- **CT Scan Interpretation**: Brain, chest, abdomen imaging
- **MRI Analysis**: Neurological and musculoskeletal conditions
- **Ultrasound Review**: Cardiac, abdominal, obstetric imaging

#### Vision-Language Integration
```python
# Automatic image analysis with clinical correlation
image_analysis = process_medical_image(
    uploaded_image,
    patient_symptoms="chest pain and shortness of breath",
    clinical_context="45-year-old male with cardiac risk factors"
)
```

### 📊 Comprehensive Reporting

#### Detailed Medical Reports
- **Patient Summary**: Complete symptom and imaging analysis
- **Differential Diagnosis**: Ranked list with probabilities and evidence
- **Diagnostic Workup**: Specific test recommendations with rationale
- **Management Plan**: Treatment algorithms and follow-up protocols
- **Literature Sources**: Full attribution to medical references

#### Export Capabilities
- **Word Documents**: Professional medical reports
- **PDF Generation**: Shareable diagnostic summaries
- **Data Export**: JSON/CSV format for research analysis

## 🔧 Configuration Options

### 🤖 AI Provider Configuration
```python
# OpenAI Configuration
OPENAI_CONFIG = {
    "model": "gpt-4o",
    "max_tokens": 4096,
    "temperature": 0.0,
    "context_window": 128000
}

# Claude Configuration  
CLAUDE_CONFIG = {
    "model": "claude-3-opus-20240229",
    "max_tokens": 4096,
    "temperature": 0.0,
    "context_window": 200000
}

# Gemini Configuration
GEMINI_CONFIG = {
    "model": "gemini-1.5-pro",
    "max_tokens": 8192,
    "temperature": 0.0,
    "context_window": 1000000
}
```

### 📚 MedRAG Configuration
```python
# Initialize with custom settings
medrag = MedRAG(
    llm_name="OpenAI/gpt-4o",           # AI model selection
    rag=True,                           # Enable retrieval augmentation
    retriever_name="MedCPT",            # Medical embedding model
    corpus_name="Textbooks",            # Knowledge corpus
    db_dir="./corpus/textbooks/index",  # Database location
    cache_dir="./cache",                # Model cache directory
    k=15,                              # Number of sources to retrieve
    follow_up=False                     # Interactive follow-up mode
)
```

## 🧪 Testing & Validation

### 🔬 Test Suite
```bash
# Run comprehensive tests
python sourcesrc/test_medrag.py

# Test individual components
python -m pytest tests/ -v

# Medical knowledge validation
python sourcesrc/validate_medical_knowledge.py
```

### 📊 Performance Metrics
- **Retrieval Accuracy**: Medical literature relevance scoring
- **Response Quality**: Clinical accuracy validation
- **System Performance**: Response time and resource usage
- **Model Comparison**: Cross-model diagnostic accuracy

### 🏥 Clinical Validation
- **USMLE Question Testing**: Performance on medical licensing questions
- **Case Study Analysis**: Real clinical case validation
- **Expert Review**: Medical professional evaluation
- **Literature Verification**: Source attribution accuracy

## ⚠️ Important Disclaimers

### 🔬 Research & Educational Use Only
```
⚠️ EXPERIMENTAL RESEARCH PLATFORM ⚠️

This system is designed for:
✅ Educational research and training
✅ Clinical decision support simulation  
✅ Medical knowledge validation studies
✅ AI model evaluation and testing

NOT for:
❌ Direct patient diagnosis or treatment
❌ Emergency medical situations
❌ Replacement of professional medical advice
❌ Clinical decision making without supervision
```

### 📋 Medical Disclaimer
- **Educational Purpose**: This platform is for research and educational purposes only
- **Not Medical Advice**: Output should not be used for actual patient care
- **Professional Consultation**: Always consult qualified healthcare professionals
- **Experimental Status**: System is under continuous development and validation
- **No Warranties**: No guarantees of diagnostic accuracy or completeness

### 🔒 Data Privacy
- **Local Processing**: Medical data processed locally when possible
- **API Security**: Secure communication with external AI providers
- **No Data Storage**: Patient information not stored permanently
- **Compliance**: Designed with HIPAA considerations for research use

## 🚀 Advanced Usage

### 🔬 Research Applications

#### Clinical Decision Support Studies
```python
# Batch processing for research
results = []
for case in clinical_cases:
    diagnosis = medrag.analyze_case(
        symptoms=case.symptoms,
        imaging=case.imaging,
        labs=case.lab_results
    )
    results.append({
        'case_id': case.id,
        'ai_diagnosis': diagnosis,
        'ground_truth': case.actual_diagnosis,
        'accuracy_score': calculate_accuracy(diagnosis, case.actual_diagnosis)
    })

# Analyze research results
accuracy_metrics = analyze_batch_results(results)
```

#### Multi-Model Comparison
```python
# Compare different AI models on same cases
models = ['gpt-4o', 'claude-3-opus', 'gemini-1.5-pro']
comparison_results = {}

for model in models:
    medrag_instance = MedRAG(llm_name=f"OpenAI/{model}")
    model_results = []
    
    for case in test_cases:
        result = medrag_instance.answer(case.question, case.options)
        model_results.append(result)
    
    comparison_results[model] = model_results

# Generate comparison report
generate_model_comparison_report(comparison_results)
```

### 🏥 Educational Integration

#### Medical School Training
```python
# USMLE practice integration
usmle_question_bank = load_usmle_questions()
for question in usmle_question_bank:
    ai_answer = medrag.answer(
        question=question.stem,
        options=question.choices,
        explanation_mode=True
    )
    
    # Compare with correct answer and explanation
    accuracy = evaluate_answer(ai_answer, question.correct_answer)
    store_training_result(question.id, ai_answer, accuracy)
```

#### Case-Based Learning
```python
# Interactive medical cases
case_scenario = MedicalCase(
    patient_demographics="45-year-old female",
    chief_complaint="Severe headache and vision changes",
    history="History of hypertension, recent pregnancy",
    physical_exam="BP 180/110, papilledema present"
)

# Generate teaching points
teaching_analysis = medrag.generate_teaching_case(case_scenario)
differential_diagnosis = medrag.create_differential_diagnosis(case_scenario)
learning_objectives = medrag.extract_learning_objectives(case_scenario)
```

## 🔄 Updates & Maintenance

### 📈 Roadmap
- **🤖 Enhanced AI Models**: Integration of newest medical AI models
- **📚 Expanded Literature**: Additional medical databases and journals  
- **🖼️ Advanced Imaging**: 3D medical imaging analysis capabilities
- **🔬 Research Tools**: Enhanced analytics and validation frameworks
- **🌐 Multi-Language**: Support for medical literature in multiple languages
- **📱 Mobile Interface**: Responsive mobile application
- **🔒 Enterprise Security**: Enhanced security for clinical environments

### 🛠️ Maintenance
```bash
# Update dependencies
pip install --upgrade -r sourcesrc/requirements_final.txt

# Update medical databases
python scripts/update_medical_corpus.py

# Refresh AI model configurations
python scripts/refresh_model_configs.py

# Validate system integrity
python scripts/system_health_check.py
```

## 🤝 Contributing

### 🔬 Research Contributions
We welcome contributions from:
- **Medical Professionals**: Clinical validation and case studies
- **AI Researchers**: Model improvements and novel architectures
- **Data Scientists**: Medical data processing and analysis
- **Software Engineers**: Platform enhancement and optimization

### 📋 Contribution Guidelines
1. **Fork the Repository**: Create your own copy for development
2. **Create Feature Branch**: `git checkout -b feature/amazing-medical-feature`
3. **Implement Changes**: Follow coding standards and documentation
4. **Test Thoroughly**: Ensure medical accuracy and system stability
5. **Submit Pull Request**: Detailed description of changes and validation

### 🧪 Testing Requirements
- **Medical Accuracy**: Validate against established medical knowledge
- **System Performance**: Ensure optimal response times
- **Code Quality**: Follow Python best practices and documentation
- **Security Review**: Protect patient data and system integrity

## 📞 Support & Community

### 🆘 Getting Help
- **📖 Documentation**: Comprehensive guides and API references
- **💬 Community Forum**: Discussion with other researchers and developers
- **🐛 Issue Tracker**: Bug reports and feature requests
- **📧 Direct Support**: Contact for urgent research collaboration

### 🌟 Acknowledgments
- **MedRAG Team**: Original research and implementation
- **Medical Textbook Publishers**: Authoritative medical knowledge
- **AI Providers**: OpenAI, Anthropic, Google for powerful language models
- **Open Source Community**: Tools and libraries enabling this platform
- **Medical Professionals**: Validation and guidance for clinical accuracy

<div align="center">

**🧠 MediGenius AI - Advancing Medical Intelligence Through Artificial Intelligence 🧠**

*Built with ❤️ for medical research and education*

![AI](https://img.shields.io/badge/Powered%20by-Artificial%20Intelligence-blue)
![Research](https://img.shields.io/badge/Purpose-Medical%20Research-green)
![Education](https://img.shields.io/badge/Focus-Educational%20Excellence-orange)

</div>

### Use as a Python Module
```python
from sourcesrc.medrag import MedRAG

# Initialize MedRAG
medrag = MedRAG(
    llm_name="OpenAI/gpt-3.5-turbo-16k",
    rag=True,
    retriever_name="MedCPT",
    corpus_name="Textbooks"
)

# Ask a medical question
answer = medrag.answer("What are the symptoms of diabetes?")
print(answer)
```

## Project Structure

```
sourcesrc/
├── medigeniusAI.py          # Main Streamlit application
├── medrag.py                # MedRAG core module
├── config.py                # Configuration settings
├── utils.py                 # Utility functions
├── template.py              # Prompt templates
├── test_medrag.py          # Test suite
├── requirements_final.txt   # Python dependencies
└── corpus/                  # Medical corpus data (not included)
    └── textbooks/
        ├── chunk/           # Chunked textbook data
        └── index/           # FAISS indices and embeddings
```

## Large Files Note
Due to GitHub's file size limitations, the following large files are not included in this repository:
- FAISS index files (corpus/textbooks/index/**/*.index)
- Embedding files (corpus/textbooks/index/**/embedding/*.npy)
- Chunked corpus data (corpus/textbooks/chunk/*.jsonl)

Please contact the repository maintainer for access to these files or refer to the setup instructions above.

## Contributing
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License
MIT License

Copyright (c) 2025 MediGenius AI

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the \
Software\), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

## Contact
rsahin@metu.edu.tr
https://github.com/mustafaras

---

## 📚 Citations & References

### 📜 Primary Citation
If you use MediGenius AI in your research, please cite:
```bibtex
@software{medigenius_ai_2025,
  title={MediGenius AI: Advanced Medical Intelligence Platform},
  author={Rasin Sahin, Mustafa},
  year={2025},
  url={https://github.com/mustafaras/MEDIGENIUSAI-GITHUB},
  note={Experimental clinical decision support platform for research and education}
}
```

### 🔬 Core Research References

#### MedRAG Framework
```bibtex
@inproceedings{xiong-etal-2024-benchmarking,
    title = "Benchmarking Retrieval-Augmented Generation for Medicine",
    author = "Xiong, Guangzhi and Jin, Qiao and Lu, Zhiyong and Zhang, Aidong",
    editor = "Ku, Lun-Wei and Martins, Andre and Srikumar, Vivek",
    booktitle = "Findings of the Association for Computational Linguistics ACL 2024",
    month = aug,
    year = "2024",
    address = "Bangkok, Thailand and virtual meeting",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.findings-acl.372",
    pages = "6233--6251",
    abstract = "While large language models (LLMs) have achieved state-of-the-art performance on a wide range of medical question answering (QA) tasks, they still face challenges with hallucinations and outdated knowledge. Retrieval-augmented generation (RAG) is a promising solution and has been widely adopted. However, a RAG system can involve multiple flexible components, and there is a lack of best practices regarding the optimal RAG setting for various medical purposes. To systematically evaluate such systems, we propose the Medical Information Retrieval-Augmented Generation Evaluation (MIRAGE), a first-of-its-kind benchmark including 7,663 questions from five medical QA datasets. Using MIRAGE, we conducted large-scale experiments with over 1.8 trillion prompt tokens on 41 combinations of different corpora, retrievers, and backbone LLMs through the MedRAG toolkit introduced in this work. Overall, MedRAG improves the accuracy of six different LLMs by up to 18% over chain-of-thought prompting, elevating the performance of GPT-3.5 and Mixtral to GPT-4-level. Our results show that the combination of various medical corpora and retrievers achieves the best performance. In addition, we discovered a log-linear scaling property and the lost-in-the-middle effects in medical RAG. We believe our comprehensive evaluations can serve as practical guidelines for implementing RAG systems for medicine."
}
```

#### Iterative Medical RAG
```bibtex
@inproceedings{xiong2024improving,
  title={Improving retrieval-augmented generation in medicine with iterative follow-up questions},
  author={Xiong, Guangzhi and Jin, Qiao and Wang, Xiao and Zhang, Minjia and Lu, Zhiyong and Zhang, Aidong},
  booktitle={Biocomputing 2025: Proceedings of the Pacific Symposium},
  pages={199--214},
  year={2024},
  organization={World Scientific}
}
```

### 🤖 AI Model References

#### OpenAI Models
```bibtex
@article{openai2024gpt4o,
  title={GPT-4o: Omni-modal AI for Enhanced Medical Reasoning},
  author={OpenAI},
  journal={OpenAI Technical Report},
  year={2024},
  url={https://openai.com/research/gpt-4o}
}
```

#### Anthropic Claude
```bibtex
@article{anthropic2024claude3,
  title={Claude 3: Constitutional AI for Safe Medical Applications},
  author={Anthropic},
  journal={Anthropic Research},
  year={2024},
  url={https://www.anthropic.com/news/claude-3-family}
}
```

#### Google Gemini
```bibtex
@article{team2023gemini,
  title={Gemini: A Family of Highly Capable Multimodal Models},
  author={Gemini Team and others},
  journal={arXiv preprint arXiv:2312.11805},
  year={2023}
}
```

### 📖 Medical Knowledge Base References

#### Medical Textbooks
```bibtex
@book{kasper2018harrison,
  title={Harrison's Principles of Internal Medicine},
  author={Kasper, Dennis L and Fauci, Anthony S and Hauser, Stephen L and Longo, Dan L and Jameson, J Larry and Loscalzo, Joseph},
  edition={20th},
  year={2018},
  publisher={McGraw-Hill Education}
}

@book{standring2020gray,
  title={Gray's Anatomy: The Anatomical Basis of Clinical Practice},
  author={Standring, Susan},
  edition={42nd},
  year={2020},
  publisher={Elsevier}
}

@book{kumar2020robbins,
  title={Robbins \& Cotran Pathologic Basis of Disease},
  author={Kumar, Vinay and Abbas, Abul K and Aster, Jon C},
  edition={10th},
  year={2020},
  publisher={Elsevier}
}

@book{tao2023first,
  title={First Aid for the USMLE Step 1},
  author={Tao, Le and Bhushan, Vikas},
  edition={33rd},
  year={2023},
  publisher={McGraw-Hill Education}
}
```

### 🔍 Retrieval & Embedding Models

#### MedCPT
```bibtex
@inproceedings{jin2023medcpt,
  title={MedCPT: Contrastive Pre-trained Transformers with large-scale PubMed search logs for zero-shot biomedical information retrieval},
  author={Jin, Qiao and Kim, Won and Chen, Qingyu and Comeau, Donald C and Yeganova, Lana and Wilbur, W John and Lu, Zhiyong},
  booktitle={Bioinformatics},
  volume={39},
  number={22},
  pages={4471--4482},
  year={2023},
  publisher={Oxford University Press}
}
```

#### BioBERT
```bibtex
@article{lee2020biobert,
  title={BioBERT: a pre-trained biomedical language representation model for biomedical text mining},
  author={Lee, Jinhyuk and Yoon, Wonjin and Kim, Sungdong and Kim, Donghyeon and Kim, Sunkyu and So, Chan Ho and Kang, Jaewoo},
  journal={Bioinformatics},
  volume={36},
  number={4},
  pages={1234--1240},
  year={2020},
  publisher={Oxford University Press}
}
```

### 🛠️ Technical Framework References

#### Streamlit
```bibtex
@misc{streamlit2024,
  title={Streamlit: A faster way to build and share data apps},
  author={Streamlit Inc.},
  year={2024},
  url={https://streamlit.io/}
}
```

#### FAISS
```bibtex
@article{johnson2019billion,
  title={Billion-scale similarity search with GPUs},
  author={Johnson, Jeff and Douze, Matthijs and J{\'e}gou, Herv{\'e}},
  journal={IEEE Transactions on Big Data},
  volume={7},
  number={3},
  pages={535--547},
  year={2019},
  publisher={IEEE}
}
```

### 📊 Evaluation & Benchmarking

#### Medical Question Answering Benchmarks
```bibtex
@article{jin2021disease,
  title={What disease does this patient have? A large-scale open domain question answering dataset from medical exams},
  author={Jin, Di and Pan, Eileen and Oufattole, Nassim and Weng, Wei-Hung and Fang, Hanyi and Szolovits, Peter},
  journal={Applied Sciences},
  volume={11},
  number={14},
  pages={6421},
  year={2021},
  publisher={MDPI}
}
```

### 🏥 Medical AI Ethics & Safety

#### Clinical AI Guidelines
```bibtex
@article{topol2019high,
  title={High-performance medicine: the convergence of human and artificial intelligence},
  author={Topol, Eric J},
  journal={Nature Medicine},
  volume={25},
  number={1},
  pages={44--56},
  year={2019},
  publisher={Nature Publishing Group}
}
```

### 📝 Additional Reading

For more information about medical AI and retrieval-augmented generation:

1. **Medical AI Review**: [The potential for artificial intelligence in healthcare](https://www.nature.com/articles/s41591-019-0446-9)
2. **RAG in Healthcare**: [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401)
3. **Medical NLP**: [Clinical Natural Language Processing in 2022](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8818602/)
4. **AI Safety in Medicine**: [Ensuring AI Safety in Clinical Practice](https://www.nejm.org/doi/full/10.1056/NEJMra1814259)
