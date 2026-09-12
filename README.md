# HybridRAG-BN: Retrieval-Augmented Framework for Bangla KBQA

This project implements **HybridRAG-BN**, a hybrid retrieval-augmented framework for Bangla Knowledge Base Question Answering (KBQA). The framework achieved **1st place** in the IEEE CS CUET ML Contest 2.0 with token-level F1 scores of 0.71654 (public) and 0.72912 (private).

## 📋 Project Overview

The framework consists of **three main approaches** that progressively improve answer quality:

1. **Approach 1** - Precision-Based RAG
2. **Approach 2** - Coverage-Based RAG
3. **Fine-tuning + Post-processing** - Answer Verification & Refinement

## 🗂️ Project Structure

```
LLM_Final_Codes/
├── Approach1(sub9)/                  # Approach 1: Precision-Based
│   ├── KB_Cleaning1_chunk+embed_gen.ipynb    # KB preprocessing & chunk creation
│   └── Rag+inference.ipynb                   # Hybrid retrieval & inference
│
├── Approach2(sub11)/                 # Approach 2: Coverage-Based
│   ├── KB_cleaning2_chunk+embed_gen.ipynb    # Less aggressive preprocessing
│   └── Rag+Inference+TestRagSave.ipynb       # Retrieval & inference
│
├── Finetune+post_process/            # Approach 3: Fine-tuning & Post-processing
│   ├── 1_TestRagSave_method.ipynb            # Load pre-computed retrieval results
│   ├── 2_-finetune2400+infernce.ipynb        # Fine-tune model for answer verification
│   └── 3_sub16-pp.ipynb                      # Post-processing & final predictions
│
├── files_generated/                  # Submission files & predictions
│   ├── 1_submission9_(approach1).csv          # Predictions from Approach 1
│   ├── 2_submission11_(approach2).csv         # Predictions from Approach 2
│   ├── 3_submission16_(finetuned).csv         # Predictions after fine-tuning
│   └── 4_submission_16(2)_(finetuned+postprocess).csv  # Final predictions
│
└── README.md                         # This file
```

---

## 🔄 Workflow Explanation

### **Overall Pipeline**

```
Knowledge Base
      │
      ├──────────────→ Approach 1
      │                 Precision RAG
      │
      └──────────────→ Approach 2
                        Coverage RAG
                            │
                            ↓
                  Fine-tuned Verification
                            │
                            ↓
                    Post-processing
                     ├─ Approach 1 fallback
                     └─ DDG retrieval
                            │
                            ↓
                       Final Answer
```

---

## 📚 Approach 1: Precision-Based RAG (sub9)

**Goal:** Prioritize answer precision and minimize hallucination through aggressive KB cleaning

### **Components:**

#### 1️⃣ **KB_Cleaning1_chunk+embed_gen.ipynb**
- **Purpose:** Prepare the knowledge base
- **Steps:**
  1. **Aggressive Preprocessing:** Remove Wikipedia boilerplate (headers, footers, navigation blocks, language-list sections)
  2. **Chunking:** Split into overlapping chunks (~800 characters, 150 character overlap)
  3. **Embedding Generation:** Use BGE-M3 model to create dense vector embeddings for all chunks
  4. **Indexing:** Store vectors for fast retrieval

- **Key Settings:**
  - Chunk size: ~800 characters
  - Overlap: 150 characters
  - Total chunks generated: ~55,182 chunks (Approach 1)

#### 2️⃣ **Rag+inference.ipynb**
- **Purpose:** Retrieve relevant context and generate answers
- **Steps:**
  1. **Hybrid Retrieval:**
     - BM25: Keyword-based retrieval
     - BGE-M3: Semantic/dense retrieval
     - Combine both using weighted fusion (65% BM25 + 35% BGE-M3)
  
  2. **Re-ranking:** Use BGE-Reranker to score retrieved passages
  
  3. **Answer Generation:**
     - Model: Gemma-4-31B-Instruct (GGUF quantized)
     - Prompt: Conservative - returns "Context-এ তথ্য নেই" when unsure
     - Max tokens: 150
  
  4. **Retry Mechanism:** If no answer found, expand search and retry

- **Key Settings:**
  - Temperature: 0 (deterministic)
  - Top-p: 1
  - BM25 k: 10, Dense k: 15, Final k: 6
  - Retry k: 10

---

## 📚 Approach 2: Coverage-Based RAG (sub11)

**Goal:** Maximize answer coverage by being less aggressive with KB cleaning

### **Components:**

#### 1️⃣ **KB_cleaning2_chunk+embed_gen.ipynb**
- **Purpose:** Prepare knowledge base with minimal preprocessing
- **Difference from Method 1:**
  - Remove ONLY: header blocks, footer blocks, first type of tool/navigation block
  - Keep more raw content to improve coverage
  - Slightly smaller chunks for better retrieval

- **Key Settings:**
  - Chunk size: ~800 characters (same as Method 1)
  - Total chunks: ~75,819 chunks (more chunks than Method 1)

#### 2️⃣ **Rag+Inference+TestRagSave.ipynb**
- **Purpose:** Retrieve context and generate answers with coverage focus
- **Differences from Method 1:**
  - **Permissive Prompting:** Encourages model to use pretrained knowledge when context is insufficient
  - Avoids abstention responses
  - Aims for maximum coverage even if not perfectly grounded

- **Key Insight:** Approach 2 achieves higher F1 than Approach 1 (0.70223 vs 0.69329 on public leaderboard)

---

## 🎓 Fine-tuning & Post-Processing

**Goal:** Verify and refine answers from Approach 2 using a specialized fine-tuned model

### **Components:**

#### 1️⃣ **1_TestRagSave_method.ipynb**
- **Purpose:** Load and organize pre-computed retrieval results from Approach 2
- **Steps:**
  1. Load Approach 2 predictions
  2. Load retrieved context for each question
  3. Prepare data for fine-tuning

#### 2️⃣ **2_-finetune2400+infernce.ipynb**
- **Purpose:** Fine-tune Gemma-4-31B model for answer verification
- **Training Setup:**
  - **Method:** LoRA (Low-Rank Adaptation) fine-tuning
  - **Base Model:** Gemma-4-31B-Instruct
  - **Quantization:** 4-bit
  - **Training Data:** 2,400 examples (80/20 train-validation split)
  - **Epochs:** 2
  - **Batch Size:** 8 (effective)

- **Model Purpose:** Acts as a "judge"
  - **Inputs:** Question + Context + Candidate Answer (from Method 2)
  - **Task:** Verify and refine the candidate answer
  - **Output:** Improved or verified answer (max 30 tokens)
  - **Strategy:** Preserve correct answers, remove unnecessary details, correct errors

- **Fine-Tuning Hyperparameters:**
  - Learning Rate: 1.5e-5
  - LoRA Rank: 16
  - LoRA Alpha: 32
  - Trainable Parameters: 122.4M (0.39%)

#### 3️⃣ **3_sub16-pp.ipynb**
- **Purpose:** Post-processing and fallback strategies
- **Steps:**

  1. **Merge with Approach 1 Answers:**
     - If fine-tuned model returns "তথ্য নেই" (no info)
     - Replace with Approach 1's answer (more conservative but sometimes has coverage)
     - Improves: 11 predictions, 4 correct

  2. **DuckDuckGo-Assisted Retrieval:**
     - For remaining unresolved cases
     - Extract entity from question
     - Search on DuckDuckGo for Wikipedia articles
     - Generate final answer from external source
     - Improves: 7 predictions, 5 correct

- **Final Results:**
  - Public F1: 0.71654 (↑ 0.01431 improvement)
  - Private F1: 0.72912 (↑ 0.00417 improvement)

---

## 🚀 How to Use

### **Step 1: Prepare Knowledge Base (Choose One)**

**Option A - Precision (Approach 1):**
```
1. Run: Approach1(sub9)/KB_Cleaning1_chunk+embed_gen.ipynb
   → Generates aggressive chunks + embeddings
```

**Option B - Coverage (Approach 2):**
```
1. Run: Approach2(sub11)/KB_cleaning2_chunk+embed_gen.ipynb
   → Generates inclusive chunks + embeddings
```

### **Step 2: Run Initial Inference**

**Approach 1:**
```
2. Run: Approach1(sub9)/Rag+inference.ipynb
   → Produces precision-focused answers
```

**Approach 2:**
```
2. Run: Approach2(sub11)/Rag+Inference+TestRagSave.ipynb
   → Produces coverage-focused answers
```

### **Step 3 (Optional): Enhance with Fine-Tuning**

```
3. Run: Finetune+post_process/1_TestRagSave_method.ipynb
   → Load Approach 2 results

4. Run: Finetune+post_process/2_-finetune2400+infernce.ipynb
   → Train answer verification model

5. Run: Finetune+post_process/3_sub16-pp.ipynb
   → Apply post-processing & generate final answers
```

---

## 🔑 Key Concepts

### **Hybrid Retrieval**
- **BM25:** Keyword matching (fast, interpretable)
- **BGE-M3:** Semantic similarity (contextual understanding)
- **Combination:** Weighted fusion gets best of both worlds

### **Model: Gemma-4-31B-Instruct**
- Large language model fine-tuned for instruction-following
- GGUF format for efficient inference
- Used across all stages of the pipeline

### **LoRA Fine-Tuning**
- Parameter-efficient fine-tuning technique
- Only 0.39% of parameters trainable
- Reduces memory and computational requirements
- Preserves base model knowledge while adapting to task

### **Two Approaches Philosophy**

| Aspect | Approach 1 (Precision) | Approach 2 (Coverage) |
|--------|-------------------|------------------|
| KB Preprocessing | Aggressive | Conservative |
| Prompting | Conservative | Permissive |
| Focus | Quality over quantity | Quantity over over-caution |
| F1 Score (Public) | 0.69329 | 0.70223 |

**Winner:** Approach 2 performs better! Sometimes less aggressive is better.

---

## 📊 Performance Metrics

| Stage | Public F1 | Private F1 | Submission File |
|-------|-----------|------------|------------------|
| Approach 1 (Precision) | 0.69329 | 0.69147 | 1_submission9_(approach1).csv |
| Approach 2 (Coverage) | 0.70223 | 0.69901 | 2_submission11_(approach2).csv |
| Approach 2 + Fine-tuned Verification | 0.71589 | 0.72495 | 3_submission16_(finetuned).csv |
| **Final (with Post-processing)** | **0.71654** | **0.72912** | 4_submission_16(2)_(finetuned+postprocess).csv |

**Cumulative Improvements:**
- Fine-tuning: +1.43% (public), +3.01% (private)
- Post-processing: +0.09% (public), +0.59% (private)

---

## 💡 Tips & Tricks

1. **Start with Approach 2:** The coverage-based approach performs better than precision-based
2. **Fine-tuning helps:** Even small improvements add up (modified only 48/1500 answers!)
3. **Post-processing matters:** Fallback strategies catch edge cases
4. **Token overlap:** When evaluating answers, token-level metrics (F1) are used
5. **External retrieval:** DuckDuckGo helps for truly missing information

---

## 🛠️ Requirements

- Python 3.8+
- Libraries: transformers, torch, llama-cpp-python, bge, unsloth
- GPU recommended for fine-tuning
- Models:
  - BGE-M3 (embedding model)
  - BGE-Reranker-v2-M3 (re-ranking model)
  - Gemma-4-31B-Instruct (LLM)

---

## 📝 Notes

- All code uses **Bangla language** prompts and outputs
- The framework is specifically optimized for **Bangla KBQA**
- Results are competitive in **low-resource language settings**
- Paper includes detailed hyperparameters and configuration in appendices

---

## 🏆 Competition Result

**IEEE CS CUET ML Contest 2.0 - Advanced Track**
- Team: Vinland_Vector
- Rank: **🥇 1st Place**
- Public Leaderboard: 0.71654 F1
- Private Leaderboard: 0.72912 F1

---

## 📖 References & Citations

### **Models & Frameworks**

1. **BGE-M3** (Dense Embedding Model)
   - HuggingFace: https://huggingface.co/BAAI/bge-m3
   - GitHub: https://github.com/FlagOpen/FlagEmbedding

2. **BGE-Reranker-v2-M3** (Cross-Encoder Re-ranker)
   - HuggingFace: https://huggingface.co/BAAI/bge-reranker-v2-m3
   - GitHub: https://github.com/FlagOpen/FlagEmbedding

3. **Gemma-4-31B-Instruct** (LLM for Answer Generation)
   - HuggingFace: https://huggingface.co/google/gemma-4-31b-instruct

4. **BM25** (Lexical Retrieval Algorithm)
   - Paper: https://en.wikipedia.org/wiki/Okapi_BM25
   - Implementation (Rank-BM25): https://github.com/dorianbrown/rank_bm25

5. **FAISS** (Vector Indexing)
   - GitHub: https://github.com/facebookresearch/faiss
   - Paper: https://arxiv.org/abs/1702.08734

6. **llama.cpp** (LLM Inference Framework)
   - GitHub: https://github.com/ggerganov/llama.cpp

7. **LoRA** (Parameter-Efficient Fine-tuning)
   - Paper: https://arxiv.org/abs/2106.09685
   - Implementation (PEFT): https://huggingface.co/docs/peft

8. **HuggingFace Transformers** (Core Library)
   - GitHub: https://github.com/huggingface/transformers
   - Documentation: https://huggingface.co/docs/transformers

### **Dataset & Competition**

- **Dataset:** Indic-RAG-Suite (from AI4Bharat) https://huggingface.co/datasets/ai4bharat/Indic-Rag-Suite
- **Dataset Subset (Used in the training and testing):** https://www.kaggle.com/code/ratnajitdhar08/creating-dataset-for-ml-contest
- **Competition:** IEEE CS CUET ML Contest 2.0 - Advanced Track
- **Paper:** Included in this repository

---

**Last Updated:** 2026-06-14
