# Mistral7b-Bhagavad-Gita-RAG-AI-Bot
🐣 Please follow me for new updates https://github.com/shrimantasatpati <br />

# 🚦 WIP 🚦
Deployments coming soon!

### Technology Stack
1. FAISS vector database
2. Google Colab - Development/ Inference using T4 GPU
3. [Gradio](https://www.gradio.app/) - Web UI, inference using free-tier Colab T4 GPU
4. [HuggingFace](https://huggingface.co/) - Transformer, Sentence transformers (for creating vector embeddings), Mistral7b quantized model
5. [LangChain](https://www.langchain.com/) - Retrieval augmented generation (RAG) using RetrievalQA chain functionality 


## 🦒 Colab

| Colab | Info
| --- | --- |
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://github.com/shrimantasatpati/Mistral7b-Bhagavad-Gita-RAG-AI-Bot/blob/main/Creating%20FAISS%20vector%20database%20for%20RAG.ipynb) | Creating FAISS vector database from Kaggle dataset
[![Open In Colab](https://github.com/shrimantasatpati/Mistral7b-Bhagavad-Gita-RAG-AI-Bot/blob/main/Mistral7b_Inference_RAG_Bhagavad_Gita.ipynb) | Mistral7b RAG Inference of Bhagavad Gita using Gradio (4bit)


Using BitandBytes configurations (load_in_4bit) for quantization - A bit loss in precision, but performance is almost at par with the Mistral7b (base) model. HuggingFace pipeline for "text-generation". AutoTokenizer and AutoModelforCasualLM from "transformers" for tokenization and loading model from HuggingFace Spaces.

### Dataset
- See - [Kaggle Dataset](https://www.kaggle.com/datasets/shrimantasatpati/bhagavad-gita-english)

### FAISS vector embeddings
- Using sentence-transformers/all-Mini-L6-V2 from [Huggingface]()
- Vector database - [Google Drive](https://drive.google.com/drive/folders/1SVZEN9426k0MPibo4CjhbBcG1ZRa-Oo1?usp=drive_link)

## Main Repo
https://github.com/mistralai/mistral-src <br />

## Paper/ Website
- https://mistral.ai/news/announcing-mistral-7b/ <br />
- https://arxiv.org/abs/2212.04356 <br />
- https://mistral.ai/



## Output
![image](https://github.com/camenduru/Mistral-colab/assets/54370274/7d74acf5-4659-4235-be6d-75b4396520d9)

![Screenshot 2023-10-09 195917](https://github.com/camenduru/Mistral-colab/assets/54370274/3d691d32-bbea-4d3e-988e-b37b3db5c83e)
