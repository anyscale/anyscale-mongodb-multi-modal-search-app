# Enterprise Multi-Modal Hybrid Search System

Welcome to the **Enterprise Multi-Modal Hybrid Search System** repository! This project demonstrates a comprehensive solution for implementing a scalable multi-modal search system that combines both legacy and modern search techniques, leveraging cutting-edge technologies and infrastructure.

## Multi-Modal Data Pipelines at Scale

### Overview
The project includes scalable offline batch inference and embeddings generation capabilities, designed to handle both text and image data. Key features include:
- Efficient processing, embedding, and upserting of text and images.
- Utilization of heterogeneous hardware for optimal performance.
- Production-ready data pipelines powered by Anyscale.
- MongoDB as the central data repository.
- Scalable updating of embeddings in MongoDB.

### Directory Structure
- **data_pipelines/** 
  - **data.py**: Handles data I/O operations, transferring data from the Data Lake to MongoDB.
  - **offline_compute.py**: Manages offline batch inference and embeddings generation.
  - **online_compute.py**: Responsible for online inference and embeddings generation.

### Technology Stack
- **ray[data]**
- **vLLM**
- **pymongo**
- **sentence-transformers**

## Multi-Modal Search at Scale

### Overview
The search backend combines legacy text matching with advanced semantic search capabilities, offering a robust hybrid search solution. Key features include:
- **Service 1**: Legacy search utilizing text matching and metadata, powered by MongoDB Atlas.
- **Service 2**: Advanced semantic search using multi-modal models and large language models (LLMs), ensuring enhanced accuracy and relevance.
- All models are self-hosted to ensure security and privacy.

### Directory Structure
- **applications/** 
  - **backend.py**: Implements the hybrid search backend.
  - **frontend.py**: Provides a Gradio-based UI for interacting with the search backend.

### Technology Stack
- **ray[serve]**
- **gradio**
- **motor**
- **sentence-transformers**






