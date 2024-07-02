# Enterprise Multi-Modal Hybrid Search System

Welcome to the **Enterprise Multi-Modal Hybrid Search System** repository! This project demonstrates a comprehensive solution for implementing a scalable multi-modal search system leveraging cutting-edge technologies and infrastructure.

We split our system into an offline data indexing stage and an online search stage.

The offline data indexing stage performs the processing, embedding, and upserting text and images into a MongoDB database that supports vector search across multiple fields and dimensions. This stage is built by running multi-modal data pipelines at scale using Anyscale for AI compute platform.

The online search stage performs the necessary search operations by combining legacy text matching with advanced semantic search capabilities offered by MongoDB. This stage is built by running a multi-modal search backend on Anyscale.

## Multi-Modal Data Pipelines at Scale

### Overview
The data pipelines show how to perform offline batch inference and embeddings generation at scale. The pipelines are designed to handle both text and image data by running multi-modal large language model instances. 

### Directory Structure

- `data_pipelines/` 
  - `data.py`: Handles data I/O operations, transferring data from the Data Lake to MongoDB.
  - `offline_compute.py`: Manages offline batch inference and embeddings computation/
  - `online_compute.py`: A naive implementation via a REST API for online inference and embeddings computation.

### Technology Stack

- `ray[data]`
- `vLLM`
- `pymongo`
- `sentence-transformers`

## Multi-Modal Search at Scale

### Overview
The search backend combines legacy lexical text matching with advanced semantic search capabilities, offering a robust hybrid search solution. 

### Directory Structure
- `applications/` 
  - `backend.py`: Implements the hybrid search backend.
  - `frontend.py`: Provides a Gradio-based UI for interacting with the search backend.

### Technology Stack
- `ray[serve]`
- `gradio`
- `motor`
- `sentence-transformers`






