# ​​Reinventing Multi-Modal Search with Anyscale and MongoDB

Welcome to the **​​Reinventing Multi-Modal Search with Anyscale and MongoDB** repository! This project demonstrates a comprehensive solution for improving search capabilities in an e-commerce setting.

# Quickstart

## Register or login to Anyscale

If you don't have an Anyscale account, you can register [here](https://console.anyscale.com/register/ha?utm_source=github&utm_medium=github&utm_content=multi-modal-search-anyscale-mongodb).

If you already have an account, [login](https://console.anyscale.com/v2?utm_source=github&utm_medium=github&utm_content=multi-modal-search-anyscale-mongodb) here.

## Create a free MongoDB cluster

If you don't have a MongoDB account, follow the instructions [here](https://www.mongodb.com/docs/guides/atlas/account/) to sign up for a free MongoDB Atlas account.

If you don't have a MongoDB cluster created, follow the instructions [here](https://www.mongodb.com/docs/guides/atlas/cluster/) to create a free MongoDB Atlas cluster.

Use the `0.0.0.0/0` network access list to allow public access to your cluster. To do so follow the instructions [here](https://www.mongodb.com/docs/guides/atlas/network-connections/).

Get the connection string for your MongoDB cluster by following the instructions [here](https://www.mongodb.com/docs/guides/atlas/connection-string/).

## Register or login to Hugging Face

If you don't have a Hugging Face account, you can register [here](https://huggingface.co/join). 

If you already have an account, [login](https://huggingface.co/login) here.

Visit the [tokens](https://huggingface.co/settings/tokens) page to generate a new API token.

Visit the following model pages and request access to these models:
- [llava-hf/llava-v1.6-mistral-7b-hf](https://huggingface.co/llava-hf/llava-v1.6-mistral-7b-hf)
- [mistralai/Mistral-7B-Instruct-v0.1](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1)

Once you have access to these models, you can proceed with the next steps.

## Prepare Anyscale Workspace 

### Step 1

Once you are logged in, go to workspaces by clicking the "**Workspaces**" tab on the left side of the home screen:

<img src="https://anyscale-public-materials.s3.us-west-2.amazonaws.com/mongodb-demo/screenshots/workspaces-tab.png" width="800px">

### Step 2

Create a new workspace by clicking the "**Create Workspace**" button:

<img src="https://anyscale-public-materials.s3.us-west-2.amazonaws.com/mongodb-demo/screenshots/workspace-create.png" width="800px">

### Step 3

Specify Workspace details as displayed below and click "**Create**":

<img src="https://anyscale-public-materials.s3.us-west-2.amazonaws.com/mongodb-demo/screenshots/workspace-form.png" width="500px">

### Step 4

Wait for the workspace to be created:

*(this can take up to one minute)*

<img src="https://anyscale-public-materials.s3.us-west-2.amazonaws.com/mongodb-demo/screenshots/workspace-creation-progress-v2.png" width="800px">  

> [!NOTE]
> Your workspace is ready!

### Step 5

Open the terminal in the workspace and clone the repository by running the following command:

```bash
git clone https://github.com/anyscale/mongodb-multi-modal-search-prototype.git
```

<img src="https://anyscale-public-materials.s3.us-west-2.amazonaws.com/mongodb-demo/screenshots/workspace-terminal.png" width="800px">

### Step 6

Go to the project directory:

```bash
cd mongodb-multi-modal-search-prototype/
```

### Step 7

Set the MongoDB connection string `DB_CONNECTION_STRING` and huggingface access token `HF_TOKEN` as environment variables under the Dependencies section of the workspace

<img src="https://anyscale-public-materials.s3.us-west-2.amazonaws.com/mongodb-demo/screenshots/workspace-dependencies.png" width="800px" alt="env-vars-setup-workspace">


### Step 8

Set the proper environment variables in the `data_pipelines/job.yaml` file

```yaml
name: enrich-data-and-upsert
entrypoint: python cli.py --data-path s3://anyscale-public-materials/mongodb-demo/raw/myntra_subset_deduped_10000.csv --nsamples 10000 --mode first_run --collection-name myntra-new
working_dir: .
requirements: requirements.txt
env_vars:
  DB_CONNECTION_STRING: <your mongodb connection string> # replace with your MongoDB connection string
  HF_TOKEN: <your huggingface token> # replace with your Hugging Face token
...
```

## 3. Run an Anyscale Job to enrich the data and insert it into MongoDB

### Step 1

Submit the job by running the following command in your workspace terminal:

```bash
cd data_pipelines/
anyscale job submit -f job.yaml
```

### Step 2

Check the status of the job by visiting the Anyscale Job interface

<img src="https://anyscale-public-materials.s3.us-west-2.amazonaws.com/mongodb-demo/screenshots/job-status.png" width="800px" alt="job-status">

### Step 3
Note, if you are using a small cluster size of "m0", "m2" or "m5" you will need to manually create the search index in MongoDB. 

If instead you chose a larger cluster size, you can skip this step. The search index would have been automatically and programmatically created for you.

To do so, follow these steps
1. Select your database 
2. Click on the "Search Indexes" tab or the "Atlas Search" tab
<img src="https://anyscale-public-materials.s3.us-west-2.amazonaws.com/mongodb-demo/screenshots/create_index_manual.png" width="800px" alt="create-index-manual">
3. Click on the "Create a Search index" button

From here on, follow these steps to build the Atlas Vector Search Index
1. Click on the "JSON Editor" Option
<img src="https://anyscale-public-materials.s3.us-west-2.amazonaws.com/mongodb-demo/screenshots/atlas_vector_search_json_editor_selected.png" width="800px" alt="json-editor-selected">

2. Click Next
3. Copy the JSON [from here](https://github.com/anyscale/mongodb-multi-modal-search-prototype/blob/main/data_pipelines/data.py#L88), select your collection in the left-hand menu and click Next
<img src="https://anyscale-public-materials.s3.us-west-2.amazonaws.com/mongodb-demo/screenshots/atlas_vector_search_json_editor_full.png" width="800px" alt="json-editor-full">

From here on, follow these steps to build the Atlas Search Index
1. Click on the "JSON Editor" Option
<img src="https://anyscale-public-materials.s3.us-west-2.amazonaws.com/mongodb-demo/screenshots/atlas_search_json_editor_selected.png" width="800px" alt="json-editor-selected">

2. Click Next
3. Copy the JSON [from here](https://github.com/anyscale/mongodb-multi-modal-search-prototype/blob/main/data_pipelines/data.py#L69), select your collection in the left-hand menu, and click Next
<img src="https://anyscale-public-materials.s3.us-west-2.amazonaws.com/mongodb-demo/screenshots/atlas_search_json_editor_full.png" width="800px" alt="json-editor-full">

## 4. Deploy an Anyscale Service to serve the application

### Step 1

Deploy the application by running the following command

```bash
cd applications/
anyscale service deploy -f app.yaml
````

### Step 2

Check the status of the service by visiting the service url.

<img src="https://anyscale-public-materials.s3.us-west-2.amazonaws.com/mongodb-demo/screenshots/service-status.png" width="800px" alt="service-deployment">

### Step 3

Visit the Application URL to see the application. You can find the URL under the service "Query" dropdown.

<img src="https://anyscale-public-materials.s3.us-west-2.amazonaws.com/mongodb-demo/screenshots/service-status-url.png" width="800px" alt="service-url">


### Step 4

Query the application by entering a query in the search bar and clicking the search button

<img src="https://anyscale-public-materials.s3.us-west-2.amazonaws.com/mongodb-demo/screenshots/ai_enabled_semantic.png" width="800px" alt="app-query">

## 5. Run an Anyscale Job to perform data updates and upsert it into MongoDB

### Step 1

In case new data is available, or changes have been made to the existing data, you can run the job to update the data in MongoDB. This time however you need to make sure to set the `mode` parameter to `update` in the `data_pipelines/job.yaml` file.

```yaml
name: enrich-data-and-upsert
entrypoint: python cli.py --data-path {path-to-updated-data} --nsamples {number-of-samples} --mode update --collection-name myntra-new  # note the mode parameter
...
```

### Step 2

Submit the job by running the following command in your workspace terminal:

```bash
cd data_pipelines/
anyscale job submit -f job.yaml
```

# Architecture

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


---

Created with ♥️ by [Anyscale](https://anyscale.com/)
