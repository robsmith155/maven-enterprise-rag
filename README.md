# SEC 10-K RAG Benchmarking Project

## Introduction

This project was completed as part of the [Maven Enterprise RAG course](https://maven.com/boring-bot/advanced-llm), aimed at developing a benchmarking dataset for Retrieval-Augmented Generation (RAG) using SEC 10-K filing reports. The project explores the process of creating a robust benchmarking dataset to evaluate and enhance the capabilities of RAG pipelines.

**Note:** LLMs were utilized to help develop and improve the code.

## Project Purpose

This project aims to develop a robust benchmarking framework for evaluating RAG systems on financial documents, with a specific focus on SEC 10-K reports. Our key objectives include:

- **Creating a multi-reference benchmark dataset**: Developing questions that require synthesizing information from multiple sections within a single report or across different companies' reports
- **Establishing evaluation metrics**: Implementing comprehensive metrics to assess retrieval quality
- **Testing various RAG pipeline configurations**: Evaluating different chunking strategies, embedding models and retrieval methods using the becnhmarking dataset

## Repo structure

### Notebooks

The notebooks used to process and generate the dataset are in numbered order in the root of the repository. The notebooks should be run in order. Briefly, they are:

* `01_process-sec-10k-reports.ipynb` - In the first notebook we download  a set 10-K filing reports using the SEC EDGAR API. We then process these reports and store the processed data in a JSON file.
* `02_dataset-generation.ipynb` - Used to generate the benchmarking dataset from the processed SEC 10-K filing reports. We use the OpenAI API to generate synthetic questions and answers based on the provided report data.
* `03_rag-pipeline-evaluation.ipynb` - Evaluates different RAG retrieval pipelines to determine the best option for finding relevant information from the SEC 10-K reports.
* `04_generate-benchmark-answers.ipynb` - The best RAG pipeline is used to generate answers to the questions in the benchmarking dataset.

Note that a nicely formated version of each notebook is hosted by `nbsanity`. For some reason the link for the first two notebooks is not working, but the most important notebooks can be viewed here:

* [Evaluating RAG pipelines](https://nbsanity.com/static/eafd0f413950bf4cabdac0b12ea80b4e/03_rag-pipelines-evaluation.html)
* [Generating benchmark answers](https://nbsanity.com/static/ac3844551deda30d709f2614a9a25c97/04_generate-benchmark-answers.html)


### Source code

The majority of the code used to process and generate the dataset is in the [src](src) directory.


## Examples

Two files are included in the [examples](examples) folder. These are:

* `benchmark_dataset_reviewed.json` - This is a JSON file containing the benchmark dataset (after manual review). This was used to evaluate the retrieval pipleine and RAG generated answers in this work.
* `report_GOOG_2016.json` - This JSON file contains a sample processed SEC 10-K filing report.

## Setup

### Python environment

The `uv` library was used to manage the virtual environment for this project. My environment used Python version 3.10.12. Once you have cloned the repository, and created a virtual environment, you can install the required packages by running (from within the root of the cloned repo):

`uv pip install -r requirements.txt`

### OpenAI API key

Ensure you have access to the OpenAI API. Set your API key as an environment variable:
   ```bash
   export OPENAI_API_KEY='your-openai-api-key'
   ```

By following these steps, you will have a setup ready to generate and evaluate RAG datasets based on SEC 10-K filings.
