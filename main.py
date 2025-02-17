#!/usr/bin/env python
import os
import time
import sys
import json
import re
import pickle
import numpy as np
import PyPDF2
from tqdm import tqdm
from dotenv import load_dotenv
from openai import OpenAI
from datasets import load_dataset
from langchain.text_splitter import RecursiveCharacterTextSplitter
import faiss


load_dotenv()
api_key = os.getenv('OPENAI_API_KEY')
if not api_key:
    print("OPENAI_API_KEY가 설정되어 있지 않습니다.")
    sys.exit(1)
client = OpenAI(api_key=api_key)


ANSWER_MAP = {
    1: "A",
    2: "B",
    3: "C",
    4: "D"
}

def evaluate(test_examples: list, batch_responses: list) -> float:
    predictions = {}
    for resp in batch_responses:
        custom_id = resp.get("custom_id")
        if custom_id:
            try:
                answer_content = resp["response"]["body"]["choices"][0]["message"]["content"]
            except Exception as e:
                answer_content = ""
            predictions[custom_id] = answer_content

    correct = 0
    total = 0
    for ex in test_examples:
        ex_id = ex["id"]
        pred = predictions.get(ex_id, "")
        match = re.search(r"\b([A-D])\b", pred.upper())
        pred_letter = match.group(1) if match else ""
        true_letter = ANSWER_MAP.get(int(ex["answer"]), "")
        if pred_letter == true_letter:
            correct += 1
        total += 1

    accuracy = correct / total if total > 0 else 0.0
    return accuracy

def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\s]", "", text)
    return text.strip()

def get_embedding(text: str) -> np.ndarray:
    try:
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=text
        )
        embedding = response.data[0].embedding
        return np.array(embedding, dtype=np.float32)
    except Exception as e:
        print(f"Embedding creation error: {e}")
        return np.zeros(1536, dtype=np.float32)

def extract_text_from_pdf(file_path: str) -> str:
    try:
        with open(file_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            text = ""
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + " "
            return clean_text(text)
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return ""

def split_text_into_chunks(text: str) -> list:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=200,
    )
    return text_splitter.split_text(text)

class CriminalLawAgent:
    def __init__(self, pdf_list=None, cache_path=None):
        self.pdf_list = pdf_list if pdf_list else []
        self.cache_path = cache_path
        self.documents = []
        self.index = None
        self.doc_embeddings = None

    def build_retrieval_corpus(self):
        if os.path.exists(self.cache_path):
            with open(self.cache_path, "rb") as f:
                cached_data = pickle.load(f)
            self.documents = cached_data["documents"]
            self.doc_embeddings = cached_data["doc_embeddings"]
            self.index = faiss.IndexFlatIP(1536)
            self.index.add(self.doc_embeddings)
            print("Loaded Faiss index and embeddings from cache.")
            return

        all_documents = []
        all_doc_embeddings = []

        for file in self.pdf_list:
            print(f"[Processing PDF] {file}")
            full_text = extract_text_from_pdf(file)
            if not full_text:
                continue
            chunks = split_text_into_chunks(full_text)
            all_documents.extend(chunks)
            embeddings = [get_embedding(chunk) for chunk in chunks]
            all_doc_embeddings.extend(embeddings)

        self.index = faiss.IndexFlatIP(1536)
        self.index.add(np.array(all_doc_embeddings, dtype=np.float32))

        with open(self.cache_path, "wb") as f:
            pickle.dump({
                "documents": all_documents,
                "doc_embeddings": np.array(all_doc_embeddings, dtype=np.float32)
            }, f)

        self.documents = all_documents
        self.doc_embeddings = np.array(all_doc_embeddings, dtype=np.float32)
        print("Created Faiss index and saved cache.")

    def retrieve_context(self, query: str, top_k=3) -> list:
        query_emb = get_embedding(clean_text(query)).reshape(1, -1)
        distances, indices = self.index.search(query_emb, top_k)
        return [self.documents[i] for i in indices[0]]

    def generate_prompt(self, example: dict, retrieved_context: list) -> str:
        prompt = (
            "You are a legal expert specialized in the field of criminal law. "
            "Given the following context, select exactly one correct answer. "
            "Your answer must be a single letter among A, B, C, D, with no additional text.\n\n"
            "Context:\n"
        )
        for i, context_chunk in enumerate(retrieved_context, start=1):
            prompt += f"[Context {i}]: {context_chunk}\n"
        prompt += "\n"
        prompt += (
            f"Question: {example['question']}\n"
            f"A: {example['A']}\n"
            f"B: {example['B']}\n"
            f"C: {example['C']}\n"
            f"D: {example['D']}\n\n"
            "Answer (A, B, C, or D only):"
        )
        return prompt

    def run_evaluation(self, input_file_name: str, output_file_name: str, benchmark_file_name: str):
        dataset_test = load_dataset("HAERAE-HUB/KMMLU", "Criminal-Law", split="test")
        test_examples = dataset_test.to_list()
        for idx, ex in enumerate(test_examples):
            ex["id"] = f"test_{idx}"
        
        self.build_retrieval_corpus()

        if not os.path.exists(input_file_name) or os.stat(input_file_name).st_size == 0:
            batch_inputs = []
            for ex in tqdm(test_examples, desc="Generating prompts"):
                context = self.retrieve_context(ex["question"])
                prompt = self.generate_prompt(ex, context)
                batch_inputs.append({
                    "custom_id": ex["id"],
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": {
                        "model": "gpt-4o-mini",
                        "messages": [{"role": "user", "content": prompt}],
                        "max_tokens": 10
                    }
                })
            with open(input_file_name, "w", encoding="utf-8") as f:
                for item in batch_inputs:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")
            print(f"Prompts generated and saved to '{input_file_name}'.")
        else:
            print(f"Input file '{input_file_name}' already exists and contains data. Skipping prompt generation.")


        batch_input_file = client.files.create(
            file=open(input_file_name, "rb"),
            purpose="batch"
        )


        batch_job = client.batches.create(
            input_file_id=batch_input_file.id,
            endpoint="/v1/chat/completions",
            completion_window="24h"
        )
        batch_id = batch_job.id
        print(f"Batch job created. Batch ID: {batch_id}")
        print("Waiting for batch job to finish...")

        start_time = time.time()
        while True:
            current_status = client.batches.retrieve(batch_id)
            print(f"Batch job status: {current_status.status}")
            if current_status.status == "completed":
                print("Batch job completed successfully.")
                break
            elif current_status.status == "failed":
                print("Batch job failed. Exiting.")
                return
            else:
                print("... 30초 후 재시도합니다.")
            time.sleep(30)
        end_time = time.time()
        benchmark_duration = end_time - start_time


        try:
            output_file_id = client.batches.retrieve(batch_id).output_file_id
            print("Output File ID:", output_file_id)
            result = client.files.content(output_file_id).content
            with open(output_file_name, "wb") as file:
                file.write(result)
            print(f"Batch job results saved to '{output_file_name}'.")
        except Exception as e:
            print("Error saving batch job results:", e)
            return

        try:
            with open(output_file_name, "r", encoding="utf-8") as f:
                batch_responses = [json.loads(line) for line in f if line.strip()]
            evaluated_accuracy = evaluate(test_examples, batch_responses)
            print("Evaluated Accuracy:", evaluated_accuracy)
        except Exception as e:
            print("Error during evaluation:", e)
            evaluated_accuracy = None

        try:
            benchmark_info = {
                "batch_id": batch_id,
                "duration_seconds": benchmark_duration,
                "accuracy_percentage": f"{evaluated_accuracy*100}%" if evaluated_accuracy is not None else None
            }
            with open(benchmark_file_name, "w", encoding="utf-8") as benchmark_file:
                json.dump(benchmark_info, benchmark_file, ensure_ascii=False, indent=1)
            print(f"Benchmark results saved to '{benchmark_file_name}'.")
        except Exception as e:
            print("Error saving benchmark results:", e)

def main():
    input_file_name = "batchinput.jsonl"
    output_file_name = "batchoutput.jsonl"
    benchmark_file_name = "benchmark_result.txt"

    agent = CriminalLawAgent(
        pdf_list=["./embedding/형사소송법.pdf"],
        cache_path="./embedding/batch_형소법_cache.pkl"
    )
    agent.run_evaluation(input_file_name, output_file_name, benchmark_file_name)

if __name__ == "__main__":
    main()
