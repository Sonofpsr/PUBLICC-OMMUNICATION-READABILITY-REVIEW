# PUBLICC-OMMUNICATION-READABILITY-REVIEW
Public Communication Readability Reviewer Develop a GenAI application that processes official public communications. The system should produce plain-language alternatives and meaning-preservation notes. The prototype must retain obligations and exceptions. Expected solution: LLM + text simplification.
# Public-communication-readability-reviewer


# Cell 1: Check Google Colab environment

import torch
import platform

print("Python version:", platform.python_version())
print("PyTorch version:", torch.__version__)

if torch.cuda.is_available():
    print("\nGPU is available!")
    print("GPU name:", torch.cuda.get_device_name(0))
    print("CUDA version:", torch.version.cuda)
else:
    print("\nGPU is not available. Please enable GPU in Runtime settings.")

print("\nDevice:", "cuda" if torch.cuda.is_available() else "cpu")



# Cell 2: Install project dependencies

!pip install -q transformers datasets sentencepiece accelerate evaluate rouge_score sacrebleu textstat

print("All required libraries installed successfully!")

# Cell 3: Download and inspect WikiAuto

from datasets import load_dataset
import pandas as pd

BASE_URL = (
    "https://huggingface.co/datasets/chaojiang06/wiki_auto/"
    "resolve/630d36642c619402531722435668486f9fbafb52/"
    "manual/"
)

data_files = {
    "train": BASE_URL + "wiki_auto-train.parquet",
    "validation": BASE_URL + "wiki_auto-dev.parquet",
    "test": BASE_URL + "wiki_auto-test.parquet"
}

# Download the dataset
dataset = load_dataset("parquet", data_files=data_files)

print("=" * 65)
print("             WIKIAUTO DATASET")
print("=" * 65)

# Dataset structure
print("\n1. DATASET STRUCTURE")
print(dataset)

# Number of records
print("\n2. DATASET SPLIT SIZES")

for split in dataset:
    print(f"{split}: {len(dataset[split]):,} records")

# Actual column names
print("\n3. COLUMN NAMES")
print(dataset["train"].column_names)

# Display five actual records
print("\n4. FIRST FIVE DATASET RECORDS")



sample_df = pd.DataFrame(
    dataset["train"].select(range(5))
)

display(sample_df)

# Display first record in full
print("\n5. FIRST RECORD DETAILS")

for column in dataset["train"].column_names:
    print(f"\n{column}:")
    print(dataset["train"][0][column])

print("\n" + "=" * 65)
print("          DATASET INSPECTION COMPLETED")
print("=" * 65)



# Cell 4: Load WikiAuto auto_acl (aligned sentence pairs)

import requests
from datasets import Dataset, DatasetDict

# Official WikiAuto auto_acl files
source_url = (
    "https://raw.githubusercontent.com/chaojiang06/wiki-auto/"
    "master/wiki-auto/ACL2020/train.src"
)

target_url = (
    "https://raw.githubusercontent.com/chaojiang06/wiki-auto/"
    "master/wiki-auto/ACL2020/train.dst"
)

print("Downloading original English sentences...")
source_response = requests.get(source_url, timeout=120)
source_response.raise_for_status()

print("Downloading simplified English sentences...")
target_response = requests.get(target_url, timeout=120)
target_response.raise_for_status()

# Read paired sentences
source_sentences = source_response.text.splitlines()
target_sentences = target_response.text.splitlines()

# Verify alignment
assert len(source_sentences) == len(target_sentences), (
    "Source and target sentence counts do not match!"
)

print("\nTotal aligned sentence pairs:", len(source_sentences))

# Create dataset
full_dataset = Dataset.from_dict({
    "normal_sentence": source_sentences,
    "simple_sentence": target_sentences
})

# Remove empty records
full_dataset = full_dataset.filter(
    lambda x: (
        len(x["normal_sentence"].strip()) > 0
        and len(x["simple_sentence"].strip()) > 0
    )
)

# Split: 80% training, 10% validation, 10% testing
split_1 = full_dataset.train_test_split(
    test_size=0.20,
    seed=42
)

split_2 = split_1["test"].train_test_split(
    test_size=0.50,
    seed=42
)

dataset = DatasetDict({
    "train": split_1["train"],
    "validation": split_2["train"],
    "test": split_2["test"]
})

print("\n========== FINAL DATASET ==========")
print(dataset)

for split in dataset:
    print(f"{split}: {len(dataset[split]):,} pairs")

print("\n========== SAMPLE SENTENCES ==========")

for i in range(3):
    print(f"\nExample {i+1}")
    print("ORIGINAL:", dataset["train"][i]["normal_sentence"])
    print("SIMPLE:  ", dataset["train"][i]["simple_sentence"])

print("\nWikiAuto training dataset is ready!")



# Cell 5: Load T5 tokenizer

from transformers import AutoTokenizer

MODEL_NAME = "google-t5/t5-small"

# Load pretrained tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

print("T5 tokenizer loaded successfully!")

# Test sentence
sample_text = (
    "The applicant shall submit the prescribed "
    "documentation within the stipulated period."
)

# T5 simplification instruction
input_text = "simplify: " + sample_text

# Tokenize
tokens = tokenizer(
    input_text,
    return_tensors="pt"
)

print("\nOriginal sentence:")
print(sample_text)

print("\nInstruction added:")
print(input_text)

print("\nToken IDs:")
print(tokens["input_ids"])

print("\nDecoded tokens:")
print(tokenizer.convert_ids_to_tokens(tokens["input_ids"][0]))

print("\nToken count:")
print(tokens["input_ids"].shape[1])



# Cell 6: Prepare WikiAuto for T5 training

MAX_INPUT_LENGTH = 128
MAX_TARGET_LENGTH = 128

# Select manageable subsets
train_data = dataset["train"].shuffle(seed=42).select(range(20000))
validation_data = dataset["validation"].shuffle(seed=42).select(range(2000))
test_data = dataset["test"].shuffle(seed=42).select(range(2000))

# Prepare T5 input and target tokens
def preprocess_function(examples):

    inputs = [
        "simplify: " + text
        for text in examples["normal_sentence"]
    ]

    targets = examples["simple_sentence"]

    model_inputs = tokenizer(
        inputs,
        max_length=MAX_INPUT_LENGTH,
        truncation=True,
        padding="max_length"
    )

    labels = tokenizer(
        text_target=targets,
        max_length=MAX_TARGET_LENGTH,
        truncation=True,
        padding="max_length"
    )

    # Ignore padding tokens when calculating training loss
    model_inputs["labels"] = [
        [
            token if token != tokenizer.pad_token_id else -100
            for token in sequence
        ]
        for sequence in labels["input_ids"]
    ]

    return model_inputs


# Tokenize all three splits
tokenized_dataset = {
    "train": train_data.map(
        preprocess_function,
        batched=True,
        remove_columns=train_data.column_names
    ),
    "validation": validation_data.map(
        preprocess_function,
        batched=True,
        remove_columns=validation_data.column_names
    ),
    "test": test_data.map(
        preprocess_function,
        batched=True,
        remove_columns=test_data.column_names
    )
}

print("DATASET TOKENIZATION COMPLETED")

for split, data in tokenized_dataset.items():
    print(f"{split}: {len(data):,} examples")

print("\nTokenized columns:")
print(tokenized_dataset["train"].column_names)

print("\nInput token shape:")
print(len(tokenized_dataset["train"][0]["input_ids"]))

print("\nTarget token shape:")
print(len(tokenized_dataset["train"][0]["labels"]))



# Cell 7: Load pretrained T5 model

import torch
from transformers import AutoModelForSeq2SeqLM

MODEL_NAME = "google-t5/t5-small"

# Load pretrained sequence-to-sequence model
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

# Move model to GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Model information
total_parameters = sum(p.numel() for p in model.parameters())
trainable_parameters = sum(
    p.numel() for p in model.parameters() if p.requires_grad
)

print("T5 MODEL LOADED SUCCESSFULLY")

print("\nModel name:", MODEL_NAME)
print("Architecture:", model.config.model_type)
print("Encoder layers:", model.config.num_layers)
print("Decoder layers:", model.config.num_decoder_layers)
print("Attention heads:", model.config.num_heads)

print("\nTotal parameters:", f"{total_parameters:,}")
print("Trainable parameters:", f"{trainable_parameters:,}")
print("Training device:", device)



# Cell 8: Configure T5 fine-tuning

from transformers import (
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq
)

training_args = Seq2SeqTrainingArguments(
    output_dir="./PS153_T5_Simplification",

    # Training configuration
    num_train_epochs=2,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    gradient_accumulation_steps=2,

    # Optimization
    learning_rate=5e-5,
    weight_decay=0.01,
    warmup_steps=100,

    # Evaluation and checkpoints
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_steps=100,
    save_total_limit=2,

    # GPU optimization
    fp16=True,

    # Reproducibility
    seed=42,

    # Training reports
    report_to="none"
)

# Data collator
data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    model=model,
    padding=True
)

print("TRAINING CONFIGURATION CREATED")

print("\nEpochs:", training_args.num_train_epochs)
print("Batch size:", training_args.per_device_train_batch_size)
print("Gradient accumulation:", training_args.gradient_accumulation_steps)
print("Learning rate:", training_args.learning_rate)
print("FP16 enabled:", training_args.fp16)
print("Evaluation strategy:", training_args.eval_strategy)

print("\nData collator initialized successfully!")



# Cell 9: Initialize Seq2Seq Trainer

from transformers import Seq2SeqTrainer

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,

    # Training dataset
    train_dataset=tokenized_dataset["train"],

    # Validation dataset
    eval_dataset=tokenized_dataset["validation"],

    # Batch preparation
    data_collator=data_collator,

    # Tokenizer configuration
    processing_class=tokenizer
)

print("SEQ2SEQ TRAINER INITIALIZED")

print("\nModel:", MODEL_NAME)
print("Training examples:", len(tokenized_dataset["train"]))
print("Validation examples:", len(tokenized_dataset["validation"]))
print("Testing examples:", len(tokenized_dataset["test"]))

print("\nTrainer ready for fine-tuning!")



# Cell 10: Fine-tune T5-small on WikiAuto

print("=" * 60)
print("       PS 153 - T5 TEXT SIMPLIFICATION TRAINING")
print("=" * 60)

print("\nTraining started...")
print("Model:", MODEL_NAME)
print("Dataset: WikiAuto")
print("Training examples:", len(tokenized_dataset["train"]))
print("Epochs:", training_args.num_train_epochs)
print("Device:", device)

# Start training
training_results = trainer.train()

print("\n" + "=" * 60)
print("       TRAINING COMPLETED")
print("=" * 60)

print("\nTraining loss:", training_results.training_loss)
print("Training runtime (seconds):",
      training_results.metrics["train_runtime"])

# Save the fine-tuned model
trainer.save_model("./PS153_T5_Simplification/final_model")
tokenizer.save_pretrained("./PS153_T5_Simplification/final_model")

print("\nFine-tuned model saved successfully!")



# Cell 11: Select the checkpoint with the lowest validation loss

import os
from transformers import AutoModelForSeq2SeqLM

# Find validation results from training history
eval_logs = [
    log for log in trainer.state.log_history
    if "eval_loss" in log
]

# Select lowest validation loss
best_result = min(eval_logs, key=lambda x: x["eval_loss"])

best_step = best_result["step"]
best_loss = best_result["eval_loss"]

checkpoint_path = (
    f"./PS153_T5_Simplification/checkpoint-{best_step}"
)

print("BEST CHECKPOINT INFORMATION")
print("Best training step:", best_step)
print("Best validation loss:", best_loss)
print("Checkpoint path:", checkpoint_path)

# Load the selected checkpoint
best_model = AutoModelForSeq2SeqLM.from_pretrained(
    checkpoint_path
)

best_model = best_model.to(device)

# Save selected model separately
best_model_path = "./PS153_T5_Simplification/best_model"

best_model.save_pretrained(best_model_path)
tokenizer.save_pretrained(best_model_path)

print("\nBest model loaded and saved successfully!")



# Cell 12: Generate simplified sentences

import torch

best_model.eval()

def simplify_text(text):
    input_text = "simplify: " + text

    inputs = tokenizer(
        input_text,
        return_tensors="pt",
        max_length=128,
        truncation=True
    ).to(device)

    with torch.no_grad():
        outputs = best_model.generate(
            **inputs,
            max_length=128,
            num_beams=4,
            early_stopping=True
        )

    simplified = tokenizer.decode(
        outputs[0],
        skip_special_tokens=True
    )

    return simplified


# Test on five unseen examples
print("=" * 70)
print("       PS 153 - TEXT SIMPLIFICATION RESULTS")
print("=" * 70)

for i in range(5):

    original = dataset["test"][i]["normal_sentence"]
    reference = dataset["test"][i]["simple_sentence"]

    prediction = simplify_text(original)

    print(f"\nEXAMPLE {i+1}")
    print("-" * 70)

    print("ORIGINAL:")
    print(original)

    print("\nMODEL SIMPLIFICATION:")
    print(prediction)

    print("\nREFERENCE SIMPLIFICATION:")
    print(reference)

    print("-" * 70)



# Cell 14: Install document processing and UI libraries

!pip install -q pypdf python-docx python-pptx gradio

print("Document processing libraries installed successfully!")



# Cell 15: Extract text from PDF, DOCX, PPTX and TXT

import os
from pypdf import PdfReader
from docx import Document
from pptx import Presentation


def extract_text(file_path):

    if not file_path:
        return "Error: No file uploaded."

    extension = os.path.splitext(file_path)[1].lower()

    extracted_text = []

    # PDF extraction
    if extension == ".pdf":

        reader = PdfReader(file_path)

        for page_number, page in enumerate(reader.pages, start=1):

            text = page.extract_text() or ""

            extracted_text.append(
                f"\n--- PAGE {page_number} ---\n{text}"
            )

    # DOCX extraction
    elif extension == ".docx":

        document = Document(file_path)

        for paragraph in document.paragraphs:
            if paragraph.text.strip():
                extracted_text.append(paragraph.text)

        # Extract tables
        for table in document.tables:

            for row in table.rows:

                row_text = " | ".join(
                    cell.text.strip() for cell in row.cells
                )

                extracted_text.append(row_text)

    # PPTX extraction
    elif extension == ".pptx":

        presentation = Presentation(file_path)

        for slide_number, slide in enumerate(
            presentation.slides, start=1
        ):

            extracted_text.append(
                f"\n--- SLIDE {slide_number} ---"
            )

            for shape in slide.shapes:

                if shape.has_text_frame:
                    if shape.text.strip():
                        extracted_text.append(shape.text)

                if shape.has_table:

                    for row in shape.table.rows:

                        row_text = " | ".join(
                            cell.text.strip()
                            for cell in row.cells
                        )

                        extracted_text.append(row_text)

    # TXT extraction
    elif extension == ".txt":

        with open(
            file_path, "r", encoding="utf-8", errors="replace"
        ) as file:

            extracted_text.append(file.read())

    else:
        return "Error: Unsupported file format."

    result = "\n".join(extracted_text).strip()

    if not result:
        return "Error: No extractable text found. The PDF may be scanned."

    return result


print("Document extraction functions created successfully!")
print("Supported formats: PDF, DOCX, PPTX, TXT")



# Cell 16: Optimized PS 153 Application

import gradio as gr
import torch
import re
import os
import time

# --------------------------------------------------
# 1. MODEL CONFIGURATION
# --------------------------------------------------

best_model.eval()

BATCH_SIZE = 8
MAX_INPUT_LENGTH = 128
MAX_NEW_TOKENS = 96
NUM_BEAMS = 2
MAX_DOCUMENT_CHARACTERS = 30000

print("Optimized inference configuration loaded.")


# --------------------------------------------------
# 2. TEXT EXTRACTION AND CHUNKING
# --------------------------------------------------

def split_into_chunks(text, max_tokens=110):

    sentences = re.split(r'(?<=[.!?])\s+', text.strip())

    chunks = []
    current_chunk = ""

    for sentence in sentences:

        sentence = sentence.strip()

        if not sentence:
            continue

        candidate = (current_chunk + " " + sentence).strip()

        token_count = len(
            tokenizer(candidate, truncation=False)["input_ids"]
        )

        if token_count <= max_tokens:
            current_chunk = candidate

        else:

            if current_chunk:
                chunks.append(current_chunk)

            # Handle exceptionally long individual sentences
            if len(tokenizer(
                sentence, truncation=False
            )["input_ids"]) > max_tokens:

                token_ids = tokenizer(
                    sentence,
                    add_special_tokens=False
                )["input_ids"]

                for start in range(0, len(token_ids), max_tokens):

                    part = token_ids[start:start + max_tokens]

                    chunks.append(
                        tokenizer.decode(
                            part,
                            skip_special_tokens=True
                        )
                    )

                current_chunk = ""

            else:
                current_chunk = sentence

    if current_chunk:
        chunks.append(current_chunk)

    return chunks


# --------------------------------------------------
# 3. BATCH TEXT SIMPLIFICATION
# --------------------------------------------------

def simplify_batch(chunks, user_prompt=""):

    # T5 was trained using the "simplify:" instruction.
    # Additional prompts are experimental guidance only.
    inputs = []

    for chunk in chunks:

        if user_prompt.strip():

            model_input = (
                "simplify: "
                + user_prompt.strip()[:100]
                + " "
                + chunk
            )

        else:
            model_input = "simplify: " + chunk

        inputs.append(model_input)

    encoded = tokenizer(
        inputs,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=MAX_INPUT_LENGTH
    ).to(device)

    with torch.inference_mode():

        if device.type == "cuda":

            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16
            ):

                generated = best_model.generate(
                    **encoded,
                    max_new_tokens=MAX_NEW_TOKENS,
                    num_beams=NUM_BEAMS,
                    early_stopping=True
                )

        else:

            generated = best_model.generate(
                **encoded,
                max_new_tokens=MAX_NEW_TOKENS,
                num_beams=NUM_BEAMS,
                early_stopping=True
            )

    return tokenizer.batch_decode(
        generated,
        skip_special_tokens=True
    )


# --------------------------------------------------
# 4. MAIN DOCUMENT PROCESSING FUNCTION
# --------------------------------------------------

def simplify_document(
    file_path,
    pasted_text,
    user_prompt,
    progress=gr.Progress()
):

    start_time = time.time()

    # Get original text
    if pasted_text and pasted_text.strip():

        original_text = pasted_text.strip()

    elif file_path:

        original_text = extract_text(file_path)

        if original_text.startswith("Error:"):
            raise gr.Error(original_text)

    else:
        raise gr.Error(
            "Please upload a document or enter text."
        )

    if len(original_text) > MAX_DOCUMENT_CHARACTERS:

        raise gr.Error(
            "Document is too large for this demo. "
            "Please use a document containing at most "
            "30,000 characters."
        )

    if not original_text.strip():
        raise gr.Error("No readable text found.")

    # Split document
    chunks = split_into_chunks(original_text)

    if not chunks:
        raise gr.Error("No processable text found.")

    simplified_chunks = []

    total_batches = (
        len(chunks) + BATCH_SIZE - 1
    ) // BATCH_SIZE

    # Process batches
    for batch_number, start in enumerate(
        range(0, len(chunks), BATCH_SIZE),
        start=1
    ):

        batch = chunks[start:start + BATCH_SIZE]

        results = simplify_batch(
            batch,
            user_prompt
        )

        simplified_chunks.extend(results)

        progress(
            batch_number / total_batches,
            desc=f"Processing batch {batch_number}/{total_batches}"
        )

    simplified_text = "\n\n".join(simplified_chunks)

    # Save output
    output_path = (
        "/content/PS153_Simplified_Communication.txt"
    )

    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as file:

        file.write(simplified_text)

    elapsed_time = time.time() - start_time

    status = (
        f"Completed successfully | "
        f"{len(chunks)} chunks | "
        f"{total_batches} batches | "
        f"{elapsed_time:.2f} seconds"
    )

    return (
        simplified_text,
        original_text[:5000],
        output_path,
        status
    )


# --------------------------------------------------
# 5. USER INTERFACE
# --------------------------------------------------

with gr.Blocks(
    theme=gr.themes.Soft(),
    title="PS 153 - Readability Reviewer"
) as demo:

    gr.Markdown(
        """
        # Public Communication Readability Reviewer

        ### AI-powered public document simplification

        Convert complex public communications into
        clearer and easier-to-understand English.
        """
    )

    with gr.Row():

        with gr.Column():

            document_input = gr.File(
                label="Upload your document",
                file_types=[
                    ".pdf",
                    ".docx",
                    ".pptx",
                    ".txt"
                ],
                type="filepath"
            )

            text_input = gr.Textbox(
                label="Or paste your original text",
                placeholder="Enter official communication...",
                lines=7
            )

            prompt_input = gr.Textbox(
                label="Additional simplification instructions",
                value=(
                    "Use clear English for ordinary citizens. "
                    "Preserve obligations, deadlines, amounts "
                    "and exceptions."
                ),
                lines=3
            )

            simplify_button = gr.Button(
                "Simplify Document",
                variant="primary"
            )

        with gr.Column():

            status_output = gr.Textbox(
                label="Processing status",
                interactive=False
            )

            simplified_output = gr.Textbox(
                label="Simplified communication",
                lines=12,
                interactive=False
            )

            download_output = gr.File(
                label="Download simplified document"
            )

    with gr.Accordion(
        "View extracted original document text",
        open=False
    ):

        original_output = gr.Textbox(
            label="Original text (first 5000 characters)",
            lines=10,
            interactive=False
        )

    simplify_button.click(
        fn=simplify_document,
        inputs=[
            document_input,
            text_input,
            prompt_input
        ],
        outputs=[
            simplified_output,
            original_output,
            download_output,
            status_output
        ]
    )

    gr.Markdown(
        """
        **Model:** Fine-tuned T5-small

        **Dataset:** WikiAuto

        **Input formats:** PDF, DOCX, PPTX and TXT

        **Important:** Custom instructions are experimental.
        Verify legal obligations and exceptions against the
        original document before relying on the output.
        """
    )


# --------------------------------------------------
# 6. LAUNCH APPLICATION
# --------------------------------------------------

print("PS 153 optimized application is ready!")

demo.queue(
    default_concurrency_limit=1
).launch(
    share=True,
    debug=False,
    show_error=True
)
