# Load model directly
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM
from datasets import load_dataset
import json
import argparse
import os
import torch
from pathlib import Path
import transformers
import faiss
import faiss.contrib.torch_utils
from tqdm import tqdm
import numpy as np
import jsonlines

from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, List, Literal
# from dynamic_memory_with_chunk import External_Memory
from torch.utils.data import Dataset, DataLoader, Sampler
from transformers import AutoTokenizer

from datasets import load_dataset
    
def apply_chat_template(chat, tokenizer, return_tensors=None):
    corpus = f"{tokenizer.bos_token}" if tokenizer.bos_token else ""
    for message in chat:
        message_text = "<|start_header_id|>{}<|end_header_id|>\n\n{}<|eot_id|>".format(message["role"], message["content"].strip())
        corpus += message_text
    if return_tensors =='pt':
        return tokenizer(corpus)
        
    return corpus

class SessionDataset(Dataset):
    def __init__(self, data_dict, tokenizer, max_length=8192):
        """
        Initialize the dataset with a dictionary of session IDs and text strings.

        Args:
            data_dict (dict): Dictionary where keys are session IDs and values are text strings.
            tokenizer: Tokenizer from transformers library.
            max_length (int): Maximum length for tokenization.
        """
        self.data = list(data_dict.values())
        self.ids = list(data_dict.keys())
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        encoded = self.tokenizer(self.data[idx], return_tensors="pt", truncation=True, max_length=self.max_length)
        return encoded['input_ids'][0], self.ids[idx], encoded['input_ids'].shape[1]  # Return input_ids and its length


class LengthSortedBatchSampler(Sampler):
    def __init__(self, dataset, batch_size):
        """
        Custom Sampler to batch elements of similar lengths together.

        Args:
            dataset: The dataset to sample from.
            batch_size: The number of elements in each batch.
        """
        self.dataset = dataset
        self.batch_size = batch_size
        # Calculate lengths
        self.lengths = [item[2] for item in dataset]  # Assuming __getitem__ returns (input_ids, id, length)
        # Sort indices by length
        self.sorted_indices = sorted(range(len(self.lengths)), key=lambda idx: self.lengths[idx])

    def __iter__(self):
        # Yield batches of indices based on sorted lengths
        for i in range(0, len(self.sorted_indices), self.batch_size):
            yield self.sorted_indices[i:i + self.batch_size]

    def __len__(self):
        return (len(self.sorted_indices) + self.batch_size - 1) // self.batch_size


@dataclass
class DataCollator(object):
    def __init__(self, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.max_len = max_len

    def pad_sequence(self, sequence, padding_value=0):
        """Pad a sequence to the desired max length."""
        if self.tokenizer.padding_side == "left":
            sequence = [torch.flip(_input_ids, [0]) for _input_ids in sequence] 
        padded_input_ids = torch.nn.utils.rnn.pad_sequence(
            sequence,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id)
        if self.tokenizer.padding_side == "left":
            padded_input_ids = torch.flip(padded_input_ids, [1])
        return padded_input_ids

    def __call__(self, batch):
        input_ids, ids = zip(*[(item[0], item[1]) for item in batch])  # Extract input_ids and ids from the batch
        input_ids = list(input_ids)
        ids = list(ids)
        batch_input_ids = self.pad_sequence(input_ids)
        return batch_input_ids, ids


@dataclass
class MemArguments:
    use_gpu_to_search: Optional[bool] = field(default=True)
    k: Optional[int] = field(default=16)
    memory_size: int = field(default=512000)
    chunk_size: Optional[int] = field(default=512)
    layer_index: int = field(default=16)

@dataclass
class TrainingArguments:
    data_dir: Path = field(default="/home/weizhi/data/longmemeval/")
    out_dir: Optional[Path] = field(default=None)
    outfile_prefix: Optional[str] = field(default=None)
    num_memory_layer: Optional[int] = field(default=16)
    batch_size: int = field(default=16)
    
    # basic parameters
    retriever: Literal['Eric', 'John', 'Graham', 'Terry'] = 'flat-contriever'
    granularity: Literal['session', 'turn'] = 'session'


@dataclass
class DataArguments:
    data_path: str = field(default=None)
    lazy_preprocess: bool = False
    is_multimodal: bool = False
    image_folder: Optional[str] = field(default=None)
    image_aspect_ratio: str = 'square'


def parse_session(data, granularity, sess_id, tokenizer):
    corpus = []

    if granularity == 'session':
        return tokenizer.apply_chat_template(data, tokenize=False) #' '.join([interact['content'] for interact in data])
        # corpus.append(text)
        # ids = [sess_id]
        # if 'answer' in sess_id and all([not turn['has_answer'] for turn in [x for x in data if x['role'] == 'user']]):
        #     ids = [sess_id.replace('answer', 'noans')]
    elif granularity == 'turn':
        ids = []
        for i_turn, turn in enumerate(data):
            if turn['role'] == 'user':
                corpus.append(turn['content'])
                if 'answer' not in sess_id:
                    ids.append(sess_id + '_' + str(i_turn+1))
                else:
                    assert 'has_answer' in turn
                    assert turn['has_answer'] in [True, False]
                    if turn['has_answer']:
                        ids.append(sess_id + '_' + str(i_turn+1))
                    else:
                        ids.append((sess_id + '_' + str(i_turn+1)).replace('answer', 'noans'))
                        assert 'answer' not in ids[-1]
    else:
        raise NotImplementedError
    
    return corpus #, ids #[timestamp for _ in corpus]

# def main(mem_cfg, data_cfg, training_cfg):

#     tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
#     model = LlamaModel.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
    
#     tokenizer.padding_side = 'left'
#     tokenizer.pad_token = "<|finetune_right_pad_id|>"
    
#     model_config = model.config
    
#     device = "cuda:2" if torch.cuda.is_available() else "cpu"
#     model = torch.nn.DataParallel(model, device_ids=[2, 3, 4, 5, 6, 7])
    
#     model.to(device)

#     multi_session_data = json.load(open(training_cfg.data_dir / "longmemeval_m.json"))
#     single_session_data = json.load(open(training_cfg.data_dir / "longmemeval_m.json"))
#     oracle_data = json.load(open(training_cfg.data_dir / "longmemeval_oracle.json"))
    
#     total_session_ids = {}
#     for index, entry in enumerate(multi_session_data):
#         for sessionid, session in zip(entry['haystack_session_ids'], entry['haystack_sessions']):
#             if not session:
#                 continue
#             if sessionid not in total_session_ids:
#                 total_session_ids[sessionid] = parse_session(session, training_cfg.granularity, sessionid, tokenizer)
#             else:
#                 try:
#                     assert parse_session(session, training_cfg.granularity, sessionid, tokenizer) == total_session_ids[sessionid]
#                 except Exception as e:
#                     import pdb
#                     pdb.set_trace()
#     print(len(total_session_ids))
    
#     # memorybank = [External_Memory(mem_cfg, model_config, i+model_config.num_hidden_layers//2, device=model.device) for i in range(model_config.num_hidden_layers//2)]
    
#     # memorybank = External_Memory(mem_cfg, model_config, 16, device=model.device)
#     # memorybank = [faiss.IndexFlatL2(model_config.hidden_size//model_config.num_attention_heads) for i in range(model_config.num_key_value_heads)]
#     # for head_idx in range(len(memorybank)):
#     #     faiss.write_index(memorybank[head_idx], str(training_cfg.data_dir / f"head_{head_idx}.index"))
    
#     memorybank = [np.memmap(training_cfg.data_dir / f"keys_layer_{i}.npy", mode="write", dtype=np.float32, shape=(len(total_session_ids), model_config.num_key_value_heads, model_config.hidden_size//model_config.num_attention_heads)) for i in range(model_config.num_hidden_layers)]
    
#     max_id_length = max([len(sessionid) for sessionid in total_session_ids.keys()])
#     vals = np.memmap(training_cfg.data_dir / "vals.npy", dtype=f'S{max_id_length}', mode='write', shape=(len(total_session_ids), ))
#     vals_strings = []
#     print('np memmap loading done')
    
#     # for idx, (sessionid, session) in tqdm(enumerate(total_session_ids.items())):
#     dataset = SessionDataset(total_session_ids, tokenizer)
#     collator = DataCollator(tokenizer, max_len=4096)
#     sampler = LengthSortedBatchSampler(dataset, training_cfg.batch_size)
#     dataloader = DataLoader(dataset, batch_sampler=sampler, collate_fn=collator)
#     global_index = 0
#     for batch_input_ids, batch_ids in tqdm(dataloader):
#         with torch.no_grad():
#             # output = model(input_ids=tokenizer(session, return_tensors='pt').input_ids, use_cache=True)
#             output = model(input_ids=batch_input_ids.to(device), use_cache=True)
#             import time
#             start = time.time()
#             for layer_index in range(len(memorybank)):

#                 memorybank[layer_index][global_index:global_index+batch_input_ids.shape[0]] = output['past_key_values'][layer_index][0][:, :, -1, :].cpu().numpy()
#         vals[global_index:global_index+batch_input_ids.shape[0]] = batch_ids
#         vals_strings += batch_ids
#         print("indexing takes {}".format(time.time() - start))
#         global_index += batch_input_ids.shape[0]
    
#     with open(str(training_cfg.data_dir / "vals_strings.txt"), "w") as f:
#         f.write("\n".join(vals_strings))

choices = ["A", "B", "C", "D"]

def format_subject(subject):
    l = subject.split("_")
    s = ""
    for entry in l:
        s += " " + entry
    return s

def format_example(df, idx, include_answer=True):
    # prompt = "The following are multiple choice questions (with answers) about {}.\n\n".format(format_subject(df.loc[idx, "subject"]))
    prompt = df[idx]["question"]
    n_choices = len(df[idx]["choices"])
    for j in range(n_choices):
        prompt += "\n{}. {}".format(choices[j], df[idx]["choices"][j])
    prompt += "\nAnswer with the option's letter from the given choices directly.\nAnswer:"
    if include_answer:
        prompt += " {}\n\n".format(choices[df[idx]["answer"]])
    return prompt

def compute_recall(mem_cfg, data_cfg, training_cfg):

    # tokenizer = AutoTokenizer.from_pretrained(training_cfg.out_dir)
    # model = LlamaForCausalLM.from_pretrained(training_cfg.out_dir)
    config = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2-1.5B-instruct").config
    model = SentenceTransformer("Alibaba-NLP/gte-Qwen2-1.5B-instruct", trust_remote_code=True)
    # In case you want to reduce the maximum length:
    model.max_seq_length = 16384
    
    queries = [
        "how much protein should a female eat",
        "summit define",
    ]
    documents = [
        "As a general guideline, the CDC's average requirement of protein for women ages 19 to 70 is 46 grams per day. But, as you can see from this chart, you'll need to increase that if you're expecting or training for a marathon. Check out the chart below to see how much protein you should be eating each day.",
        "Definition of summit for English Language Learners. : 1  the highest point of a mountain : the top of a mountain. : 2  the highest level. : 3  a meeting or series of meetings between the leaders of two or more governments.",
    ]

    # query_embeddings = model.encode(queries, prompt_name="query")
    # document_embeddings = model.encode(documents)
    # import pdb
    # pdb.set_trace()
    
    # tokenizer.padding_side = 'left'
    # tokenizer.pad_token = "<|finetune_right_pad_id|>"
    
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    # model = torch.nn.DataParallel(model, device_ids=[4, 5, 6, 7])
    
    model.to(device).half()
    # model.eval()
    head_dim = int(config.hidden_size / config.num_attention_heads)
    # eos_token_id = [tokenizer.encode(x)[-1] for x in [tokenizer.eos_token, "\n\n"]]
    head_index = [faiss.IndexFlatIP(head_dim) for i in range(config.num_attention_heads)]

    multi_session_data = json.load(open(training_cfg.data_dir / "longmemeval_m"))
    single_session_data = json.load(open(training_cfg.data_dir / "longmemeval_s"))
    oracle_data = json.load(open(training_cfg.data_dir / "longmemeval_oracle"))

    total_session_ids = {}
    for index, entry in enumerate(multi_session_data):
        for sessionid, session in zip(entry['haystack_session_ids'], entry['haystack_sessions']):
            if not session:
                continue
            if sessionid not in total_session_ids:
                session_chats = parse_session(session, training_cfg.granularity, sessionid, model.tokenizer)
                total_session_ids[sessionid] = session_chats

            else:
                try:
                    assert parse_session(session, training_cfg.granularity, sessionid, model.tokenizer) == total_session_ids[sessionid]
                except Exception as e:
                    pass
    
    batch_size = 16
    session_items = list(total_session_ids.items()) # Convert items to list for easier slicing
    num_sessions = len(session_items)

    print(f"Starting batch processing for {num_sessions} sessions with batch size {batch_size}...")

    # Iterate through the sessions in steps of batch_size
    for i in range(0, num_sessions, batch_size):
        # Determine the slice for the current batch
        batch_slice = session_items[i : min(i + batch_size, num_sessions)]

        # Separate session_ids and chats for the current batch (if needed)
        # batch_session_ids = [item[0] for item in batch_slice] # Uncomment if you need IDs later
        batch_chats = [item[1] for item in batch_slice]
        current_batch_actual_size = len(batch_chats)

        if current_batch_actual_size == 0:
            continue # Skip if batch is empty for any reason

        print(f"Processing batch {i//batch_size + 1}/{(num_sessions + batch_size - 1)//batch_size} (Size: {current_batch_actual_size})...")

        # 1. Encode the batch of chats
        # This should return a 2D array/tensor: (current_batch_actual_size, embedding_dimension)
        batch_embeddings = model.encode(batch_chats, batch_size=current_batch_actual_size) # Explicitly pass batch size for clarity if API supports it

        # Ensure it's a numpy array for consistent processing
        if not isinstance(batch_embeddings, np.ndarray):
            batch_embeddings = np.array(batch_embeddings)

        # 2. Reshape the batch embeddings
        # Reshape from (batch_size, embedding_dim) to (batch_size, num_attention_heads, head_dim)
        try:
            # embedding_dim = config.num_attention_heads * head_dim
            # The -1 infers the head_dim automatically
            batch_embeddings_reshaped = batch_embeddings.reshape(
                current_batch_actual_size,
                config.num_attention_heads,
                -1
            )
        except ValueError as e:
            print(f"Error reshaping embeddings for batch starting at index {i}: {e}")
            print(f"  Original batch_embeddings shape: {batch_embeddings.shape}")
            print(f"  Expected first dimension: {current_batch_actual_size}")
            print(f"  Number of attention heads: {config.num_attention_heads}")
            print(f"  Is embedding dimension ({batch_embeddings.shape[1]}) divisible by num_attention_heads ({config.num_attention_heads})?")
            # Handle error appropriately - e.g., skip batch, log, raise
            continue # Skipping this batch

        # 3. Add embeddings to the corresponding head indexes
        for head_idx in range(config.num_attention_heads):
            head_embeddings_batch = batch_embeddings_reshaped[:, head_idx, :]

            # Ensure data is C-contiguous and float32 if required by index (like FAISS)
            if head_embeddings_batch.size > 0:
                try:
                    # head_embeddings_batch_prepared = np.ascontiguousarray(head_embeddings_batch, dtype=np.float32)
                    # Add the batch of vectors for this head to the corresponding index
                    head_index[head_idx].add(head_embeddings_batch)
                except Exception as e:
                    print(f"Error adding embeddings to head_index[{head_idx}] for batch starting at {i}: {e}")
    
    try:
        # exist_ok=True prevents an error if the directory already exists
        os.makedirs(training_cfg.out_dir, exist_ok=True)
        print(f"Output directory '{training_cfg.out_dir}' ensured.")
    except OSError as e:
        print(f"Error: Could not create directory '{training_cfg.out_dir}'. {e}")
        # Depending on your needs, you might want to exit or raise the error here
        # For now, we'll print the error and attempt to continue,
        # but saving will likely fail if the directory wasn't created.
        pass # Or raise e

    # --- Save FAISS Head Indices ---
    print(f"\nSaving {len(head_index)} FAISS head indices to '{training_cfg.out_dir}'...")
    save_successful_indices = 0
    for head_idx, index_obj in enumerate(head_index):
        # Create a unique, indexed filename for each head index
        index_filename = f"head_index_{head_idx}.index"
        index_filepath = os.path.join(training_cfg.out_dir, index_filename) # Use os.path.join for cross-platform compatibility

        try:
            print(f"  Attempting to save index for head {head_idx} to '{index_filepath}'...")
            # Use faiss utility function to write the index to a file
            faiss.write_index(index_obj, index_filepath)
            print(f"  Successfully saved '{index_filename}'")
            save_successful_indices += 1
        except AttributeError as e:
            print(f"  Error saving index {head_idx}: Object doesn't seem to be a valid FAISS index. {e}")
        except Exception as e:
            # Catch other potential errors during writing (e.g., permissions, disk full)
            print(f"  Error saving index for head {head_idx} to '{index_filepath}': {e}")
            # Optional: Decide if you want to continue saving others or stop

    if save_successful_indices == len(head_index):
        print("All FAISS indices saved successfully.")
    else:
        print(f"Warning: Only {save_successful_indices} out of {len(head_index)} FAISS indices were saved successfully due to errors.")


    # --- Save total_session_ids Dictionary to JSON ---
    print(f"\nSaving total_session_ids dictionary to '{training_cfg.out_dir}'...")
    json_filename = "total_session_ids.json"
    json_filepath = os.path.join(training_cfg.out_dir, json_filename)

    try:
        print(f"  Attempting to save dictionary to '{json_filepath}'...")
        with open(json_filepath, 'w', encoding='utf-8') as f:
            # Use indent=4 for pretty-printing the JSON file (makes it human-readable)
            # ensure_ascii=False is good practice for handling various characters
            json.dump(total_session_ids, f, indent=4, ensure_ascii=False)
        print(f"  Successfully saved '{json_filename}'")
    except TypeError as e:
        # This can happen if total_session_ids contains non-serializable types (e.g., sets, custom objects)
        print(f"  Error: Could not serialize total_session_ids to JSON. Check data types. {e}")
    except IOError as e:
        # This can happen due to file system issues (permissions, disk full)
        print(f"  Error: Could not write JSON file to '{json_filepath}'. {e}")
    except Exception as e:
        print(f"  An unexpected error occurred while saving JSON: {e}")

    print(f"\nSaving process finished. Check the '{training_cfg.out_dir}' directory.")

    # dataset_len = len(total_session_ids)
    # key_cache = np.memmap(training_cfg.data_dir / f"keys_layer_{mem_cfg.layer_index}.npy", mode="r", dtype=np.float32, shape=(len(total_session_ids), model_config.num_key_value_heads, model_config.hidden_size//model_config.num_attention_heads))
    # key_cache = torch.from_numpy(np.array(key_cache))
    
    # memorybank = [External_Memory(mem_cfg, model_config, i+model_config.num_hidden_layers//2, device=model.device) for i in range(model_config.num_hidden_layers//2)]
    # memorybank = External_Memory(mem_cfg, model_config, 16, device=model.device)
    # memorybank = [faiss.IndexFlatIP(model_config.hidden_size//model_config.num_attention_heads) for i in range(model_config.num_key_value_heads)]
    # for head_idx in range(len(memorybank)):
    #     memorybank[head_idx].add(key_cache[:, head_idx, :].contiguous())
    
    # max_id_length = max([len(sessionid) for sessionid in total_session_ids.keys()])
    # # vals = np.memmap(training_cfg.data_dir / "vals.npy", dtype=f'S{max_id_length}', mode='r', shape=(len(total_session_ids), ))
    # vals = open(training_cfg.data_dir / "vals_strings.txt").read().split("\n")
    # vals_to_idx = {id: i for i, id in enumerate(vals)}

    # for head_idx in range(len(memorybank)):
    #     faiss.write_index(memorybank[head_idx], str(training_cfg.data_dir / f"head_{head_idx}.index"))
    # head_wise_recall = [0 for i in range(model_config.num_attention_heads)]


    # dev_questions = dataset['dev']
    # subject_to_demonstrations = {}
    # for i in range(0, dev_questions.shape[0], 5):
    #     subject = dev_questions[i]["subject"]
    #     demonstration_prompt = "The following are multiple choice questions (with answers) about {}.\n\n".format(format_subject(subject))
    #     for j in range(i, i+5):
    #         assert dev_questions[j]["subject"] == subject
    #         demonstration_prompt += format_example(dev_questions, j)
    #     subject_to_demonstrations[subject] = demonstration_prompt
    
    # test_questions = dataset["test"]

    # # Build Full Metadata Structure
    # index = {}
    # acc = []
    # # layer_reuse_rate = [[] for i in range(28)]
    # reuse_rate = []

    # for index, entry in tqdm(enumerate(qa_data)):
    #     # sessionid_to_session = {sessionid: parse_session(session, training_cfg.granularity, sessionid, tokenizer) for sessionid, session in zip(entry['haystack_session_ids'], entry['haystack_sessions'])}
        
    #     prompt = entry["context"] + "Question: {} Answer:".format(entry["input"])
    #     # current_session_indices = []
    #     # current_session_ids = []
    #     # for sessionid, session in zip(entry['haystack_session_ids'], entry['haystack_sessions']):
    #     #     if session:
    #     #         current_session_indices.append(vals_to_idx[sessionid])
    #     #         current_session_ids.append(sessionid)
        
    #     # current_keys = torch.index_select(key_cache, 0, torch.tensor(current_session_indices))
    #     # memorybank = [faiss.IndexFlatIP(model_config.hidden_size//model_config.num_attention_heads) for i in range(model_config.num_key_value_heads)]
    #     # for head_idx in range(len(memorybank)):
    #     #     memorybank[head_idx].add(current_keys[:, head_idx, :].contiguous())
        
    #     inputs = tokenizer(prompt, return_tensors='pt')


    #     # with torch.no_grad():
    #     output = model(input_ids=inputs['input_ids'].to(device), use_cache=True, output_attentions=True)
    #     head_group_size = model_config.num_attention_heads // model_config.num_key_value_heads
    #         for head_index in range(model_config.num_attention_heads):

    #             distance, session_id_index = memorybank[head_index//head_group_size].search(output['attentions'][mem_cfg.layer_index][0, head_index, -1:, :].cpu(), mem_cfg.k)
                
    #             # distance, session_id_index = memorybank[head_index].search(output['past_key_values'][mem_cfg.layer_index][0][0, head_index, -1:, :].cpu(), mem_cfg.k)
    #             retrieved_session_ids = [current_session_ids[retrieved_session_index] for retrieved_session_index in session_id_index[0]]
    #             # print(entry['answer_session_ids'], retrieved_session_ids)
    #             head_wise_recall[head_idx] += any(answer_session_id in retrieved_session_ids for answer_session_id in entry['answer_session_ids'])
    
    # for head_idx in range(model_config.num_attention_heads):
    #     print(head_idx, head_wise_recall[head_idx] / len(multi_session_data))
        
if __name__ == '__main__':
    parser = transformers.HfArgumentParser(
        (MemArguments, DataArguments, TrainingArguments))
    mem_cfg, data_cfg, training_cfg = parser.parse_args_into_dataclasses()
    # main(mem_cfg, data_cfg, training_cfg)
    compute_recall(mem_cfg, data_cfg, training_cfg)