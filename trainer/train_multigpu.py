from transformers import LlamaTokenizer, LlamaConfig, LlamaForCausalLM
from transformers import AdamW, get_scheduler, set_seed

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
os.environ['NCCL_P2P_DISABLE'] = '1'
os.environ['NCCL_IB_DISABLE'] = '1'
os.environ["WANDB_MODE"] = "offline"
from datasets import load_dataset, load_from_disk
from accelerate import Accelerator
import datasets, transformers
from huggingface_hub import Repository
from torch.nn import CrossEntropyLoss
from torch.utils.data import IterableDataset
from torch.utils.data.dataloader import DataLoader
from torch.utils.tensorboard import SummaryWriter
from argparse import Namespace
import torch
import logging
from tqdm import tqdm
import wandb

class ConstantLengthDataset(IterableDataset):

    def __init__(self, tokenizer, dataset, infinite=False, seq_length=1024,
                 num_of_sequences=1024, chars_per_token=3.6):
        self.tokenizer = tokenizer
        self.concat_token_id = tokenizer.bos_token_id
        self.dataset = dataset
        self.seq_length = seq_length
        self.input_characters = seq_length * chars_per_token * num_of_sequences
        self.epoch = 0
        self.infinite = infinite

    def __iter__(self):
        iterator = iter(self.dataset)
        more_examples = True
        while more_examples:
            buffer, buffer_len = [], 0
            while True:
                if buffer_len >= self.input_characters:
                    break
                try:
                    buffer.append(next(iterator)['anchor'])
                    buffer_len += len(buffer[-1])
                except StopIteration:
                    if self.infinite:
                        iterator = iter(self.dataset)
                        self.epoch += 1
                        logger.info(f"Dataset epoch: {self.epoch}")
                    else:
                        more_examples = False
                        break
            tokenized_inputs = tokenizer(buffer, truncation=False)['input_ids']
            all_token_ids = []
            for tokenized_input in tokenized_inputs:
                all_token_ids.extend(tokenized_input + [self.concat_token_id])
            for i in range(0, len(all_token_ids), self.seq_length):
                input_ids = all_token_ids[i: i + self.seq_length]
                if len(input_ids) == self.seq_length:
                    yield torch.tensor(input_ids)
class ConstantLengthDatasetExp(IterableDataset):

    def __init__(self, tokenizer, dataset, infinite=False, seq_length=1024,
                 num_of_sequences=1024, chars_per_token=3.6):
        self.tokenizer = tokenizer
        self.concat_token_id = tokenizer.bos_token_id
        self.dataset = dataset
        self.seq_length = seq_length
        self.input_characters = seq_length * chars_per_token * num_of_sequences
        self.epoch = 0
        self.infinite = infinite

    def __iter__(self):
        iterator = iter(self.dataset)
        more_examples = True

        while more_examples:
            try:
                # 从数据集中获取下一个条目
                item = next(iterator)['anchor']
                # 对该条目进行tokenize
                tokenized_input = self.tokenizer(item, truncation=False)['input_ids']

                # 如果token数大于等于seq_length，截断并返回
                for i in range(0, len(tokenized_input), self.seq_length):
                    input_ids = tokenized_input[i: i + self.seq_length]

                    # 如果token的数量达到了seq_length，返回该部分
                    if len(input_ids) == self.seq_length:
                        yield torch.tensor(input_ids)

            except StopIteration:
                # 当数据集遍历完毕时，判断是否需要无限循环
                if self.infinite:
                    iterator = iter(self.dataset)
                    self.epoch += 1
                    logger.info(f"Dataset epoch: {self.epoch}")
                else:
                    more_examples = False
                    break
def setup_logging(project_name):
    logger = logging.getLogger(__name__)
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S", level=logging.INFO, handlers=[
            logging.FileHandler(f"log/debug_{accelerator.process_index}.log"),
            logging.StreamHandler()])
    if accelerator.is_main_process: # we only want to setup logging once
        wandb.init(project=project_name, config=args)
        run_name = wandb.run.name
        tb_writer = SummaryWriter()
        tb_writer.add_hparams(vars(args), {'0': 0})
        logger.setLevel(logging.INFO)
        datasets.utils.logging.set_verbosity_info()
        transformers.utils.logging.set_verbosity_info()
    else:
        tb_writer = None
        run_name = ''
        logger.setLevel(logging.ERROR)
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
    return logger, tb_writer, run_name

def create_dataloaders(dataset_name, args):
    train_data = load_from_disk(dataset_name)['train']
    valid_data = load_from_disk(dataset_name)['validation']
    train_dataset = ConstantLengthDataset(tokenizer, train_data, infinite=True,
                                          seq_length=args.seq_length)
    valid_dataset = ConstantLengthDataset(tokenizer, valid_data, infinite=False,
                                          seq_length=args.seq_length)
    valid_extrapolate_dataset = ConstantLengthDatasetExp(tokenizer, valid_data, infinite=False,
                                          seq_length=args.seq_length)
    train_dataloader = DataLoader(train_dataset, batch_size=args.train_batch_size)
    eval_dataloader = DataLoader(valid_dataset, batch_size=args.valid_batch_size)
    eval_extrapolate_dataloader = DataLoader(valid_extrapolate_dataset, batch_size=args.valid_batch_size)
    return train_dataloader, eval_dataloader, eval_extrapolate_dataloader

def get_grouped_params(model, args, no_decay=["bias", "LayerNorm.weight"]):
    params_with_wd, params_without_wd = [], []
    for n, p in model.named_parameters():
        if any(nd in n for nd in no_decay): params_without_wd.append(p)
        else: params_with_wd.append(p)
    return [{'params': params_with_wd, 'weight_decay': args.weight_decay},
            {'params': params_without_wd, 'weight_decay': 0.0}]

def log_metrics(step, metrics):
    logger.info(f"Step {step}: {metrics}")
    if accelerator.is_main_process:
        wandb.log(metrics)
        [tb_writer.add_scalar(k, v, step) for k, v in metrics.items()]

def evaluate(args):
    losses = []
    count = 0
    correct = 0
    model.eval()
    for step, batch in enumerate(eval_dataloader):
        with torch.no_grad():
            outputs = model(batch, labels=batch, use_cache=False)

        logits = outputs.logits[:, :-1].contiguous().view(-1, 32017)
        labels = batch[:, 1:].contiguous().view(-1).to(logits.device)
        pred = torch.argmax(logits, dim=-1)
        correct += (pred.squeeze() == labels).tolist().count(True)
        count += logits.size(0)
        loss = outputs.loss.repeat(args.valid_batch_size)
        losses.append(accelerator.gather(loss))
        if args.max_eval_steps > 0 and step >= args.max_eval_steps: break
    loss = torch.mean(torch.cat(losses))
    try: perplexity = torch.exp(loss)
    except OverflowError: perplexity = float("inf")
    return loss.item(), perplexity.item(), correct/count

# Accelerator
accelerator = Accelerator()
acc_state = {str(k): str(v) for k, v in accelerator.state.__dict__.items()}

# Hyperparameters
project_name = 'anchorcoder'
dataset_name = 'anchoreddataset'
config = {"train_batch_size": 8,
          "valid_batch_size": 1,
          "weight_decay": 0.1,
          "shuffle_buffer": 1_000,
          "learning_rate": 5e-4,
          "lr_scheduler_type": "cosine",
          "num_warmup_steps": 200,
          "gradient_accumulation_steps": 1,
          "gradient_checkpointing": False,
          "max_train_steps": 30000,
          "max_eval_steps": 100,
          "seq_length": 512,
          "extrapolate_length": 4096,
          "seed": 42,
          "save_checkpoint_steps": 1000,
          # "save_checkpoint_steps": 100,
          "log_step": 1000}
args = Namespace(**config, **acc_state)
samples_per_step = accelerator.state.num_processes * args.train_batch_size
set_seed(args.seed)

# Logging
logger, tb_writer, run_name = setup_logging(project_name.split("/")[0])
logger.info(accelerator.state)

# Load model and tokenizer
# if accelerator.is_main_process:
#     hf_repo = Repository("./", clone_from=project_name, revision=run_name)

tokenizer = LlamaTokenizer.from_pretrained('codellama-7b')
config = LlamaConfig(
    vocab_size=tokenizer.vocab_size,
    hidden_size=512,
    num_hidden_layers=16,
    num_attention_heads=8,
    intermediate_size=512*4,
    max_position_embeddings=4096,
    attn_implementation='eager',
    anchor_layer=4,
    multilayer=12,
    per=4
)

# 从零初始化LLaMA模型
model = LlamaForCausalLM(config)
model.resize_token_embeddings(len(tokenizer))

print('para:', sum(x.numel() for x in model.parameters()))
# Load dataset and dataloader
train_dataloader, eval_dataloader, eval_extrapolation_dataloader = create_dataloaders(dataset_name, args)

# Prepare the optimizer and learning rate scheduler
optimizer = AdamW(get_grouped_params(model, args), lr=args.learning_rate)
lr_scheduler = get_scheduler(name=args.lr_scheduler_type, optimizer=optimizer,
                             num_warmup_steps=args.num_warmup_steps,
                             num_training_steps=args.max_train_steps)
def get_lr(): return optimizer.param_groups[0]['lr']

# Prepare everything with our `accelerator`.
model, optimizer, train_dataloader, eval_dataloader, eval_extrapolation_dataloader = accelerator.prepare(
    model, optimizer, train_dataloader, eval_dataloader, eval_extrapolation_dataloader)

# Train model
model.train()
completed_steps = 0
for step, batch in enumerate(tqdm(train_dataloader, total=args.max_train_steps, leave=False)):
    outputs = model(batch, labels=batch, use_cache=False)
    # outputs = model(batch, labels=batch, use_cache=False)
    loss = outputs.loss

    loss = loss / args.gradient_accumulation_steps
    accelerator.backward(loss)
    if step % args.gradient_accumulation_steps == 0:
        accelerator.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        completed_steps += 1
    if step % args.save_checkpoint_steps == 0 and step > 0:
        logger.info('Evaluating and saving model checkpoint')
        eval_loss, perplexity, acc = evaluate(args)
        # eval_loss_, perplexity_, acc_ = evaluate_extrapolation(args)
        log_metrics(step, {'lr': get_lr(), 'samples': step * samples_per_step, 'steps': completed_steps,
                           'loss/train': loss.item(),
                           'loss/eval': eval_loss, 'perplexity': perplexity, 'acc': acc,
                           })
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained("./test", save_function=accelerator.save)
        # if accelerator.is_main_process:
        #     hf_repo.push_to_hub(commit_message=f'step {step}')
        model.train()
    if completed_steps >= args.max_train_steps:
        break

# Evaluate and save the last checkpoint
logger.info('Evaluating and saving model after training')
eval_loss, perplexity, acc = evaluate(args)
log_metrics(step,
            {'lr': get_lr(), 'samples': step * samples_per_step, 'steps': completed_steps, 'loss/train': loss.item(),
             'loss/eval': eval_loss,
             'perplexity': perplexity, 'acc': acc})
accelerator.wait_for_everyone()
unwrapped_model = accelerator.unwrap_model(model)
unwrapped_model.save_pretrained("./test", save_function=accelerator.save)
# if accelerator.is_main_process:
#     hf_repo.push_to_hub(commit_message=f'final model')