from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch
from torch import nn
from torch import optim
import matplotlib.pyplot as plt
import torch_optimizer
import csv
import time
import random

class T5(nn.Module):

    def __init__(self):
        super().__init__()
        t5_type = "t5-base"
        self.tokenizer = T5Tokenizer.from_pretrained(t5_type, model_max_length=512)
        self.lm = T5ForConditionalGeneration.from_pretrained(t5_type)
        self.task_prefix = ""

    def forward(self, inputs, mask, labels):
        if self.training:
            outputs = self.lm(
                input_ids=inputs,
                attention_mask=mask,
                labels=labels,
            )
        else:
            outputs = self.lm.generate(
                input_ids=inputs,
                attention_mask=mask,
                do_sample=False,  # disable sampling to test if batching affects output
            )
        return outputs

    def preproc(self, train_sequences):
        input_sequences, output_sequences = train_sequences
        # Hyperparams for batching
        max_source_length = 512
        max_target_length = 512

        # Encode the original input sequences
        encoding = self.tokenizer(
            [self.task_prefix + sequence for sequence in input_sequences],
            padding="longest",
            max_length=max_source_length,
            truncation=True,
            return_tensors="pt",
        )
        input_ids, attention_mask = encoding.input_ids, encoding.attention_mask

        if output_sequences is None:
            return input_ids, attention_mask, None

        # encode the targets
        target_encoding = self.tokenizer(
            output_sequences,
            padding="longest",
            max_length=max_target_length,
            truncation=True,
            return_tensors="pt",
        )
        labels = target_encoding.input_ids

        # replace padding token id's of the labels by -100 so it's ignored by the loss
        labels[labels == self.tokenizer.pad_token_id] = -100

        return input_ids, attention_mask, labels


def train_t5(task_prefix, train_sequences, dev_sequences, load_path=None):
    model = T5()
    if load_path is not None:
        model.load_state_dict(torch.load(load_path))
    model.task_prefix = task_prefix

    # Preprocess the inputs
    input_ids, attention_mask, labels = model.preproc(train_sequences)
    dev_input_ids, dev_attention_mask, dev_labels = model.preproc(dev_sequences)

    print("Done with preproc")

    model.zero_grad()
    model.train()
    optimizer = torch_optimizer.Adafactor(model.parameters(), lr=3e-4)
    num_epochs = 3

    losses = []
    dev_losses = []
    loss_fcn = nn.CrossEntropyLoss()
    print("Beginning Training")

    ex_idx = input_ids
    batch_size = 50
    print(f"Start Time: 0")
    start_time = time.time()
    for t in range(num_epochs):
        loss_this_epoch = 0.0
        for ex_idx in range(0, len(input_ids), batch_size):
            print(f"Training example {ex_idx}")
            print(f"Start Example: {time.time() - start_time}")
            outputs = model(input_ids[ex_idx: ex_idx + batch_size, :], attention_mask[ex_idx: ex_idx + batch_size, :], labels[ex_idx: ex_idx + batch_size, :]).logits
            # dev_outputs = model(dev_input_ids, dev_attention_mask, dev_labels).logits
            print(f"Compute Logits: {time.time() - start_time}")
            loss = loss_fcn(outputs.view(-1, outputs.size(-1)), labels[ex_idx: ex_idx + batch_size, :].view(-1))
            print(f"Compute Loss: {time.time() - start_time}")
            # dev_loss = loss_fcn(dev_outputs.view(-1, dev_outputs.size(-1)), dev_labels.view(-1))
            '''
            loss = model.lm(input_ids=input_ids, attention_mask=attention_mask, labels=labels).loss
            dev_loss = model.lm(input_ids=dev_input_ids, attention_mask=dev_attention_mask, labels=dev_labels).loss
            '''

            model.zero_grad()
            loss.backward()
            print(f"Compute backwards: {time.time() - start_time}")
            optimizer.step()
            print(f"Compute step: {time.time() - start_time}")
            loss_this_epoch += loss.item()
        print(f"Epoch {t + 1}: Loss = {loss_this_epoch/len(input_ids)*batch_size}")
        losses.append(loss_this_epoch)
        # dev_losses.append(dev_loss.item())
    model.eval()
    torch.save(model.state_dict(), "model_weights")
    return model

def import_examples(filename):
    inputs, labels = [], []
    lines_to_read = 10000
    with open(filename, encoding="utf8") as fd:
        rd = csv.reader((line.replace('\0', '') for line in fd), delimiter="\t")
        i = 0
        for row in rd:
            if len(row) == 2:
                inputs.append(row[0] + " </s>")
                labels.append(row[1] + " </s>")
            if i >= lines_to_read:
                break
            i += 1
        print("Parsed Correctly")
    return inputs, labels


def main():
    # encode the inputs
    task_prefix = "grammar_error_correction: "

    # input_sequences = ["I know answer.", "There are many disease in the world."]
    # output_sequences = ["I know the answer.", "There are many diseases in the world."]
    # sentences = ["I know problem."]
    # golds = ["I know the problem."]
    inputs, outputs = import_examples("../C4_200M01.tsv")

    train_exs = 50
    test_exs = 10
    indices = range(len(inputs))
    random_idx = random.sample(indices, train_exs + test_exs)

    input_sequences = [inputs[i] for i in random_idx[:train_exs]]
    sentences = [inputs[i] for i in random_idx[train_exs:]]
    output_sequences = [outputs[i] for i in random_idx[:train_exs]]
    golds = [outputs[i] for i in random_idx[train_exs:]]
    model = train_t5(task_prefix, (input_sequences, output_sequences), (input_sequences, output_sequences), "model_weights")

    input_ids, attention_mask, _ = model.preproc((sentences, golds))
    pred = model.tokenizer.batch_decode(model(input_ids, attention_mask, None), skip_special_tokens=True)
    for sentence, out, gold in zip(sentences, pred, golds):
        print("Original:  " + sentence)
        print("Predicted: " + out)
        print("Gold:      " + gold)
        print("")

if __name__ == "__main__":
    main()