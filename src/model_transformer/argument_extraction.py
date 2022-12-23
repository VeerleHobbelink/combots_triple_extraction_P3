import glob
import torch
from transformers import AutoTokenizer, AutoModel, AutoConfig
from tqdm import tqdm
from pathlib import Path

# Catch warnings
from transformers import logging
logging.set_verbosity(40)

from src.model_transformer.utils import *


class ArgumentExtraction(torch.nn.Module):    # NOTE: don't hardcode output_dim
    def __init__(self, base_model='albert-base-v2', path=None, output_dim = 161, sep='<eos>'): # You need to change the output_dim to the total number of BIO-tags (including 0,1,2)
        """ Init model with multi-span extraction heads for SPO arguments.

            params:
            str base_model: Transformer architecture to use (default: albert-base-v2)
            str path:       Path to pretrained model
        """
        super().__init__()
        print('loading %s for argument extraction' % base_model)
        self._model = AutoModel.from_pretrained(base_model)
        self._base = base_model
        self._sep = sep

        # Load and extend tokenizer with special SPEAKER tokens
        self._tokenizer = AutoTokenizer.from_pretrained(base_model)
        self._tokenizer.add_tokens(['SPEAKER1', 'SPEAKER2'], special_tokens=True)
        # print(len(self._tokenizer))
        self._model.resize_token_embeddings(len(self._tokenizer))

        # Add token classification heads
        hidden_size = AutoConfig.from_pretrained(base_model).hidden_size
        self._subj_head = torch.nn.Linear(hidden_size, output_dim)
        self._pred_head = torch.nn.Linear(hidden_size, output_dim)
        self._obj_head = torch.nn.Linear(hidden_size, output_dim)
        self._output_dim = output_dim

        self._relu = torch.nn.ReLU()
        self._softmax = torch.nn.Softmax(dim=-1)

        # Set GPU if available
        self._device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.to(self._device)

        # Load model / tokenizer if pretrained model is given
        if path:
            print('\t- Loading pretrained')
            model_path = Path(path + '/argument_extraction_' + base_model + '.zip')
            self.load_state_dict(torch.load(model_path, map_location=self._device))

    def forward(self, input_ids, speaker_ids):
        """ Computes BIO label probabilities for each token
        """
        # Feed dialog through transformer
        y = self._model(input_ids=input_ids, token_type_ids=speaker_ids)
        h = self._relu(y.last_hidden_state)

        # Predict spans
        y_subj = self._softmax(self._subj_head(h))
        y_pred = self._softmax(self._pred_head(h))
        y_obj_ = self._softmax(self._obj_head(h))

        # Permute output as tensor of shape (N, |C|, seq_len)
        y_subj = y_subj.permute(0, 2, 1)
        y_pred = y_pred.permute(0, 2, 1)
        y_obj_ = y_obj_.permute(0, 2, 1)
        # print(y_subj)
        return y_subj, y_pred, y_obj_

    def _retokenize_tokens(self, tokens):
        # Tokenize each token individually (keeping track of subwords)
        input_ids = [[self._tokenizer.cls_token_id]]
        for t in tokens:
            if t != '<eos>':
                input_ids.append(self._tokenizer.encode(t, add_special_tokens=False))
            else:
                input_ids.append([self._tokenizer.eos_token_id])

        # Flatten input_ids
        f_input_ids = torch.LongTensor([[i for ids in input_ids for i in ids]]).to(self._device)

        # Determine how often we need to repeat the labels
        repeats = [len(ids) for ids in input_ids]

        # Set speaker IDs
        speaker_ids = [0] + [tokens[:i + 1].count(self._sep) % 2 for i in range(len(tokens))][:-1]  # TODO: make pretty
        speaker_ids = self._repeat_speaker_ids(speaker_ids, repeats)

        return f_input_ids, speaker_ids, repeats

    def _repeat_speaker_ids(self, speaker_ids, repeats):
        """ Repeats speaker IDs for oov tokens.
        """
        rep_speaker_ids = np.repeat([0] + list(speaker_ids), repeats=repeats)
        return torch.LongTensor([rep_speaker_ids]).to(self._device)

    def _repeat_labels(self, labels, repeats):
        """ Repeats BIO labels for OOV tokens. Ensure B-labeled tokens are repeated
            as B-I-I etc.
        """
        # Repeat each label b the amount of subwords per token
        rep_labels = []
        for label, rep in zip([0] + list(labels), repeats):
            # Outside
            if label == 0:
                rep_labels += [label] * rep
            # Beginning + Inside
            elif label % 2 == 1:    # uneven labels are B-tags
                rep_labels += [label] + ([label+1] * (rep - 1))  # If label = B -> B-I-I-I...
            else: 
                rep_labels += [label] + ([label] * (rep - 1)) # if label = I, do not add 1 but keep the same 
        return torch.LongTensor([rep_labels]).to(self._device)

    def fit(self, tokens, labels, epochs=2, lr=1e-5, weight=800):
        """ Fits the model to the annotations
        """
        # Re-tokenize to obtain input_ids and associated labels
        X = []
        for token_seq, (subj_labels, pred_labels, _obj_labels) in zip(tokens, labels):
            input_ids, speaker_ids, repeats = self._retokenize_tokens(token_seq)
            subj_labels = self._repeat_labels(subj_labels, repeats)  # repeat when split into subwords
            pred_labels = self._repeat_labels(pred_labels, repeats)
            _obj_labels = self._repeat_labels(_obj_labels, repeats)
            X.append((input_ids, speaker_ids, subj_labels, pred_labels, _obj_labels))

        # Set up optimizer
        optim = torch.optim.AdamW(self.parameters(), lr=lr)

        # Higher weight for B- and I-tags to account for class imbalance
        class_weights = torch.Tensor([1] + [weight] * (self._output_dim - 1)).to(self._device)
        criterion = torch.nn.CrossEntropyLoss(weight=class_weights)

        print('Training!')
        for epoch in range(epochs):
            losses = []
            random.shuffle(X)
            for input_ids, speaker_ids, subj_y, pred_y, obj_y in tqdm(X):
                # Forward pass
                subj_y_hat, pred_y_hat, obj_y_hat = self(input_ids, speaker_ids)

                # Compute loss
                loss = criterion(subj_y_hat, subj_y)
                loss += criterion(pred_y_hat, pred_y)
                loss += criterion(obj_y_hat, obj_y)
                losses.append(loss.item())

                optim.zero_grad()
                loss.backward()
                optim.step()

            print("mean loss =", np.mean(losses))

        # Save model to file
        torch.save(self.state_dict(), 'argument_extraction_%s' % self._base)

    def predict(self, token_seq):
        """ Predicts """
        # Retokenize token sequence
        input_ids, speaker_ids, _ = self._retokenize_tokens(token_seq)

        # Invert tokenization for viewing
        subwords = self._tokenizer.convert_ids_to_tokens(input_ids[0])

        # Forward-pass
        predictions = self(input_ids, speaker_ids)
        subjs = predictions[0].cpu().detach().numpy()[0]
        preds = predictions[1].cpu().detach().numpy()[0]
        objs = predictions[2].cpu().detach().numpy()[0]

        return subjs, preds, objs, subwords


if __name__ == '__main__':
    annotations = load_annotations('<path_to_annotation_file')

    # Convert argument annotations to BIO-tag sequences
    tokens, labels = [], []
    for ann in annotations:
        # Map triple arguments to BIO tagged masks
        labels.append((triple_to_bio_tags(ann, 0),
                       triple_to_bio_tags(ann, 1),
                       triple_to_bio_tags(ann, 2)))

        # Flatten turn sequence
        tokens.append([t for ts in ann['tokens'] for t in ts + ['<eos>']])

    # Fit model to data
    model = ArgumentExtraction()
    model.fit(tokens, labels)
    torch.save(model.state_dict(), 'models/argument_extraction_albert-v2_31_03_2022')

