from typing import Text, Dict, List, Type, Any, Optional
import os, logging

import numpy as np

import torch
from torch.utils.data import Dataset

from joblib import dump, load

from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from transformers import AutoConfig
from transformers import Trainer, TrainingArguments

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

from rasa.nlu.classifiers.classifier import IntentClassifier
from rasa.engine.recipes.default_recipe import DefaultV1Recipe
from rasa.engine.graph import ExecutionContext, GraphComponent
from rasa.engine.storage.resource import Resource
from rasa.shared.nlu.training_data.training_data import TrainingData
from rasa.shared.nlu.training_data.message import Message
from rasa.engine.storage.storage import ModelStorage

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class CustomDataset(Dataset):
    """
    Dataset for training the model.
    """

    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]).squeeze() for key, val in self.encodings.items()}
        item["label"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

    def __len__(self):
        return len(self.labels)


def compute_metrics(pred):
    """
    Helper function to compute aggregated metrics from predictions.
    """
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="weighted"
    )
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}

@DefaultV1Recipe.register(
    DefaultV1Recipe.ComponentType.INTENT_CLASSIFIER, is_trainable=True
)
class TransformerClassifier(IntentClassifier, GraphComponent):
    name = "transformer_classifier"
    provides = ["intent"]
    requires = ["text"]
    model_name = "albert-base-v2"

    @classmethod
    def required_components(cls) -> List[Type]:
        return []

    @staticmethod
    def get_default_config() -> Dict[Text, Any]:
        return {
            'epochs': 15,
            'batch_size': 24,
            'warmup_steps': 500,
            'weight_decay': 0.01,
            'learning_rate': 2e-5,
            'scheduler_type': 'constant',
            'max_length': 64
        }

    @staticmethod
    def supported_languages() -> Optional[List[Text]]:
        """Determines which languages this component can work with.

        Returns: A list of supported languages, or `None` to signify all are supported.
        """
        return None

    @classmethod
    def create(
        cls,
        config: Dict[Text, Any],
        model_storage: ModelStorage,
        resource: Resource,
        execution_context: ExecutionContext,
    ) -> GraphComponent:
        return cls(config, execution_context.node_name, model_storage, resource)

    def __init__(
        self,
        config: Dict[Text, Any],
        name: Text,
        model_storage: ModelStorage,
        resource: Resource,
    ) -> None:
        self.name = name
        self.label2id = {}
        self.id2label = {}
        self._define_model() 
        
        # We need to use these later when saving the trained component.
        self._model_storage = model_storage
        self._resource = resource

    def _define_model(self):
        """
        Loads the pretrained model and the configuration after the data has been preprocessed.
        """
        self.config = AutoConfig.from_pretrained(self.model_name)
        self.config.id2label = self.id2label
        self.config.label2id = self.label2id
        self.config.num_labels = len(self.id2label)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name, config=self.config
        )

    def _compute_label_mapping(self, labels):
        """
        Maps the labels to integers and stores them in the class attributes.
        """

        label_encoder = LabelEncoder()
        integer_encoded = label_encoder.fit_transform(labels)
        self.label2id = {}
        self.id2label = {}
        for label in np.unique(labels):
            self.label2id[label] = int(label_encoder.transform([label])[0])
        for i in integer_encoded:
            self.id2label[int(i)] = label_encoder.inverse_transform([i])[0]

    def _preprocess_data(self, data, params):
        """
        Preprocesses the data to be used for training.
        """

        documents = []
        labels = []
        for message in data.training_examples:
            if "text" in message.data:
                documents.append(message.data["text"])
                labels.append(message.data["intent"])
        self._compute_label_mapping(labels)
        targets = [self.label2id[label] for label in labels]
        encodings = self.tokenizer(
            documents,
            padding="max_length",
            max_length=params.get("max_length", 64),
            truncation=True,
        )
        dataset = CustomDataset(encodings, targets)

        return dataset

    def train(self, training_data: TrainingData) -> TrainingData:
        """
        Preprocesses the data, loads the model, configures the training and trains the model.
        """

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        component_config = self.get_default_config()
        dataset = self._preprocess_data(training_data, component_config)
        self._define_model() # apparently this has to be executed here, otherwise the training process will fail
                             # with the input/label shape matching error (why? I have no idea....)

        training_args = TrainingArguments(
            output_dir="./custom_model",
            num_train_epochs=component_config.get("epochs", 15),
            evaluation_strategy="no",
            per_device_train_batch_size=component_config.get("batch_size", 24),
            warmup_steps=component_config.get("warmup_steps", 500),
            weight_decay=component_config.get("weight_decay", 0.01),
            learning_rate=component_config.get("learning_rate", 2e-5),
            lr_scheduler_type=component_config.get("scheduler_type", "constant"),
            save_strategy="no",
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=dataset,
            compute_metrics=compute_metrics,
        )

        trainer.train()

        self.persist()

        return self._resource

    def _process_intent_ranking(self, outputs):
        """
        Processes the intent ranking, sort in descending order based on confidence. Get only top 10

        Args:
            outputs: model outputs

        Returns:
            intent_ranking (list) - list of dicts with intent name and confidence (top 10 only)
        """

        confidences = [float(x) for x in outputs["logits"][0]]
        intent_names = list(self.label2id.keys())
        intent_ranking_all = zip(confidences, intent_names)
        intent_ranking_all_sorted = sorted(
            intent_ranking_all, key=lambda x: x[0], reverse=True
        )
        intent_ranking = [
            {"confidence": x[0], "intent": x[1]} for x in intent_ranking_all_sorted[:10]
        ]
        return intent_ranking

    def _predict(self, text):
        """
        Predicts the intent of the input text.

        Args:
            text (str): input text

        Returns:
            prediction (string) - intent name
            confidence (float) - confidence of the intent
            intent_ranking (list) - list of dicts with intent name and confidence (top 10 only)
        """
        component_config = self.get_default_config()

        inputs = self.tokenizer(
            text,
            padding="max_length",
            max_length=component_config.get("max_length", 64),
            truncation=True,
            return_tensors="pt",
        ).to(DEVICE)

        outputs = self.model(**inputs)

        confidence = float(outputs["logits"][0].max())
        prediction = self.id2label[int(outputs["logits"][0].argmax())]
        intent_ranking = self._process_intent_ranking(outputs)

        return prediction, confidence, intent_ranking

    def process(self, messages: List[Message]) -> List[Message]:
        """
        Processes the input given from Rasa. Attaches the output to the message object.

        Args:
            message (Message): input message
        """

        for message in messages:
            text = message.data["text"]
            prediction, confidence, intent_ranking = self._predict(text)

            message.set(
                "intent", {"name": prediction, "confidence": confidence}, add_to_output=True
            )
            message.set("intent_ranking", intent_ranking, add_to_output=True)
        
        return messages
        
    def process_training_data(self, training_data):
        self.process(training_data.training_examples)
        return training_data

    def persist(self) -> None:
        with self._model_storage.write_to(self._resource) as model_dir:
            tokenizer_filename = "tokenizer_{}".format(self.name)
            model_filename = "model_{}".format(self.name)
            config_filename = "config_{}".format(self.name)
            tokenizer_path = os.path.join(model_dir, tokenizer_filename)
            model_path = os.path.join(model_dir, model_filename)
            config_path = os.path.join(model_dir, config_filename)
            self.tokenizer.save_pretrained(tokenizer_path)
            self.model.save_pretrained(model_path)
            self.config.save_pretrained(config_path)

    # @classmethod
    # def load(
    #     cls, meta, model_dir=None, model_metadata=None, cached_component=None, **kwargs
    # ):
    #     """
    #     Loads the model, tokenizer and configuration from the given path.

    #     Returns:
    #         component (Component): loaded component
    #     """

    #     tokenizer_filename = meta.get("tokenizer")
    #     model_filename = meta.get("model")
    #     config_filename = meta.get("config")
    #     tokenizer_path = os.path.join(model_dir, tokenizer_filename)
    #     model_path = os.path.join(model_dir, model_filename)
    #     config_path = os.path.join(model_dir, config_filename)

    #     x = cls(meta)
    #     x.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    #     x.config = AutoConfig.from_pretrained(config_path)
    #     x.id2label = x.config.id2label
    #     x.label2id = x.config.label2id
    #     x.model = AutoModelForSequenceClassification.from_pretrained(
    #         model_path, config=x.config
    #     ).to(DEVICE)

    #     return x

    @classmethod
    def load(
        cls,
        config: Dict[Text, Any],
        model_storage: ModelStorage,
        resource: Resource,
        execution_context: ExecutionContext,
    ) -> GraphComponent:
        with model_storage.read_from(resource) as model_dir:
            component = cls(
                config, execution_context.node_name, model_storage, resource
            )

            tokenizer_filename = config["tokenizer"]
            model_filename = config["model"]
            config_filename = config["config"]
            tokenizer_path = os.path.join(model_dir, tokenizer_filename)
            model_path = os.path.join(model_dir, model_filename)
            config_path = os.path.join(model_dir, config_filename)

            component.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
            component.config = AutoConfig.from_pretrained(config_path)
            component.id2label = component.config.id2label
            component.label2id = component.config.label2id
            component.model = AutoModelForSequenceClassification.from_pretrained(
                model_path, config=component.config
            ).to(DEVICE)

            return component
