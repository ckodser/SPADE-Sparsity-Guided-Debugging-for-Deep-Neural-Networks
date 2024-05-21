import os
import numpy as np
import sys
# sys.path.append('yolov5')

import torch
import torch.nn as nn
import torchvision

from .resnet_imagenet import resnet18 as resnet18_celeba
from .datautils import set_seed
from .ModifiedResNet import resnet50 as mresnet50
from transformers import ViTModel, ViTForImageClassification
from transformers import DeiTForImageClassification
from transformers import BertModel, BertPreTrainedModel, BertForSequenceClassification, RobertaForSequenceClassification

DEV = torch.device('cuda:0')


def VisualizeImageGrayscale(image_3d, percentile=99):
    r"""Returns a 3D tensor as a grayscale 2D tensor.
    This method sums a 3D tensor across the absolute value of axis=2, and then
    clips values at a given percentile.
    """
    eps = 1e-8
    if len(np.shape(image_3d)) == 3:
        image_2d = np.sum(np.abs(image_3d), axis=0)

        vmax = np.percentile(image_2d, percentile)
        vmin = np.min(image_2d)

        return np.clip((image_2d - vmin) / (vmax - vmin + eps), 0, 1)

    if len(np.shape(image_3d)) == 4:
        ans = np.sum(image_3d, axis=1)
        for i in range(np.shape(image_3d)[0]):
            image_2d = np.sum(np.abs(image_3d[i]), axis=0)

            vmax = np.percentile(image_2d, percentile)
            vmin = np.min(image_2d)

            image_2d = np.clip((image_2d - vmin) / (vmax - vmin + eps), 0, 1)
            ans[i] = image_2d
        return ans


class IntegratedGradientsResNet50:
    """A CoreSaliency class that implements the integrated gradients method.
    https://arxiv.org/abs/1703.01365
    """

    def __init__(self, call_model_function):
        self.call_model_function = call_model_function

    def GetMask(self, x_value, x_baseline=None, x_steps=25, batch_size=1):
        x_value = x_value.numpy()
        if x_baseline is None:
            x_baseline = np.zeros_like(x_value)

        assert x_baseline.shape == x_value.shape

        x_diff = x_value - x_baseline

        total_gradients = np.zeros_like(x_value, dtype=np.float32)

        x_step_batched = []
        for alpha in np.linspace(0, 1, x_steps):
            x_step = x_baseline + alpha * x_diff
            x_step_batched.append(x_step)
            if len(x_step_batched) == batch_size or alpha == 1:
                x_step_batched = np.asarray(x_step_batched)
                call_model_output = self.call_model_function(x_step_batched)
                total_gradients += call_model_output.sum(axis=0)
                x_step_batched = []

        integrated_grad = total_gradients * x_diff / x_steps
        return VisualizeImageGrayscale(integrated_grad)


def attention_mesure(intervention_mask, grad):
    absgrad = torch.sum(grad, dim=0)
    ind = torch.argmax(absgrad)
    row = ind // absgrad.shape[0]
    column = ind % absgrad.shape[0]
    return intervention_mask[row, column]


def find_layers(module, layers=[nn.Conv2d, nn.Linear], name=''):
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_layers(
            child, layers=layers, name=name + '.' + name1 if name != '' else name1
        ))
    return res


def is_depthwise(module):
    return type(module) == nn.Conv2d and module.in_channels == module.groups


def get_logits(model, dataloader, seed):
    with torch.no_grad():
        model.eval()
        dev = next(iter(model.parameters())).device
        preds = []
        ys = []
        set_seed(seed)
        for i, (x, y) in enumerate(dataloader):
            preds.append(model(x.to(dev)).cpu().detach().numpy())
            ys.append(y)  # (y.numpy())
        preds = np.concatenate(preds)
        return preds


def run(model, batch, loss=False, retmoved=False):
    dev = next(iter(model.parameters())).device
    if retmoved:
        return (batch[0].to(dev), batch[1].to(dev))
    out = model(batch[0].to(dev))
    if loss:
        return nn.functional.cross_entropy(out, batch[1].to(dev)).item() * batch[0].shape[0]
    return out



def run_bert(model, batch, loss=False, retmoved=False):
    dev = next(iter(model.parameters())).device
    for k, v in batch.items():
        batch[k] = v.to(DEV)
    if retmoved:
        return batch
    out = model(**batch)
    if loss:
        return out['loss'].item() * batch[k].shape[0]
    return torch.cat([out['start_logits'], out['end_logits']])


def get_run(model):
    if 'bert' in model:
        return run_bert
    return run


from torchvision.models import resnet18, resnet34, resnet50, resnet101


class FakeDataset():
    def __init__(self):
        self.mean = torch.tensor([0, 0, 0])
        self.std = torch.tensor([1, 1, 1])


def get_attacker_rn50():
    train_dataset = FakeDataset()
    ORG_model = mresnet50()
    model = attacker.AttackerModel(ORG_model, train_dataset)
    return model


def get_attacker_rn50_celeba():
    train_dataset = FakeDataset()
    ORG_model = mresnet50()
    ORG_model.fc = torch.nn.Sequential(torch.nn.Linear(2048, 40, bias=True), nn.Sigmoid())
    model = attacker.AttackerModel(ORG_model, train_dataset)
    return model


def get_attacker_rn50_food101():
    train_dataset = FakeDataset()
    ORG_model = mresnet50(pretrained=False)
    ORG_model.fc = torch.nn.Linear(2048, 101, bias=True)
    model = attacker.AttackerModel(ORG_model, train_dataset)
    return model


def get_rn50_celeba():
    ORG_model = mresnet50(pretrained=True)
    ORG_model.fc = torch.nn.Sequential(torch.nn.Linear(2048, 40, bias=True), nn.Sigmoid())
    return ORG_model


def get_rn50_food101():
    ORG_model = mresnet50(pretrained=False)
    ORG_model.fc = torch.nn.Linear(2048, 101, bias=True)
    return ORG_model


class ConvNext_B(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = torchvision.models.convnext_base(weights="IMAGENET1K_V1")

    def forward(self, x, with_latent=False, **kwargs):
        if not with_latent:
            x = self.backbone(x)
            return x
        else:
            raise NotImplementedError


class ConvNext_S(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = torchvision.models.convnext_small(weights="IMAGENET1K_V1")

    def forward(self, x, with_latent=False, **kwargs):
        if not with_latent:
            x = self.backbone(x)
            return x
        else:
            raise NotImplementedError


class ConvNext_T(nn.Module):
    def __init__(self, last_activation=None, num_classes=1000):
        super().__init__()
        self.backbone = torchvision.models.convnext_tiny(weights="IMAGENET1K_V1")
        if last_activation is not None:
            self.backbone.classifier = nn.Sequential(
                self.backbone.classifier[0], self.backbone.classifier[1],
                nn.Linear(in_features=768, out_features=num_classes, bias=True),
                last_activation
            )
        elif num_classes != 1000:
            self.backbone.classifier = nn.Sequential(
                self.backbone.classifier[0], self.backbone.classifier[1],
                nn.Linear(in_features=768, out_features=num_classes, bias=True))

    def forward(self, x, with_latent=False, **kwargs):
        if not with_latent:
            x = self.backbone(x)
            return x
        else:
            raise NotImplementedError


class mobileNetV2(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = torch.hub.load('pytorch/vision:v0.10.0', 'mobilenet_v2', pretrained=True)

    def forward(self, x, with_latent=False, **kwargs):
        if not with_latent:
            x = self.backbone(x)
            return x
        else:
            raise NotImplementedError


class mobileNetV2_celeba(mobileNetV2):
    def __init__(self):
        super().__init__()
        self.backbone.classifier = torch.nn.Sequential(torch.nn.Dropout(p=0.2, inplace=False),
                                                       torch.nn.Linear(1280, 40, bias=True), nn.Sigmoid())


class BertForSentimentClassification(BertPreTrainedModel):
    """ code from https://github.com/MadryLab/DebuggableDeepNetworks/blob/main/language/models.py """

    def __init__(self, config):
        super().__init__(config)
        self.bert = BertModel(config)
        # self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # The classification layer that takes the [CLS] representation and outputs the logit
        self.cls_layer = nn.Linear(config.hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        '''
        Inputs:
            -input_ids : Tensor of shape [B, T] containing token ids of sequences
            -attention_mask : Tensor of shape [B, T] containing attention masks to be used to avoid contibution of PAD tokens
            (where B is the batch size and T is the input length)
        '''
        # Feed the input to Bert model to obtain contextualized representations
        output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        reps, pooler_output = output.last_hidden_state, output.pooler_output
        # Obtain the representations of [CLS] heads
        cls_reps = reps[:, 0]

        # cls_reps = self.dropout(cls_reps)
        logits = self.cls_layer(cls_reps)
        return logits


class BertForToxicClassification(BertPreTrainedModel):
    """ code from https://github.com/MadryLab/DebuggableDeepNetworks/blob/main/language/models.py """

    def __init__(self, config):
        super().__init__(config)
        self.bert = BertModel(config)
        # self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # The classification layer that takes the [CLS] representation and outputs the logit
        self.classifier = nn.Linear(config.hidden_size, 6)

    def set_pooler_output(self, val):
        self.pooled_output = val
        return self

    def forward(self, input_ids, attention_mask):
        '''
        Inputs:
            -input_ids : Tensor of shape [B, T] containing token ids of sequences
            -attention_mask : Tensor of shape [B, T] containing attention masks to be used to avoid contibution of PAD tokens
            (where B is the batch size and T is the input length)
        '''
        # Feed the input to Bert model to obtain contextualized representations
        output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        reps, pooler_output = output.last_hidden_state, output.pooler_output
        # Obtain the representations of [CLS] heads
        if not self.pooled_output:
            cls_reps = reps[:, 0]
        else:
            cls_reps = pooler_output
        # cls_reps = self.dropout(cls_reps)
        logits = self.classifier(cls_reps)
        return logits

    def only_one_part(self, ind):
        new_linear = nn.Linear(self.config.hidden_size, 1)
        new_linear.weight.data = self.classifier.weight.data[ind:ind + 1]
        new_linear.bias.data = self.classifier.bias.data[ind:ind + 1]
        self.classifier = new_linear
        return self


fake_train_dataset = FakeDataset()
get_models = {
    'rn18': lambda: resnet18(pretrained=True),
    'rn18_celeba': lambda: resnet18(pretrained=False, num_classes=40),
    'rn34': lambda: resnet34(pretrained=True),
    'rn50': lambda: mresnet50(pretrained=True),
    'rn50_celeba': get_rn50_celeba,
    'rn50_food101': get_rn50_food101,  #
    'AttackerRN50': get_attacker_rn50,
    'AttackerRN50_celeba': get_attacker_rn50_celeba,
    'AttackerRN50_food101': get_attacker_rn50_food101,  #
    'rn101': lambda: resnet101(pretrained=True),

    "convNext-B": lambda: ConvNext_B(),
    "AttackerConvNext-B": lambda: attacker.AttackerModel(ConvNext_B(), fake_train_dataset),
    "convNext-S": lambda: ConvNext_S(),
    "AttackerConvNext-S": lambda: attacker.AttackerModel(ConvNext_S(), fake_train_dataset),
    "convNext-T": lambda: ConvNext_T(),
    "AttackerConvNext-T": lambda: attacker.AttackerModel(ConvNext_T(), fake_train_dataset),
    "convNext-T-celeba": lambda: ConvNext_T(last_activation=nn.Sigmoid(), num_classes=40),
    "AttackerConvNext-T-celeba": lambda: attacker.AttackerModel(
        ConvNext_T(last_activation=nn.Sigmoid(), num_classes=40), fake_train_dataset),
    "barissayil/bert-sentiment-analysis-sst": lambda: BertForSentimentClassification.from_pretrained(
        "barissayil/bert-sentiment-analysis-sst"),
    'jigsaw-toxic': lambda: BertForToxicClassification.from_pretrained('unitary/toxic-bert').set_pooler_output(
        True).only_one_part(0),
    'jigsaw-severe_toxic': lambda: BertForToxicClassification.from_pretrained('unitary/toxic-bert').set_pooler_output(
        True).only_one_part(1),
    'jigsaw-obscene': lambda: BertForToxicClassification.from_pretrained('unitary/toxic-bert').set_pooler_output(
        True).only_one_part(2),
    'jigsaw-threat': lambda: BertForToxicClassification.from_pretrained('unitary/toxic-bert').set_pooler_output(
        True).only_one_part(3),
    'jigsaw-insult': lambda: BertForToxicClassification.from_pretrained('unitary/toxic-bert').set_pooler_output(
        True).only_one_part(4),
    'jigsaw-identity_hate': lambda: BertForToxicClassification.from_pretrained('unitary/toxic-bert').set_pooler_output(
        True).only_one_part(5),

    'jigsaw-alt-toxic': lambda: BertForToxicClassification.from_pretrained(
        'unitary/unbiased-toxic-roberta').set_pooler_output(False),
    'jigsaw-alt-severe_toxic': lambda: BertForToxicClassification.from_pretrained(
        'unitary/unbiased-toxic-roberta').set_pooler_output(False),
    'jigsaw-alt-obscene': lambda: BertForToxicClassification.from_pretrained(
        'unitary/unbiased-toxic-roberta').set_pooler_output(False),
    'jigsaw-alt-threat': lambda: BertForToxicClassification.from_pretrained(
        'unitary/unbiased-toxic-roberta').set_pooler_output(False),
    'jigsaw-alt-insult': lambda: BertForToxicClassification.from_pretrained(
        'unitary/unbiased-toxic-roberta').set_pooler_output(False),
    'jigsaw-alt-identity_hate': lambda: BertForToxicClassification.from_pretrained(
        'unitary/unbiased-toxic-roberta').set_pooler_output(False),
    'mobileNetv2': lambda: mobileNetV2(),
    'mobileNetv2_celeba': lambda: mobileNetV2_celeba(),
    'AttackerMobileNetv2': lambda: attacker.AttackerModel(mobileNetV2(), fake_train_dataset),
    'AttackerMobileNetv2_celeba': lambda: attacker.AttackerModel(mobileNetV2_celeba(), fake_train_dataset)
}


def get_model(model):
    model = get_models[model]()
    # print(model)
    model = model.to(DEV)
    model.eval()
    return model


def get_functions(model):
    return lambda: get_model(model), None, get_run(model)