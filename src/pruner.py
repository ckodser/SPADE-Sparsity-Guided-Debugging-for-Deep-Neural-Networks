import torch
import numpy as np

from .datautils import *
from .modelutils import *
from .trueobs import *
from .sparseGPT import SparseGPT
from .checkpoints import load_checkpoint

from torchvision.utils import save_image
from pathlib import Path
from tqdm import tqdm
from src.modelutils import BertForSentimentClassification
from src.TransformerExplainability.BERT_explainability.modules.layers_ours import Linear as TransformerLinear
import hashlib
import copy
import matplotlib.pyplot as plt

DATASET_PATH = {"imagenet": "/home/Datasets/ILSVRC/Data/CLS-LOC",
                "celeba": "/localhome/asoltani/celeba",
                "food101": "/localhome/asoltani/food101"}


def get_hash_array(array):
    # Convert the array of floats into a string
    str_array = ''.join(str(e) for e in array)

    # Create a new SHA256 hash object
    hash_object = hashlib.sha256()

    # Update the hash object with the bytes of the string
    hash_object.update(str_array.encode())

    # Get the hexadecimal representation of the hash
    hex_dig = hash_object.hexdigest()

    return hex_dig


def get_hash_image(image):
    if isinstance(image, tuple):
        return image[1]
    if isinstance(image, list):
        hash_arr = []
        for im in image:
            hash_arr.append(np.sum(np.array(im.getdata())))
        return get_hash_array(hash_arr)
    else:
        return np.sum(np.array(image.getdata()))


def get_hash_model(model):
    return torch.sum(list(find_layers(model).items())[-1][1].weight.data).item()


def get_existed_model(run_dir, modelp):
    saved_model = torch.load(f"{run_dir}/model.pth")
    modelp.load_state_dict(saved_model.state_dict())
    return modelp


def model_exist(run_dir):
    return Path(f"{run_dir}/model.pth").is_file()


def obs_core(train_data, modelp, layer_filter, run_dir, test, sparsity, sparsity_function=None, dont_train_gen=False,
             always_damp=0):
    if sparsity_function is None:
        sparsity_function = lambda x: [sparsity]
        yield_mode = False
    else:
        yield_mode = True

    # find layers we want to change
    layersp = find_layers(modelp, [TransformerLinear, nn.Conv2d, nn.Linear])
    layersp = {k: v for k, v in layersp.items() if layer_filter(k)}

    if yield_mode:
        first_layer_name = list(layersp.keys())[0]
        sparsity_count = len(sparsity_function(first_layer_name))
        if Path(f"{run_dir}/model_0.pth").is_file() and Path(f"{run_dir}/model_{sparsity_count - 1}.pth").is_file():
            print("load", run_dir)
            for i in range(sparsity_count):
                saved_model = torch.load(f"{run_dir}/model_{i}.pth")
                modelp.load_state_dict(saved_model.state_dict())
                yield modelp
            saved_model = []
            return
        if dont_train_gen:
            raise ValueError
    else:
        if model_exist(run_dir):
            print("load", run_dir)
            return get_existed_model(run_dir, modelp)

    # build dataloaders
    workers = 8
    dataloader = DataLoader(
        train_data, batch_size=15, num_workers=workers, pin_memory=True, shuffle=True)
    testloader = DataLoader(
        train_data, batch_size=15, num_workers=workers, pin_memory=True, shuffle=False)

    print(f"layersp={layersp}")

    # make TrueOBS objects over each layer
    trueobs = {}
    save_weights = {}
    for name in layersp:
        layer = layersp[name]
        trueobs[name] = TrueOBS(layer, os.path.join(run_dir, "TrueOBS_hessian_log.txt"), always_damp=always_damp)

    # this function save baches for puring as a forward hook
    def add_batch(name):
        def tmp(layer, inp, out):
            trueobs[name].add_batch(inp[0].data, out.data)

        return tmp

    # add forward hooks
    handles = []
    for name in trueobs:
        handles.append(layersp[name].register_forward_hook(add_batch(name)))

    dev = next(iter(modelp.parameters())).device
    for j, batch in enumerate(dataloader):
        # print(j)
        with torch.no_grad():
            if isinstance(modelp, BertForSentimentClassification) or isinstance(modelp, BertForToxicClassification):
                text, mask, label = batch
                for name in trueobs:
                    trueobs[name].set_mask(mask.to(dev))
                modelp(text.to(dev), mask.to(dev))

            else:
                run(modelp, batch)

    for h in handles:
        h.remove()
    print(f"run_dir:{run_dir}")
    for name in trueobs:
        print(name)
        if ("conv" in name or "downsample.0" in name or isinstance(trueobs[name].layer, nn.Conv2d)) and (
                "unstructure" not in run_dir):
            print(f'{name}, Block pruning ...  size={trueobs[name].kernel_size}')
            trueobs[name].prepare_blocked(size=trueobs[name].kernel_size)
            print("prepare blocked Done")
            W = trueobs[name].prune_blocked(np.array(sparsity_function(name)))

        elif "value" in name or "fc" in name or "classifier" in name or "dense" in name or "pwff" in name or "attn" in name or "proj" in name or "unstructure" in run_dir or isinstance(
                trueobs[name].layer, nn.Linear):
            print(f'{name}, Unstructured pruning ...')
            trueobs[name].prepare_unstr()
            W = trueobs[name].prune_unstr(np.array(sparsity_function(name)))

        elif "key" in name or "query" in name:
            print(f'{name}, Block pruning ...  size={trueobs[name].columns}')
            trueobs[name].prepare_blocked(size=trueobs[name].columns)
            print("prepare blocked Done")
            W = trueobs[name].prune_blocked(np.array(sparsity_function(name)))
        else:
            print(f"{name} puring approach is not defined")
            raise ValueError

        if yield_mode:
            save_weights[name] = copy.deepcopy(W)
        else:
            layer = layersp[name]
            layer.weight.data = W[0].reshape(layer.weight.shape).to(layer.weight.device)
        trueobs[name].free()
        trueobs[name] = None
        gc.collect()
        torch.cuda.empty_cache()

    if yield_mode:
        for i in range(sparsity_count):
            for name in trueobs:
                layer = layersp[name]
                layer.weight.data = save_weights[name][i].reshape(layer.weight.shape).to(layer.weight.device)
            torch.save(modelp, f"{run_dir}/model_{i}.pth")
            yield modelp
        return
    else:
        torch.save(modelp, f"{run_dir}/model.pth")
        return modelp


def prune_one_layer(gpt, sparsity, name, percdamp, run_dir):
    if ("conv" in name or "downsample.0" in name or isinstance(gpt.layer, nn.Conv2d)) and (
            "unstructure" not in run_dir):
        W = gpt.layer.weight.data
        print(f'{name}, Block pruning ...  size={W.shape[2] * W.shape[3]}')
        gpt.fasterprune(
            sparsity, prunen=0, prunem=0, percdamp=percdamp, blocksize=W.shape[2] * W.shape[3]
        )
    elif "value" in name or "fc" in name or "classifier" in name or "dense" in name or "pwff" in name or "attn" in name or "proj" in name or "unstructure" in run_dir or isinstance(
            gpt.layer, nn.Linear) or "key" in name or "query" in name:
        print(f'{name}, Unstructured pruning ...')
        gpt.fasterprune(
            sparsity, prunen=0, prunem=0, percdamp=percdamp, blocksize=1
        )
    else:
        print(f"{name} puring approach is not defined")
        raise ValueError
    weight = gpt.layer.weight.data.clone()
    gpt.free()
    gpt = None
    return weight


def sparseGPT_core(train_data, modelp, layer_filter, run_dir, test, sparsity, pruning_groups=[], sparsity_function=None,
                   dont_train_gen=False, percdamp=0, load_and_save=True, nlp_dataset=False, tokenizer=None, keys=None):
    assert len(pruning_groups) > 0
    assert (tokenizer is not None) or (not nlp_dataset)
    assert (keys is not None) or (not nlp_dataset)

    if sparsity_function is None:
        if nlp_dataset:
            sparsity_function = lambda x: sparsity
        else:
            sparsity_function = lambda x: [sparsity]

    # find layers we want to change
    layersp = find_layers(modelp, [TransformerLinear, nn.Conv2d, nn.Linear])
    layersp = {k: v for k, v in layersp.items() if layer_filter(k)}

    if load_and_save and model_exist(run_dir):
        print("load", run_dir)
        return get_existed_model(run_dir, modelp)

    # build dataloaders
    workers = 16
    dataloader = DataLoader(
        train_data, batch_size=256, num_workers=workers, pin_memory=True, shuffle=True)
    testloader = DataLoader(
        train_data, batch_size=256, num_workers=workers, pin_memory=True, shuffle=False)

    print(f"layersp={layersp}")
    for prune_group_name in pruning_groups:
        print(f"prune group {prune_group_name} ...")

        gpts = {}
        for name in layersp:
            if prune_group_name in name:
                gpts[name] = SparseGPT(layersp[name])

        # this function save baches for puring as a forward hook
        def add_batch(name):
            def tmp(layer, inp, out):
                gpts[name].add_batch(inp[0].data, out.data)
                if len(dataloader) == 0:
                    print("Prune WithIn forward (for faster execution, currently disabled by if condition. ")
                    print(f"prune {name}...")
                    if ("conv" in name or "downsample.0" in name or isinstance(gpts[name].layer, nn.Conv2d)) and (
                            "unstructure" not in run_dir):
                        W = gpts[name].layer.weight.data
                        print(f'{name}, Block pruning ...  size={W.shape[2] * W.shape[3]}')
                        gpts[name].fasterprune(
                            sparsity_function(name), prunen=0, prunem=0, percdamp=percdamp,
                            blocksize=W.shape[2] * W.shape[3]
                        )

                    elif "value" in name or "fc" in name or "classifier" in name or "dense" in name or "pwff" in name or "attn" in name or "proj" in name or "unstructure" in run_dir or isinstance(
                            gpts[name].layer, nn.Linear) or "key" in name or "query" in name:
                        print(f'{name}, Unstructured pruning ...')
                        gpts[name].fasterprune(
                            sparsity_function(name), prunen=0, prunem=0, percdamp=percdamp, blocksize=1
                        )
                    else:
                        print(f"{name} puring approach is not defined")
                        raise ValueError

                    gpts[name].free()
                    gpts[name] = None
                    gc.collect()
                    torch.cuda.empty_cache()

            return tmp

        # add forward hooks
        handles = []
        for name in gpts:
            handles.append(layersp[name].register_forward_hook(add_batch(name)))

        dev = next(iter(modelp.parameters())).device
        for j, batch in enumerate(dataloader):
            # print(j)
            with torch.no_grad():
                if nlp_dataset:
                    texts = (
                        (batch[keys[0]],) if keys[1] is None else (batch[keys[0]], batch[keys[1]])
                    )
                    tokenized_output = tokenizer(*texts, padding=True, max_length=512, truncation=True,
                                                 return_tensors="pt")
                    tokenized_output = {k: v.cuda() for k, v in tokenized_output.items()}
                    modelp(**tokenized_output)

                else:
                    input_x = batch[0].cuda()
                    modelp(input_x)

        for h in handles:
            h.remove()

        for name in gpts:
            prune_one_layer(gpts[name], sparsity_function(name), name, percdamp, run_dir)

    if load_and_save:
        torch.save(modelp, f"{run_dir}/model.pth")
    return modelp


def make_obs_model(dataset_name, model_name, checkpoint_path, given_example_id, checkpoint_name, sparsity, seed, jitter,
                   gauss, local, random_crop, same_identity, spurious_obj, nsamples, output_path, purining_name,
                   layer_filter, modelp=None, single_image=None, dont_train=False, dont_train_gen=False,
                   normalize_single_image=True, random_remove=False, initial_center_crop=False, model_hash=None,
                   sparsity_function=None, always_damp=0, celeba_normalize=False, percdamp=0, pruning_groups=[],
                   load_and_save=True):  # retrain_sparse_model=False):

    # get real dataset
    dataset_path = DATASET_PATH[dataset_name]
    print(dataset_path)

    if not (local and seed == -1 and single_image is not None):
        _, test_data = get_dataset(dataset_name, dataset_path, noaug=False, testaug=True, jitter=jitter, gauss=gauss,
                                   normalize=not local, random_crop=random_crop, spurious_obj=spurious_obj)

        # get model
    if modelp is None:
        get_model, test, run = get_functions(model_name)
        checkpoint_path = checkpoint_path
        modelp = get_model()
        modelp = load_checkpoint(checkpoint_path, modelp, checkpoint_name)

        # build training dataset
    if local:
        if seed == -1:
            if single_image is None:
                example_id = [given_example_id]
                test_data = repeated_with_augs_given_id(test_data, nsamples, example_id, same_identity=same_identity)
            else:
                example_id = [
                    f"image_{get_hash_image(single_image)}_Model_{get_hash_model(modelp) if model_hash is None else model_hash}"]
                test_data = SingleImageDataset(single_image, nsamples, normalize=normalize_single_image, gauss=gauss,
                                               jitter=jitter, random_crop=random_crop, random_remove=random_remove,
                                               initial_center_crop=initial_center_crop,
                                               celeba_normalize=celeba_normalize)
        else:
            example_id, test_data = repeated_with_augs(test_data, nsamples, seed, same_identity=same_identity)
        print(f"example_id:{example_id}")

        run_dir = f'{output_path}/tmp/OBS/{dataset_name}/{checkpoint_name}{purining_name}_{"gauss" if gauss else ""}_{"jitter" if jitter else ""}{"local" if local else ""}{"_same_identity" if same_identity else ""}{"_no_random_crop" if not random_crop else ""}{"_" + spurious_obj if spurious_obj is not None else ""}{"_randomRemove" if random_remove else ""}{"_initial_center_crop" if initial_center_crop else ""}/{example_id[0]}_{sparsity}'

        print("run dir:", run_dir, model_exist(run_dir))
        if (model_exist(run_dir) or dont_train) and load_and_save:
            return example_id, get_existed_model(run_dir, modelp)

        train_data = test_data
    else:
        run_dir = f'{output_path}/tmp/OBS/{dataset_name}/{checkpoint_name}{purining_name}_{"gauss" if gauss else ""}_{"jitter" if jitter else ""}{"local" if local else ""}{"_no_random_crop" if not random_crop else ""}{"_" + spurious_obj if spurious_obj is not None else ""}{"_initial_center_crop" if initial_center_crop else ""}/total_{sparsity}'  # {"_retrain" if retrain_sparse_model else ""}

        train_data, _ = get_dataset(dataset_name, dataset_path, noaug=False, testaug=True, jitter=jitter, gauss=gauss,
                                    normalize=not local, random_crop=random_crop, spurious_obj=spurious_obj)

        train_data = select_sub_dataset(train_data, nsamples, seed)
        example_id = None
        print("run dir:", run_dir, model_exist(run_dir))
        if (model_exist(run_dir) or dont_train) and load_and_save:
            return None, get_existed_model(run_dir, modelp)

    os.makedirs('{}'.format(run_dir), exist_ok=True)

    # get test function
    test = get_test_function(dataset_name)

    if "sparseGPT" in run_dir:
        modelp = sparseGPT_core(train_data, modelp, layer_filter, run_dir, test, sparsity=sparsity,
                                pruning_groups=pruning_groups, sparsity_function=sparsity_function, percdamp=percdamp,
                                load_and_save=load_and_save)
        return example_id, modelp

    if sparsity_function is None:
        gen = obs_core(train_data, modelp, layer_filter, run_dir, test, sparsity, sparsity_function,
                       always_damp=always_damp)
        for m in gen:
            pass
        print("NON GEN PATH (single model return)")
        return example_id, modelp
    else:
        gen = obs_core(train_data, modelp, layer_filter, run_dir, test, sparsity, sparsity_function, dont_train_gen,
                       always_damp=always_damp)
        print("GEN PATH (generator return)")
        return example_id, gen


def make_obs_model_nlp(dataset_name, model_name, checkpoint_name, sparsity, seed, nsamples, output_path, purining_name,
                       layer_filter, modelp, single_sentence, dont_train=False, strong_aug=False):
    example_id = [f"image_{single_sentence[:10]}_Model_{get_hash_model(modelp)}"]
    test_data = SingleSentence(single_sentence, nsamples, return_sentences=False, strong_aug=strong_aug)

    run_dir = f'{output_path}/tmp/OBS/{dataset_name}/{checkpoint_name}{purining_name}/{example_id[0]}{sparsity}'
    os.makedirs('{}'.format(run_dir), exist_ok=True)
    print(run_dir)

    gen = obs_core(test_data, modelp, layer_filter, run_dir, None, sparsity)
    for m in gen:
        pass
    print("NON GEN PATH (single model return)")
    return example_id, modelp
