from src.modelutils import find_layers
import torch
import gc
from lucent.optvis import render, param, transform, objectives
from captum.attr import (
    GradientShap,
    DeepLift,
    DeepLiftShap,
    IntegratedGradients,
    LayerConductance,
    NeuronConductance,
    NoiseTunnel,
    Occlusion,
    NoiseTunnel,
    Saliency,
    GuidedGradCam,
    GuidedBackprop,
    InputXGradient,
    Deconvolution
)
from captum.attr import LRP as CaptumLRP
import src.AttributionEvaluation.attributors as atev
from lime import lime_image
import numpy as np
import math
from src.LRP.Relative_Attributing_Propagation.modules.resnet import resnet50celeba as LRPresnet50celeba
from src.LRP.Relative_Attributing_Propagation.modules.resnet import resnet50 as LRPresnet50
from torch.autograd import Variable
from captum.attr import *

def get_last_conv(model, model_type):
    if is_resnet50(model_type):
        return model.layer4
    if is_mobilenet(model_type) or is_convNext(model_type):
        return model.backbone.features


def is_resnet50(model_type):
    key_words = ["Resnet-50", "rn50", "RN50"]
    for key in key_words:
        if key in model_type:
            return True
    return False


def is_mobilenet(model_type):
    key_words = ["mobile", "Mobile"]
    for key in key_words:
        if key in model_type:
            return True
    return False


def is_convNext(model_type):
    key_words = ["ConvNext"]
    for key in key_words:
        if key in model_type:
            return True
    return False


def is_cnn(model_type):
    key_words = ["Resnet", "rn50", "RN50", "rn18", "resnet", "mobile", "Mobile", "cnn", "CNN", "Conv"]
    for key in key_words:
        if key in model_type:
            return True
    return False


def is_vit(model_type):
    key_words = ["vit", "ViT"]
    for key in key_words:
        if key in model_type:
            return True
    return False


def zero_tensor_grad(image_batch):  # set grad to zero (if possible)
    try:
        image_batch.grad.zero_()
    except:
        pass
    return image_batch


def get_saliency_map(modifiedmodel, image, pc, model_type):
    image_batch = torch.unsqueeze(image, dim=0).cuda()
    image_batch.requires_grad = True
    modifiedmodel.eval()
    modifiedmodel.zero_grad()
    saliency = GuidedGradCam(modifiedmodel, get_last_conv(modifiedmodel, model_type))
    grads = saliency.attribute(image_batch, target=pc)
    del saliency
    return torch.sum(torch.abs(grads[0].detach().clone()), 0).cpu()



def get_attribution_map(modifiedmodel, image, model_type, pc, importance, model_id, approaches_list=None):
    image_batch = torch.unsqueeze(image, dim=0).cuda()
    image_batch.requires_grad = True
    ind = -1
    approaches = []
    modifiedmodel.eval()
    ################################
    modifiedmodel.zero_grad()
    image_batch = zero_tensor_grad(image_batch)

    if approaches_list is None or "IntegratedGradients" in approaches_list:
        ################################
        approaches.append("IntegratedGradients")
        modifiedmodel.zero_grad()
        image_batch = zero_tensor_grad(image_batch)
        ind += 1
        try:
            ig = IntegratedGradients(modifiedmodel)
            grad_mean = ig.attribute(image_batch, torch.zeros_like(image_batch), pc).cpu()
            importance[ind][model_id] = grad_mean[0].detach().clone()
            del grad_mean
            modifiedmodel.zero_grad()
            image_batch = zero_tensor_grad(image_batch)
        except:
            pass

    if approaches_list is None or "Occlusion" in approaches_list:
        ################################
        approaches.append("Occlusion")
        modifiedmodel.zero_grad()
        image_batch = zero_tensor_grad(image_batch)
        ind += 1
        try:
            occlusion = Occlusion(modifiedmodel)
            attributions_occ = occlusion.attribute(image_batch,
                                                   strides=(3, 8, 8),
                                                   target=pc,
                                                   sliding_window_shapes=(3, 15, 15),
                                                   baselines=0).cpu()
            importance[ind][model_id] = attributions_occ[0].detach().clone()
            del attributions_occ
        except:
            pass

    if approaches_list is None or "gradient_shap" in approaches_list:
        ################################
        approaches.append("gradient_shap")
        modifiedmodel.zero_grad()
        image_batch = zero_tensor_grad(image_batch)
        ind += 1
        try:
            gradient_shap = GradientShap(modifiedmodel)
            rand_img_dist = torch.cat([image_batch * 0, image_batch * 1])
            attributions_gs = gradient_shap.attribute(image_batch,
                                                      n_samples=50,
                                                      stdevs=0.0001,
                                                      baselines=rand_img_dist,
                                                      target=pc)
            importance[ind][model_id] = attributions_gs[0].detach().clone()
            del attributions_gs
            del gradient_shap
            del rand_img_dist
        except:
            pass

    if approaches_list is None or "DeepLift" in approaches_list:
        ################################
        approaches.append("DeepLift")
        modifiedmodel.zero_grad()
        image_batch = zero_tensor_grad(image_batch)
        ind += 1
        try:
            dl = DeepLift(modifiedmodel)
            attr_dl = dl.attribute(image_batch, target=pc, baselines=image_batch * 0)
            importance[ind][model_id] = attr_dl[0].detach().clone()
            del attr_dl
            del dl
        except:
            pass

    if approaches_list is None or "saliency" in approaches_list:
        ################################
        approaches.append("saliency")
        modifiedmodel.zero_grad()
        image_batch = zero_tensor_grad(image_batch)
        ind += 1
        try:
            saliency = Saliency(modifiedmodel)
            grads = saliency.attribute(image_batch, target=pc)
            importance[ind][model_id] = grads[0].detach().clone()
            del saliency
            del grads
        except:
            pass

    if approaches_list is None or "GuidedGradCam" in approaches_list:
        ################################
        if is_cnn(model_type):
            approaches.append("GuidedGradCam")
            modifiedmodel.zero_grad()
            image_batch = zero_tensor_grad(image_batch)
            ind += 1
            try:
                saliency = GuidedGradCam(modifiedmodel, get_last_conv(modifiedmodel, model_type))
                grads = saliency.attribute(image_batch, target=pc)
                importance[ind][model_id] = grads[0].detach().clone()
                del saliency
                del grads
            except:
                pass

    if approaches_list is None or "LRP" in approaches_list:
        ################################
        if is_resnet50(model_type):
            approaches.append("LRP")
            modifiedmodel.zero_grad()
            image_batch = zero_tensor_grad(image_batch)
            ind += 1
            try:
                if "celeba" in model_type:
                    LRPm = LRPresnet50celeba()
                elif "food101" in model_type:
                    LRPm = LRPresnet50(num_classes=101)
                else:
                    LRPm = LRPresnet50()

                def translate_key(k):
                    if "fc.0" in k:
                        k = "fc." + k[len("fc.0."):]
                    return k

                LRPm.load_state_dict({translate_key(k): v for k, v in modifiedmodel.state_dict().items()})
                LRPm = LRPm.cuda()

                imageun = torch.unsqueeze(image, dim=0)
                input_LRP = Variable(imageun, volatile=True).cuda()
                input_LRP.requires_grad = True
                output = LRPm(input_LRP)
                T = torch.zeros_like(output)
                T[0, pc] = 1

                grads = LRPm.relprop(R=output * T, alpha=1).sum(dim=1, keepdim=True)
                importance[ind][model_id][0] = grads[0, 0].clone().detach()
            except:
                print("LRP failed")
            try:
                del LRPm
                del imageun
                del input_LRP
                del output
                del T
                del grads
            except:
                print("LRP failed")
                pass


    if approaches_list is None or "GuidedBackprop" in approaches_list:
        ################################
        approaches.append("GuidedBackprop")
        modifiedmodel.zero_grad()
        image_batch = zero_tensor_grad(image_batch)
        ind += 1
        try:
            gbp = GuidedBackprop(modifiedmodel)
            grads = gbp.attribute(image_batch, target=pc)
            importance[ind][model_id] = grads[0].detach().clone()
            del gbp
            del grads
        except:
            pass

    if approaches_list is None or "InputXGradient" in approaches_list:
        ###############################
        approaches.append("InputXGradient")
        modifiedmodel.zero_grad()
        ind += 1
        try:
            ixg = InputXGradient(modifiedmodel)
            grads = ixg.attribute(image_batch, target=pc)
            importance[ind][model_id] = grads[0].detach().clone()
            del ixg
            del grads
        except:
            pass

    if approaches_list is None or "Lime" in approaches_list:
        ###############################
        approaches.append("Lime")
        modifiedmodel.zero_grad()
        ind += 1

        def batch_predict(images):
            batch = torch.tensor(images).permute(0, 3, 1, 2)
            device = "cuda"
            batch = batch.to(device)

            logits = modifiedmodel(batch)
            probs = torch.nn.functional.softmax(logits, dim=1)
            return probs.detach().cpu().numpy()

        try:
            explainer = lime_image.LimeImageExplainer()
            explanation = explainer.explain_instance(np.array(image.permute(1, 2, 0)),
                                                     batch_predict,  # classification function
                                                     top_labels=950,
                                                     hide_color=0,
                                                     num_samples=1000)  # number of images that will be sent to classification function
            ans = torch.zeros((224, 224))
            for k, w in explanation.local_exp[pc]:
                if w > 0:
                    ans[explanation.segments == k] += w
            importance[ind][model_id][0] = ans
        except:
            print("Lime Failed!")

        try:
            del explainer
            del explanation
            del ans
        except:
            pass
        ##########################################3

    return importance, approaches