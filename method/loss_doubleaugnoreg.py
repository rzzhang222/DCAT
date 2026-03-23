import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from core.metrics import accuracy
from core.trainer import SmoothCrossEntropyLoss
from core.trainer import track_bn_stats


def double_loss(criterion_mi, model, x_l, y, x_u, x_s,optimizer,
               step_size=0.003, epsilon=0.031, perturb_steps=10, lamb=1.0, beta=1.0,
               contrast_label='auto', attack='linf-pgd', label_smoothing=0.1,
               wamodel=None, consistency_cost=0., consistency_prop_label=0.,
               ):
    device = y.device
    num_y = y.size(0)
    if x_u is None:
        #print(True)
        x_natural = x_l
        x_u = torch.empty(0,0)
    else:
        x_natural = torch.cat([x_l, x_u], dim=0)

    criterion_ce = SmoothCrossEntropyLoss(reduction='mean', smoothing=label_smoothing)
    criterion_kl = nn.KLDivLoss(reduction='batchmean')

    model.train()
    track_bn_stats(model, False)
    fea_nat, logits_natural = model(x_natural, feats=True)
    p_natural = F.softmax(logits_natural, dim=1).detach()

    if lamb > 0:
        if contrast_label == 'auto':
            _, pseudo_labels = torch.max(logits_natural, 1)
            labels_4_contrast = pseudo_labels.detach()
            mask = None
        elif contrast_label == 'fixed':
            labels_4_contrast = y.detach()
            mask = None
        elif contrast_label:
            labels_4_contrast = None
            mask = None

        x_adv = x_natural.detach() + torch.FloatTensor(x_natural.shape).uniform_(-epsilon, epsilon).to(device).detach()
        x_adv = torch.clamp(x_adv, 0.0, 1.0)
        for _ in range(perturb_steps):
            x_adv.requires_grad_()
            with torch.enable_grad():
                fea_adv, logits_adv = model(x_adv, feats=True)
                loss_atk = criterion_kl(F.log_softmax(logits_adv, dim=1), p_natural)
                if beta > 0:
                    loss_con=-1*F.cosine_similarity(fea_nat,fea_adv).mean()
                    #loss_con = criterion_mi(fea_nat, fea_adv, y=labels_4_contrast, mask=mask)
                    loss = loss_atk + beta * loss_con
                else:
                    loss = loss_atk
            grad = torch.autograd.grad(loss, [x_adv])[0]
            x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
            x_adv = torch.min(torch.max(x_adv, x_natural - epsilon), x_natural + epsilon)
            x_adv = torch.clamp(x_adv, 0.0, 1.0)
        x_adv = Variable(torch.clamp(x_adv, 0.0, 1.0), requires_grad=False)

    model.train()
    track_bn_stats(model, True)
    optimizer.zero_grad()

    fea_nat, logits_natural = model(x_natural, feats=True)
    fea_s,_=model(x_s,feats=True)
    loss_natural_sup = criterion_ce(logits_natural[:num_y], y)
    if consistency_cost > 0 and x_u.size(0) > 0:
        softmax = torch.nn.Softmax(1)
        num_label_4_consist = int(x_u.size(0) * consistency_prop_label)
        prob_s = softmax(logits_natural[num_y - num_label_4_consist:])
        prob_t = softmax(wamodel(x_natural[num_y - num_label_4_consist:])).detach()
        loss_consistency = consistency_cost * torch.mean((prob_s - prob_t) ** 2, dim=[0, 1])
        loss_natural = loss_natural_sup + loss_consistency
    else:
        loss_natural = loss_natural_sup+beta*criterion_mi(fea_nat, fea_nat, y=labels_4_contrast, mask=mask)

    if lamb > 0:
        fea_adv, logits_adv = model(x_adv, feats=True)
        loss_adv = criterion_kl(F.log_softmax(logits_adv, dim=1), F.softmax(logits_natural, dim=1))
        loss_con = torch.tensor(0)
        if beta > 0:
            loss_con=-1*F.cosine_similarity(fea_nat,fea_adv).mean()
            #loss_con = criterion_mi(fea_nat, fea_adv, y=labels_4_contrast, mask=mask)
        loss_robust = loss_adv + beta * loss_con
        batch_metrics = {
            'clean_acc': accuracy(y, logits_natural[:num_y].detach()),
            'adversarial_acc': accuracy(y, logits_adv[:num_y].detach())}
    else:
        loss_robust = torch.tensor(0)
        batch_metrics = {
            'clean_acc': accuracy(y, logits_natural[:num_y].detach())}

    loss1 = loss_natural + lamb * loss_robust
    
    
    x_natural=x_s
    fea_nat,_ = model(x_natural, feats=True)
    #p_natural = F.softmax(logits_natural, dim=1).detach()

    if lamb > 0:
        if contrast_label == 'auto':
            _, pseudo_labels = torch.max(logits_natural, 1)
            labels_4_contrast = pseudo_labels.detach()
            mask = None
        elif contrast_label == 'fixed':
            labels_4_contrast = y.detach()
            mask = None
        elif contrast_label:
            labels_4_contrast = None
            mask = None

        x_adv = x_natural.detach() + torch.FloatTensor(x_natural.shape).uniform_(-epsilon, epsilon).to(device).detach()
        x_adv = torch.clamp(x_adv, 0.0, 1.0)
        for _ in range(perturb_steps):
            x_adv.requires_grad_()
            with torch.enable_grad():
                fea_adv, logits_adv = model(x_adv, feats=True)
                loss_atk = criterion_kl(F.log_softmax(logits_adv, dim=1), p_natural)
                if beta > 0:
                    loss_con=-1*F.cosine_similarity(fea_nat,fea_adv).mean()
                    #loss_con = criterion_mi(fea_nat, fea_adv, y=labels_4_contrast, mask=mask)
                    loss = loss_atk + beta * loss_con
                else:
                    loss = loss_atk
            grad = torch.autograd.grad(loss, [x_adv])[0]
            x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
            x_adv = torch.min(torch.max(x_adv, x_natural - epsilon), x_natural + epsilon)
            x_adv = torch.clamp(x_adv, 0.0, 1.0)
        x_adv = Variable(torch.clamp(x_adv, 0.0, 1.0), requires_grad=False)

    model.train()
    track_bn_stats(model, True)
    optimizer.zero_grad()

    #fea_nat, logits_natural = model(x_natural, feats=True)
    fea_s,_=model(x_s,feats=True)
    loss_natural_sup = criterion_ce(logits_natural[:num_y], y)
    if consistency_cost > 0 and x_u.size(0) > 0:
        softmax = torch.nn.Softmax(1)
        num_label_4_consist = int(x_u.size(0) * consistency_prop_label)
        prob_s = softmax(logits_natural[num_y - num_label_4_consist:])
        prob_t = softmax(wamodel(x_natural[num_y - num_label_4_consist:])).detach()
        loss_consistency = consistency_cost * torch.mean((prob_s - prob_t) ** 2, dim=[0, 1])
        loss_natural = loss_natural_sup + loss_consistency
    else:
        loss_natural = loss_natural_sup+beta*criterion_mi(fea_nat, fea_nat, y=labels_4_contrast, mask=mask)

    if lamb > 0:
        fea_adv, logits_adv = model(x_adv, feats=True)
        loss_adv = criterion_kl(F.log_softmax(logits_adv, dim=1), F.softmax(logits_natural, dim=1))
        loss_con = torch.tensor(0)
        if beta > 0:
            loss_con=-1*F.cosine_similarity(fea_nat,fea_adv).mean()
            #loss_con = criterion_mi(fea_nat, fea_adv, y=labels_4_contrast, mask=mask)
        loss_robust = loss_adv + beta * loss_con
    #     batch_metrics = {
    #         'clean_acc': accuracy(y, logits_natural[:num_y].detach()),
    #         'adversarial_acc': accuracy(y, logits_adv[:num_y].detach())}
    # else:
    #     loss_robust = torch.tensor(0)
    #     batch_metrics = {
    #         'clean_acc': accuracy(y, logits_natural[:num_y].detach())}

    loss = (loss1+loss_natural + lamb * loss_robust)/2.0    
    loss_dict = {
        'loss': loss,
        'nat_loss': loss_natural,
        'rob_loss': loss_robust,
    }

    return loss_dict, batch_metrics

def triple_loss(use_str,criterion_mi, model, x_l, y, x_u, x_s,x_ss,optimizer,
               step_size=0.003, epsilon=0.031, perturb_steps=10, lamb=1.0, beta=1.0,
               contrast_label='auto', attack='linf-pgd', label_smoothing=0.1,
               wamodel=None, consistency_cost=0., consistency_prop_label=0.,
               ):
               
    #my added test code
    #beta=beta/2.0
    #my added test code end
               
    device = y.device
    num_y = y.size(0)
    if x_u is None:
        #print(True)
        x_natural = x_l
        x_u = torch.empty(0,0)
    else:
        x_natural = torch.cat([x_l, x_u], dim=0)

    criterion_ce = SmoothCrossEntropyLoss(reduction='mean', smoothing=label_smoothing)
    criterion_kl = nn.KLDivLoss(reduction='batchmean')

    model.train()
    track_bn_stats(model, False)
    fea_nat, logits_natural = model(x_natural, feats=True)
    p_natural = F.softmax(logits_natural, dim=1).detach()

    if lamb > 0:
        if contrast_label == 'auto':
            _, pseudo_labels = torch.max(logits_natural, 1)
            labels_4_contrast = pseudo_labels.detach()
            mask = None
        elif contrast_label == 'fixed':
            labels_4_contrast = y.detach()
            mask = None
        elif contrast_label:
            labels_4_contrast = None
            mask = None

        x_adv = x_natural.detach() + torch.FloatTensor(x_natural.shape).uniform_(-epsilon, epsilon).to(device).detach()
        x_adv = torch.clamp(x_adv, 0.0, 1.0)
        for _ in range(perturb_steps):
            x_adv.requires_grad_()
            with torch.enable_grad():
                fea_adv, logits_adv = model(x_adv, feats=True)
                loss_atk = criterion_kl(F.log_softmax(logits_adv, dim=1), p_natural)
                if beta > 0:
                    #loss_con=-1*F.cosine_similarity(fea_nat,fea_adv).mean()
                    #loss_con = criterion_mi(fea_nat, fea_adv, y=labels_4_contrast, mask=mask)
                    loss_con=criterion_mi(fea_nat, fea_adv, y=labels_4_contrast, mask=mask)
                    loss = loss_atk + beta * loss_con
                else:
                    loss = loss_atk
            grad = torch.autograd.grad(loss, [x_adv])[0]
            x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
            x_adv = torch.min(torch.max(x_adv, x_natural - epsilon), x_natural + epsilon)
            x_adv = torch.clamp(x_adv, 0.0, 1.0)
        x_adv = Variable(torch.clamp(x_adv, 0.0, 1.0), requires_grad=False)

    model.train()
    track_bn_stats(model, True)
    optimizer.zero_grad()

    fea_nat, logits_natural = model(x_natural, feats=True)
    #fea_s,_=model(x_s,feats=True)
    loss_natural_sup = criterion_ce(logits_natural[:num_y], y)
    if consistency_cost > 0 and x_u.size(0) > 0:
        softmax = torch.nn.Softmax(1)
        num_label_4_consist = int(x_u.size(0) * consistency_prop_label)
        prob_s = softmax(logits_natural[num_y - num_label_4_consist:])
        prob_t = softmax(wamodel(x_natural[num_y - num_label_4_consist:])).detach()
        loss_consistency = consistency_cost * torch.mean((prob_s - prob_t) ** 2, dim=[0, 1])
        loss_natural = loss_natural_sup + loss_consistency
    else:
        loss_natural = loss_natural_sup+beta*criterion_mi(fea_nat, fea_nat, y=labels_4_contrast, mask=mask)

    if lamb > 0:
        fea_adv, logits_adv = model(x_adv, feats=True)
        loss_adv = criterion_kl(F.log_softmax(logits_adv, dim=1), F.softmax(logits_natural, dim=1))
        loss_con = torch.tensor(0)
        if beta > 0:
            #loss_con=-1*F.cosine_similarity(fea_nat,fea_adv).mean()
            #loss_con = criterion_mi(fea_nat, fea_adv, y=labels_4_contrast, mask=mask)
            loss_con=criterion_mi(fea_nat, fea_adv, y=labels_4_contrast, mask=mask)
        loss_robust = loss_adv + beta * loss_con
        batch_metrics = {
            'clean_acc': accuracy(y, logits_natural[:num_y].detach()),
            'adversarial_acc': accuracy(y, logits_adv[:num_y].detach())}
    else:
        loss_robust = torch.tensor(0)
        batch_metrics = {
            'clean_acc': accuracy(y, logits_natural[:num_y].detach())}

    loss1 = loss_natural + lamb * loss_robust
    
    #this part comment, less strong aug start
#    x_natural=x_s
#    fea_nat, logits_natural = model(x_natural, feats=True)
#    p_natural = F.softmax(logits_natural, dim=1).detach()
#
#    if lamb > 0:
#        if contrast_label == 'auto':
##            _, pseudo_labels = torch.max(logits_natural, 1)
##            labels_4_contrast = pseudo_labels.detach()
#            mask = None
#        elif contrast_label == 'fixed':
#            labels_4_contrast = y.detach()
#            mask = None
#        elif contrast_label:
#            labels_4_contrast = None
#            mask = None
#
#        x_adv = x_natural.detach() + torch.FloatTensor(x_natural.shape).uniform_(-epsilon, epsilon).to(device).detach()
#        x_adv = torch.clamp(x_adv, 0.0, 1.0)
#        for _ in range(perturb_steps):
#            x_adv.requires_grad_()
#            with torch.enable_grad():
#                fea_adv, logits_adv = model(x_adv, feats=True)
#                loss_atk = criterion_kl(F.log_softmax(logits_adv, dim=1), p_natural)
#                if beta > 0:
#                    #loss_con=-1*F.cosine_similarity(fea_nat,fea_adv).mean()
#                    #loss_con = criterion_mi(fea_nat, fea_adv, y=labels_4_contrast, mask=mask)
#                    loss_con=criterion_mi(fea_nat, fea_adv, y=labels_4_contrast, mask=mask)
#                    loss = loss_atk + beta * loss_con
#                else:
#                    loss = loss_atk
#            grad = torch.autograd.grad(loss, [x_adv])[0]
#            x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
#            x_adv = torch.min(torch.max(x_adv, x_natural - epsilon), x_natural + epsilon)
#            x_adv = torch.clamp(x_adv, 0.0, 1.0)
#        x_adv = Variable(torch.clamp(x_adv, 0.0, 1.0), requires_grad=False)
#
#    model.train()
#    track_bn_stats(model, True)
#    optimizer.zero_grad()
#
#    #fea_nat, logits_natural = model(x_natural, feats=True)
#    #fea_s,_=model(x_s,feats=True)
#    loss_natural_sup = criterion_ce(logits_natural[:num_y], y)
#    if consistency_cost > 0 and x_u.size(0) > 0:
#        softmax = torch.nn.Softmax(1)
#        num_label_4_consist = int(x_u.size(0) * consistency_prop_label)
#        prob_s = softmax(logits_natural[num_y - num_label_4_consist:])
#        prob_t = softmax(wamodel(x_natural[num_y - num_label_4_consist:])).detach()
#        loss_consistency = consistency_cost * torch.mean((prob_s - prob_t) ** 2, dim=[0, 1])
#        loss_natural = loss_natural_sup + loss_consistency
#    else:
#        loss_natural = loss_natural_sup+beta*criterion_mi(fea_nat, fea_nat, y=labels_4_contrast, mask=mask)
#
#    if lamb > 0:
#        fea_adv, logits_adv = model(x_adv, feats=True)
#        loss_adv = criterion_kl(F.log_softmax(logits_adv, dim=1), F.softmax(logits_natural, dim=1))
#        loss_con = torch.tensor(0)
#        if beta > 0:
#            #loss_con=-1*F.cosine_similarity(fea_nat,fea_adv).mean()
#            #loss_con = criterion_mi(fea_nat, fea_adv, y=labels_4_contrast, mask=mask)
#            loss_con=criterion_mi(fea_nat, fea_adv, y=labels_4_contrast, mask=mask)
#        loss_robust = loss_adv + beta * loss_con
#    #     batch_metrics = {
#    #         'clean_acc': accuracy(y, logits_natural[:num_y].detach()),
#    #         'adversarial_acc': accuracy(y, logits_adv[:num_y].detach())}
#    # else:
#    #     loss_robust = torch.tensor(0)
#    #     batch_metrics = {
#    #         'clean_acc': accuracy(y, logits_natural[:num_y].detach())}
#    #logits_adv0=logits_adv.copy()
#    #fea_adv0=fea_adv.copy()
#    loss2 = loss1+(loss_natural + lamb * loss_robust)*0.5
    
    
    
    #this part comment, less strong aug end
    #use_str_here=True
    if use_str: 
     x_natural=x_ss
    #else: 
    #   x_natural=x_s   
     fea_nat, logits_natural = model(x_natural, feats=True)
     p_natural = F.softmax(logits_natural, dim=1).detach()

     if lamb > 0:
        if contrast_label == 'auto':
#            _, pseudo_labels = torch.max(logits_natural, 1)
#            labels_4_contrast = pseudo_labels.detach()
            mask = None
        elif contrast_label == 'fixed':
            labels_4_contrast = y.detach()
            mask = None
        elif contrast_label:
            labels_4_contrast = None
            mask = None

        x_adv2 = x_natural.detach() + torch.FloatTensor(x_natural.shape).uniform_(-epsilon, epsilon).to(device).detach()
        x_adv2 = torch.clamp(x_adv2, 0.0, 1.0)
        for _ in range(perturb_steps):
            x_adv2.requires_grad_()
            with torch.enable_grad():
                fea_adv2, logits_adv2 = model(x_adv2, feats=True)
                loss_atk = criterion_kl(F.log_softmax(logits_adv2, dim=1), p_natural)
                if beta > 0:
                    #loss_con=-1*F.cosine_similarity(fea_nat,fea_adv2).mean()
                    #loss_con = criterion_mi(fea_nat, fea_adv, y=labels_4_contrast, mask=mask)
                    loss_con=criterion_mi(fea_nat, fea_adv, y=labels_4_contrast, mask=mask)
                    loss = loss_atk + beta * loss_con
                else:
                    loss = loss_atk
            grad = torch.autograd.grad(loss, [x_adv2])[0]
            x_adv2 = x_adv2.detach() + step_size * torch.sign(grad.detach())
            x_adv2 = torch.min(torch.max(x_adv2, x_natural - epsilon), x_natural + epsilon)
            x_adv2 = torch.clamp(x_adv2, 0.0, 1.0)
        x_adv2 = Variable(torch.clamp(x_adv2, 0.0, 1.0), requires_grad=False)

     model.train()
     track_bn_stats(model, True)
     optimizer.zero_grad()

    #fea_nat, logits_natural = model(x_natural, feats=True)
    #fea_s,_=model(x_s,feats=True)
     loss_natural_sup = criterion_ce(logits_natural[:num_y], y)
     if consistency_cost > 0 and x_u.size(0) > 0:
        softmax = torch.nn.Softmax(1)
        num_label_4_consist = int(x_u.size(0) * consistency_prop_label)
        prob_s = softmax(logits_natural[num_y - num_label_4_consist:])
        prob_t = softmax(wamodel(x_natural[num_y - num_label_4_consist:])).detach()
        loss_consistency = consistency_cost * torch.mean((prob_s - prob_t) ** 2, dim=[0, 1])
        loss_natural = loss_natural_sup + loss_consistency
     else:
        loss_natural = loss_natural_sup+beta*criterion_mi(fea_nat, fea_nat, y=labels_4_contrast, mask=mask)

     if lamb > 0:
        fea_adv2, logits_adv2 = model(x_adv2, feats=True)
        loss_adv = criterion_kl(F.log_softmax(logits_adv2, dim=1), F.softmax(logits_natural, dim=1))
        loss_con = torch.tensor(0)
        if beta > 0:
            #loss_con=-1*F.cosine_similarity(fea_nat,fea_adv2).mean()
            #loss_con = criterion_mi(fea_nat, fea_adv, y=labels_4_contrast, mask=mask)
            loss_con=criterion_mi(fea_nat, fea_adv, y=labels_4_contrast, mask=mask)
        loss_robust = loss_adv + beta * loss_con
    #     batch_metrics = {
    #         'clean_acc': accuracy(y, logits_natural[:num_y].detach()),
    #         'adversarial_acc': accuracy(y, logits_adv[:num_y].detach())}
    # else:
    #     loss_robust = torch.tensor(0)
    #     batch_metrics = {
    #         'clean_acc': accuracy(y, logits_natural[:num_y].detach())}

     loss2 =(loss1+(loss_natural + lamb * loss_robust)*1.0)/2.0
     loss3=loss2+0.2*criterion_kl(F.log_softmax(logits_adv2,dim=1),F.softmax(logits_adv,dim=1))
     loss=loss3
    
    
    else:
       loss=loss1
    #since i am doing ablation, comment the line after this
     #loss=loss+0.1*(criterion_kl(F.log_softmax(logits_adv2,dim=1),F.softmax(logits_adv,dim=1))+beta*criterion_mi(fea_adv2,fea_adv,y=labels_4_contrast, mask=mask))
    #loss=loss3    
#    fea_nat, logits_natural = model(x_natural, feats=True)
#    p_natural = F.softmax(logits_natural, dim=1).detach()
#
#    if lamb > 0:
#        if contrast_label == 'auto':
#            _, pseudo_labels = torch.max(logits_natural, 1)
#            labels_4_contrast = pseudo_labels.detach()
#            mask = None
#        elif contrast_label == 'fixed':
#            labels_4_contrast = y.detach()
#            mask = None
#        elif contrast_label:
#            labels_4_contrast = None
#            mask = None
#
#        x_adv = x_natural.detach() + torch.FloatTensor(x_natural.shape).uniform_(-epsilon, epsilon).to(device).detach()
#        x_adv = torch.clamp(x_adv, 0.0, 1.0)
#        for _ in range(perturb_steps):
#            x_adv.requires_grad_()
#            with torch.enable_grad():
#                fea_adv, logits_adv = model(x_adv, feats=True)
#                #loss_atk = criterion_kl(F.log_softmax(logits_adv, dim=1), p_natural)
#                if beta > 0:
#                    loss_con=-1*F.cosine_similarity(fea_nat,fea_adv).mean()
#                    #loss_con = criterion_mi(fea_nat, fea_adv, y=labels_4_contrast, mask=mask)
#                    loss = loss_con
#                else:
#                    loss = loss_atk
#            grad = torch.autograd.grad(loss, [x_adv])[0]
#            x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
#            x_adv = torch.min(torch.max(x_adv, x_natural - epsilon), x_natural + epsilon)
#            x_adv = torch.clamp(x_adv, 0.0, 1.0)
#        x_adv = Variable(torch.clamp(x_adv, 0.0, 1.0), requires_grad=False)
#
#    model.train()
#    track_bn_stats(model, True)
#    optimizer.zero_grad()
#
#    #fea_nat, logits_natural = model(x_natural, feats=True)
#    #fea_s,_=model(x_s,feats=True)
#    #loss_natural_sup = criterion_ce(logits_natural[:num_y], y)
#    if consistency_cost > 0 and x_u.size(0) > 0:
#        softmax = torch.nn.Softmax(1)
#        num_label_4_consist = int(x_u.size(0) * consistency_prop_label)
#        prob_s = softmax(logits_natural[num_y - num_label_4_consist:])
#        prob_t = softmax(wamodel(x_natural[num_y - num_label_4_consist:])).detach()
#        loss_consistency = consistency_cost * torch.mean((prob_s - prob_t) ** 2, dim=[0, 1])
#        loss_natural = loss_natural_sup + loss_consistency
#    else:
#        loss_natural = criterion_mi(fea_nat, fea_nat, y=labels_4_contrast, mask=mask)
#
#    if lamb > 0:
#        fea_adv, logits_adv = model(x_adv, feats=True)
#        #loss_adv = criterion_kl(F.log_softmax(logits_adv, dim=1), F.softmax(logits_natural, dim=1))
#        loss_con = torch.tensor(0)
#        if beta > 0:
#            loss_con=-1*F.cosine_similarity(fea_nat,fea_adv).mean()
#            #loss_con = criterion_mi(fea_nat, fea_adv, y=labels_4_contrast, mask=mask)
#        loss_robust = loss_con
#    #     batch_metrics = {
#    #         'clean_acc': accuracy(y, logits_natural[:num_y].detach()),
#    #         'adversarial_acc': accuracy(y, logits_adv[:num_y].detach())}
#    # else:
#    #     loss_robust = torch.tensor(0)
#    #     batch_metrics = {
#    #         'clean_acc': accuracy(y, logits_natural[:num_y].detach())}

    #loss = loss2+0.1*(loss_natural + lamb* loss_robust)
    loss_dict = {
        'loss': loss,
        'nat_loss': loss_natural,
        'rob_loss': loss_robust,
    }

    return loss_dict, batch_metrics

