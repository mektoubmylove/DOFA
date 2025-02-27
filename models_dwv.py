# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

# Ce fichier définit un modèle Vision Transformer (ViT) modifié, appelé OFAViT, 
# qui utilise des poids dynamiques adaptés aux longueurs d'onde.

from functools import partial
from wave_dynamic_layer import Dynamic_MLP_OFA, Dynamic_MLP_Decoder
from operator import mul
from torch.nn.modules.utils import _pair
from torch.nn import Conv2d, Dropout
import numpy as np

import torch
import torch.nn as nn
import pdb
import math
from functools import reduce
import json

from timm.models.vision_transformer import PatchEmbed, Block

class OFAViT(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """
    """ 
    Implémentation d'un Vision Transformer (ViT) avec des poids dynamiques basés sur les longueurs d'onde.
    Utilise `Dynamic_MLP_OFA` pour l'embedding des patches et une architecture Transformer classique.
    """
    def __init__(self, img_size=224, patch_size=16, drop_rate=0.,
                 embed_dim=1024, depth=24, num_heads=16, wv_planes=128, num_classes=45,
                 global_pool=True, mlp_ratio=4., norm_layer=nn.LayerNorm):
        super().__init__()

        self.wv_planes = wv_planes   # Nombre de dimensions pour l'encodage des longueurs d'onde
        self.global_pool = global_pool      # Indique si on utilise un pooling global à la fin du réseau

        # Définition de la normalisation finale
        if self.global_pool:
            norm_layer = norm_layer
            embed_dim = embed_dim
            self.fc_norm = norm_layer(embed_dim)
        else:
            self.norm = norm_layer(embed_dim)
        
        # Embedding des patches avec une MLP dynamique basée sur les longueurs d'onde
        self.patch_embed = Dynamic_MLP_OFA(wv_planes=128, inter_dim=128, kernel_size=16, embed_dim=embed_dim)

        # Nombre de patches obtenus après la découpe de l'image
        self.num_patches = (img_size // patch_size) ** 2

        # Token CLS (appris) utilisé pour la classification
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # Embedding de position basé sur sinusoïdes (fixe, non appris)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim), requires_grad=False)  # fixed sin-cos embedding

        # Définition des blocs Transformer
        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(depth)])

        # Dropout avant la couche de classification
        self.head_drop = nn.Dropout(drop_rate)

        # Couche de classification finale
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()


    def forward_features(self, x, wave_list):
        """
        Passe avant du modèle sans la couche de classification finale.
        x : entrée image (batch, C, H, W)
        wave_list : liste des longueurs d'onde associées
        """
        # embed patches
        wavelist = torch.tensor(wave_list, device=x.device).float() # Conversion de wave_list en tenseur torch (pour le GPU)
        self.waves = wavelist    # Stockage des longueurs d'onde

        x, _ = self.patch_embed(x, self.waves)  # Embedding des patches avec les informations spectrales

        x = x + self.pos_embed[:, 1:, :]  # Ajout de l'embedding de position fixe

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)      # Étendre à toute la batch
        x = torch.cat((cls_tokens, x), dim=1)                 # Concaténer le token CLS

        # apply Transformer blocks
        for block in self.blocks:
            x = block(x)

         # Pooling global si activé
        if self.global_pool:
            x = x[:, 1:, :].mean(dim=1)  # global pool without cls token , Moyenne sur tous les tokens (sans CLS)
            outcome = self.fc_norm(x) # Normalisation finale
        else:
            x = self.norm(x)      # Normalisation
            outcome = x[:, 0]     # On récupère le token CLS pour la classification
        return outcome

    def forward_head(self, x, pre_logits=False):
        """
        Applique la couche de classification finale.
        """
        x = self.head_drop(x)
        return x if pre_logits else self.head(x)

    def forward(self, x, wave_list):
        """
        Passe avant complet du modèle.
        """
        x = self.forward_features(x, wave_list)
        x = self.forward_head(x)
        return x


def vit_small_patch16(**kwargs):
    """ ViT Small avec un embedding de 384 dimensions et 12 blocs. """
    model = OFAViT(
        patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def vit_base_patch16(**kwargs):
    """ ViT Base avec un embedding de 768 dimensions et 12 blocs. """
    model = OFAViT(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_large_patch16(**kwargs):
    """ ViT Large avec un embedding de 1024 dimensions et 24 blocs. """
    model = OFAViT(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_huge_patch14(**kwargs):
    """ ViT Huge avec un embedding de 1280 dimensions et 32 blocs. """
    model = OFAViT(
        patch_size=14, embed_dim=1280, depth=32, num_heads=16, mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model
