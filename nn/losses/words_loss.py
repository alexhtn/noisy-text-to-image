import torch
from torch import nn
from ..attention.function import attention_function


def words_loss(region_features, words_emb, gamma1=4.0, gamma2=5.0, gamma3=10.0):
    """
        words_emb(query): batch x nef x seq_len
        img_features(context): batch x nef x 17 x 17
    """

    labels = torch.arange(region_features.size(0), dtype=torch.long, device=region_features.device)
    att_maps = []
    similarities = []
    batch_size = region_features.size(0)
    words_num = words_emb.size(2)
    for i in range(batch_size):
        # Get the i-th text description
        # -> 1 x nef x words_num
        word = words_emb[i, :, :].unsqueeze(0).contiguous()
        # -> batch_size x nef x words_num
        word = word.repeat(batch_size, 1, 1)
        # batch x nef x 17*17
        context = region_features
        """
            word(query): batch x nef x words_num
            context: batch x nef x 17 x 17
            weighted_context: batch x nef x words_num
            attn: batch x words_num x 17 x 17
        """
        weighted_context, attn = attention_function(word, context, gamma1)
        att_maps.append(attn[i].unsqueeze(0).contiguous())
        # --> batch_size x words_num x nef
        word = word.transpose(1, 2).contiguous()
        weighted_context = weighted_context.transpose(1, 2).contiguous()
        # --> batch_size*words_num x nef
        word = word.view(batch_size * words_num, -1)
        weighted_context = weighted_context.view(batch_size * words_num, -1)
        #
        # -->batch_size*words_num
        row_sim = torch.cosine_similarity(word, weighted_context, dim=1, eps=1e-8)
        # --> batch_size x words_num
        row_sim = row_sim.view(batch_size, words_num)

        # Eq. (10)
        row_sim *= gamma2
        row_sim.exp_()
        row_sim = row_sim.sum(dim=1, keepdim=True)
        row_sim = torch.log(row_sim)

        # --> 1 x batch_size
        # similarities(i, j): the similarity between the i-th image and the j-th text description
        similarities.append(row_sim)

    # batch_size x batch_size
    similarities = torch.cat(similarities, 1)

    similarities = similarities * gamma3
    similarities1 = similarities.transpose(0, 1)
    loss0 = nn.CrossEntropyLoss()(similarities, labels)
    loss1 = nn.CrossEntropyLoss()(similarities1, labels)

    return loss0, loss1, att_maps
