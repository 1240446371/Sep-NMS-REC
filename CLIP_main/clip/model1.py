class CLIP_MODEL(nn.Module):
    def __init__(self, nef, sent_type):
        super(CLIP_MODEL, self).__init__()
        if cfg.TRAIN.FLAG:
            self.nef = nef
        else:
            self.nef = 512  # define a uniform ranker
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = clip.load("../RN101.pt", device=self.device)
        for p in self.model.parameters():
            p.requires_grad = False
        self.model.float()
        self.model.eval()
        self.sent_type = sent_type
        self.MLP = MLP_CLIP(nef=self.nef, sent_type=self.sent_type)

    def image_encode(self, x):
        x = nn.functional.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
        image_code, cnn_feature = self.model.encode_image(x)
        # RN101: image_code size ([batch_size, 512]); cnn_feature size ([batch_size, 1024, 14, 14])
        image_code, cnn_feature = self.MLP.image_feature(image_code, cnn_feature)
        # cnn_feature = cnn_feature.type(torch.FloatTensor).cuda()
        # cnn_feature = self.emb_features(cnn_feature)
        # # image_code = image_code.type(torch.FloatTensor).cuda()
        # image_code = self.emb_cnn_code(image_code)
        return cnn_feature, image_code

    def sent_encode(self, text):
        text_features, word_embs = self.model.encode_text(text)
        word_embs, text_features = self.MLP.sent_feature(word_embs, text_features)
        # # RN101: text_features size ([batch_size, 512]); word_embs size ([batch_size, 512, 77])
        # text_features = text_features.type(torch.FloatTensor).cuda()
        # word_embs = word_embs.type(torch.FloatTensor).cuda()
        # word_embs = word_embs.permute(0, 2, 1)
        # word_embs = self.dense_word(word_embs)
        # word_embs = self.LayerNorm(word_embs)
        # word_embs = word_embs.permute(0, 2, 1)
        # text_features = self.dense_sent(text_features)
        return word_embs, text_features

    def forward(self, image_code, text_features):
        logits_per_i


class MLP_CLIP(nn.Module):
    def __init__(self, nef, sent_type):
        super(MLP_CLIP, self).__init__()
        self.sent_type = sent_type
        self.nef = nef
        self.define_module()
        self.init_trainable_weights()

    def define_module(self):
        self.emb_features = conv1x1(1024, self.nef)

        self.dense_word = nn.Linear(512, self.nef)
        if self.sent_type == 'CLIP_FT':
            self.dense_sent = nn.Linear(512, self.nef)
            self.emb_cnn_code = nn.Linear(512, self.nef)
        self.LayerNorm = torch.nn.LayerNorm(512, eps=1e-12)
        # self.dense_word_2 = nn.Linear(512, self.nef)

    def init_trainable_weights(self):
        initrange = 0.1
        self.emb_features.weight.data.uniform_(-initrange, initrange)
        if self.sent_type == 'CLIP_FT':
            self.emb_cnn_code.weight.data.uniform_(-initrange, initrange)

    def image_feature(self, image_code, cnn_feature):
        cnn_feature = cnn_feature.type(torch.FloatTensor).cuda()
        cnn_feature = self.emb_features(cnn_feature)
        # image_code = image_code.type(torch.FloatTensor).cuda()
        if self.sent_type == 'CLIP_FT':
            image_code = self.emb_cnn_code(image_code)
        return image_code, cnn_feature

    def sent_feature(self, word_embs, text_features):
        text_features = text_features.type(torch.FloatTensor).cuda()
        word_embs = word_embs.type(torch.FloatTensor).cuda()
        word_embs = word_embs.permute(0, 2, 1)
        word_embs = self.dense_word(word_embs)
        word_embs = self.LayerNorm(word_embs)
        # word_embs = self.dense_word_2(word_embs)
        word_embs = word_embs.permute(0, 2, 1)
        if self.sent_type == 'CLIP_FT':
            text_features = self.dense_sent(text_features)
        return word_embs, text_features
        
        