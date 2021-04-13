from torch.utils.data import Dataset


class IanDataset(Dataset):

    def __init__(self, path):
        data = np.load(path)
        self.aspects = torch.from_numpy(data['aspects']).long()
        self.contexts = torch.from_numpy(data['contexts']).long()
        self.labels = torch.from_numpy(data['labels']).long()
        self.aspect_lens = torch.from_numpy(data['aspect_lens']).long()
        self.context_lens = torch.from_numpy(data['context_lens']).long()
        self.len = self.labels.shape[0]
        aspect_max_len = self.aspects.size(1)
        context_max_len = self.contexts.size(1)
        self.aspect_mask = torch.zeros(aspect_max_len, aspect_max_len)
        self.context_mask = torch.zeros(context_max_len, context_max_len)
        for i in range(aspect_max_len):
            self.aspect_mask[i, 0:i + 1] = 1
        for i in range(context_max_len):
            self.context_mask[i, 0:i + 1] = 1

    def __getitem__(self, index):
        return self.aspects[index], self.contexts[index], self.labels[index], \
               self.aspect_mask[self.aspect_lens[index] - 1], self.context_mask[self.context_lens[index] - 1]

    def __len__(self):
        return self.len