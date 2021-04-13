import argparse
from model import *
import os
import time
from torch.utils.data import DataLoader
import torch.optim as optim
from constants import *
from helper import *
from dataset import IanDataset


def main():
    start_time = time.time()
    test_data = get_final_data('test')
    train_data = get_final_data('train')

    embedding = load_word_embeddings()
    train_dataset = IanDataset('dataset_train.npz')
    test_dataset = IanDataset('dataset_test.npz')
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    model = IAN(embedding).cuda()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=l2_reg)
    max_acc = 0
    for epoch in range(n_epoch):
        train_total_cases = 0
        train_correct_cases = 0
        for data in train_loader:
            aspects, contexts, labels, aspect_masks, context_masks = data
            aspects, contexts, labels = aspects.cuda(), contexts.cuda(), labels.cuda()
            aspect_masks, context_masks = aspect_masks.cuda(), context_masks.cuda()
            optimizer.zero_grad()
            outputs = model(aspects, contexts, aspect_masks, context_masks)
            _, predicts = outputs.max(dim=1)
            train_total_cases += labels.shape[0]
            train_correct_cases += (predicts == labels).sum().item()
            loss = criterion(outputs, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()
        train_accuracy = train_correct_cases / train_total_cases
        test_total_cases = 0
        test_correct_cases = 0
        for data in test_loader:
            aspects, contexts, labels, aspect_masks, context_masks = data
            aspects, contexts, labels = aspects.cuda(), contexts.cuda(), labels.cuda()
            aspect_masks, context_masks = aspect_masks.cuda(), context_masks.cuda()
            outputs = model(aspects, contexts, aspect_masks, context_masks)
            _, predicts = outputs.max(dim=1)
            test_total_cases += labels.shape[0]
            test_correct_cases += (predicts == labels).sum().item()
        test_accuracy = test_correct_cases / test_total_cases
        print('[epoch %03d] train accuracy: %.4f test accuracy: %.4f' % (epoch, train_accuracy, test_accuracy))
        max_acc = max(max_acc, test_accuracy)
    print('max test accuracy:', max_acc)
    end_time = time.time()
    print('Time Costing: %s' % (end_time - start_time))

if __name__ == '__main__':
    main()
