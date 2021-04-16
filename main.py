import argparse
from model import *
import os
import time
from torch.utils.data import DataLoader
import torch.optim as optim
from constants import *
from helper import *
from dataset import sentenceDataset

#rest-att-true 76.77
#ret -att-false 0.74137
#laptop-att-false 66.03
#laptop-att-tru 0.6809
def main(getWeights = False):
    if getWeights:
        model = IAN(embedding).cuda()
        model.load_state_dict(torch.load(MODEL_PATH))
        model.eval()

    start_time = time.time()
    test_data = get_final_data('test',topic)
    train_data = get_final_data('train',topic)

    embedding = get_embeddings()
    train_dataset = sentenceDataset('dataset_train.npz')
    test_dataset = sentenceDataset('dataset_test.npz')
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    if getWeights:
        model = IAN(embedding).cuda()
        model.load_state_dict(torch.load(MODEL_PATH))
        # model.eval()
        for data in test_loader:
            aspects, contexts, labels = data
            aspects, contexts, labels = aspects.cuda(), contexts.cuda(), labels.cuda()
            outputs = model(aspects, contexts)
            _, predicts = outputs.max(dim=1)
            test_total_cases += labels.shape[0]
            test_correct_cases += (predicts == labels).sum().item()
        test_accuracy = test_correct_cases / test_total_cases
        print(test)

    else:
        model = IAN(embedding).cuda()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=l2_reg)
    max_acc = 0
    for epoch in range(n_epoch):
        train_total_cases = 0
        train_correct_cases = 0
        for data in train_loader:
            aspects, contexts, labels= data
            aspects, contexts, labels = aspects.cuda(), contexts.cuda(), labels.cuda()
            optimizer.zero_grad()
            outputs = model(aspects, contexts)
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
            aspects, contexts, labels = data
            aspects, contexts, labels = aspects.cuda(), contexts.cuda(), labels.cuda()
            outputs = model(aspects, contexts)
            _, predicts = outputs.max(dim=1)
            test_total_cases += labels.shape[0]
            test_correct_cases += (predicts == labels).sum().item()
        test_accuracy = test_correct_cases / test_total_cases
        print('[epoch %03d] train accuracy: %.4f test accuracy: %.4f' % (epoch, train_accuracy, test_accuracy))
        if test_accuracy>max_acc:
            max_acc = max(max_acc, test_accuracy)
            torch.save(model.state_dict(), model_path + model_name)
            print("updated model is saved")
        
    print('max test accuracy:', max_acc)
    end_time = time.time()
    print('Time Costing: %s' % (end_time - start_time))

main()
