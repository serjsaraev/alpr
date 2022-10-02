import torch
import numpy as np
from tqdm import tqdm
import pandas as pd

from custom_metrics import get_iou
from sklearn.metrics import auc


class ModelTrainer:

    def __init__(self, model, model_name, train_loader, val_loader, optimizer, device, scheduller=None,
                 conf_thresh=0.5, val_classes=None):

        self.model = model
        self.model_name = model_name
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduller = scheduller

        self.conf_thresh = conf_thresh
        self.val_classes = val_classes

    def fit_epoch(self):

        obj_loss = []
        reg_loss = []
        global_loss = []

        self.model.train()
        for idx, batch in tqdm(enumerate(self.train_loader)):
            images = [b[0].to(self.device) for b in batch]
            targets = [{k: v.to(self.device) for k, v in b[1].items()} for b in batch]

            self.optimizer.zero_grad()
            try:
                losses = self.model(images, targets)
            except:
                print('Error in training step')
                continue

            loss = sum([v for v in losses.values()])
            loss.backward()

            self.optimizer.step()
            if self.scheduller is not None:
                self.scheduller.step()

            global_loss.append(loss.item())
            obj_loss.append(losses['loss_objectness'].item())
            reg_loss.append(losses['loss_rpn_box_reg'].item())

            if idx % 20 == 0:
                print('Global Loss:', np.mean(global_loss))
                print('Object Loss:', np.mean(obj_loss))
                print('Reg Loss:', np.mean(reg_loss))

        return np.mean(global_loss)

    def eval_sample(self, pred, target, iou_thresh, result_dict):
        pred_bboxes = pred['boxes'].cpu().numpy()
        scores = pred['scores'].cpu().numpy()
        pred_labels = pred['labels'].cpu().numpy()

        true_bboxes = target['boxes'].cpu().numpy()
        true_labels = target['labels'].cpu().numpy()
        image_id = target['image_id'].cpu().numpy()[0]

        for i in range(len(pred_bboxes)):

            pred_box = pred_bboxes[i]
            pred_label = pred_labels[i]

            result_dict['image_id'].append(image_id)
            result_dict['pred_label'].append(pred_label)
            result_dict['score'].append(scores[i])

            max_iou = 0
            max_id = -1
            for j in range(len(true_bboxes)):

                true_box = true_bboxes[j]
                true_label = true_labels[j]

                if true_label != pred_label:
                    continue

                IoU = get_iou(true_box, pred_box)
                if (IoU >= iou_thresh) and (IoU > max_iou):
                    max_iou = IoU
                    max_id = j

            if max_id >= 0:
                result_dict['TP'].append(1)
                true_bboxes = np.delete(true_bboxes, max_id, axis=0)
                true_labels = np.delete(true_labels, max_id, axis=0)

            else:
                result_dict['TP'].append(0)

        return result_dict

    def compute_metrics(self, predictions, targets):

        iou_list_results = {str(round(iou, 2)): {class_id: {'precision_list': [], 'recall_list': []}
                                                 for class_id in self.val_classes}
                            for iou in np.linspace(0.5, 0.95, 10)}
        result_metrics = {'map_list': [], 'map@50': 0, 'map@50_95': 0, 'precsion': 0, 'recall': 0}

        for iou in tqdm(iou_list_results):
            result_dict = {'image_id': [], 'pred_label': [], 'score': [], 'TP': []}
            n_boxes = {label: 0 for label in self.val_classes}

            for pred, true in zip(predictions, targets):
                result_dict = self.eval_sample(pred, true, float(iou), result_dict)
                for label in true['labels']:
                    n_boxes[label.item()] += 1

            df = pd.DataFrame(result_dict)
            iou_list_results[iou]['ap_list'] = []

            if iou == '0.5':
                iou_list_results[iou]['precision_list'] = []
                iou_list_results[iou]['recall_list'] = []

            for class_id in self.val_classes:
                df_per_class = df[df['pred_label'] == class_id]
                df_per_class = df_per_class.sort_values(by='score', ascending=False)

                TP = 0
                preds_count = 0
                for idx, row in df_per_class.iterrows():
                    TP += row['TP']
                    preds_count += 1

                    precision = TP / preds_count
                    recall = TP / n_boxes[class_id]

                    iou_list_results[iou][class_id]['precision_list'].append(precision)
                    iou_list_results[iou][class_id]['recall_list'].append(recall)

                    if (iou == '0.5') and (row['score'] >= self.conf_thresh - 0.02) and (
                            row['score'] <= self.conf_thresh + 0.02):
                        iou_list_results[iou][class_id]['precision'] = precision
                        iou_list_results[iou][class_id]['recall'] = recall

                iou_list_results[iou][class_id]['AP'] = auc(iou_list_results[iou][class_id]['recall_list'],
                                                            iou_list_results[iou][class_id]['precision_list'])

                iou_list_results[iou]['ap_list'].append(iou_list_results[iou][class_id]['AP'])

                if iou == '0.5':
                    try:
                        iou_list_results[iou]['precision_list'].append(iou_list_results[iou][class_id]['precision'])
                        iou_list_results[iou]['recall_list'].append(iou_list_results[iou][class_id]['recall'])
                    except:
                        iou_list_results[iou]['precision_list'].append(0)
                        iou_list_results[iou]['recall_list'].append(0)

        for iou in iou_list_results:
            if iou == '0.5':
                result_metrics['precision'] = np.mean(iou_list_results[iou]['precision_list'])
                result_metrics['recall'] = np.mean(iou_list_results[iou]['recall_list'])

            result_metrics['map_list'].append(np.mean(iou_list_results[iou]['ap_list']))

        result_metrics['map@50'] = result_metrics['map_list'][0]
        result_metrics['map@50_95'] = np.mean(result_metrics['map_list'])

        return result_metrics

    @torch.no_grad()
    def eval_epoch(self):

        self.model.eval()

        predictions = []
        answers = []
        for idx, batch in tqdm(enumerate(self.val_loader)):
            images = [b[0].to(self.device) for b in batch]
            targets = [{k: v.to(self.device) for k, v in b[1].items()} for b in batch]

            outputs = self.model(images)
            predictions.extend(outputs)
            answers.extend(targets)
        print(predictions)
        result = self.compute_metrics(predictions, answers)

        return result

    def train_net(self, num_epochs):

        best_map = 0
        for epoch in range(1, num_epochs + 1):

            train_loss = self.fit_epoch()
            result_metrics = self.eval_epoch()

            if result_metrics['map@50_95'] >= best_map:
                best_map = result_metrics['map@50_95']
                torch.save(self.model, f'DetectionModels/{self.model_name}_{epoch}.pth')

            with open(f'Logs/{self.model_name}.txt', 'a') as f:
                string = f"Precision={result_metrics['precision']} Recall={result_metrics['recall']} MAP50={result_metrics['map@50']} MAP50_95={result_metrics['map@50_95']}\n"
                f.write(string)

            print('Epoch:', epoch)
            print('Precsion:', result_metrics['precision'])
            print('Recall:', result_metrics['recall'])
            print('MAP@50:', result_metrics['map@50'])
            print('MAP@50_95:', result_metrics['map@50_95'])