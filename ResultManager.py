class ResultManager(object):
    def __init__(self):
        self.best_auroc_index=0
        self.best_auprc_index=0
        self.aurocs=[]
        self.auprcs=[]
        self.models=[]
        self.x_test_samples=[]
        self.y_test_samples=[]
        self.cnt = 0
        
    def append_result(self, auroc, auprc, x_sample, y_sample, model):
        self.aurocs.append(auroc)
        self.auprcs.append(auprc)
        self.x_test_samples.append(x_sample)
        self.y_test_samples.append(y_sample)
        self.models.append(model)

        best_auroc = self.aurocs[self.best_auroc_index]
        best_auprc = self.auprcs[self.best_auprc_index]
        if auroc > best_auroc:
            self.best_auroc_index = self.cnt
        if auprc > best_auprc:
            self.best_auprc_index = self.cnt
        self.cnt += 1
