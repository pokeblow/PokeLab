from Poke.evaluation.predict import Prediction, PokeResult
import torch
from dataset import RandomClsDataset
from Method.model.model_demo import DemoNet


class TestPrediction(Prediction):
    def configure_model(self):
        self.net = DemoNet(20, 3)

    def configure_results(self):
        self.result_1 = PokeResult(result_name='image_1', result_type='image')
        self.result_2 = PokeResult(result_name='image_2', result_type='image')
        self.result_3 = PokeResult(result_name='value_1', result_type='value')
        self.result_4 = PokeResult(result_name='value_2', result_type='value')

        return self.result_1, self.result_2, self. result_3, self.result_4

    @torch.no_grad()
    def predict_step(self, batch_idx, batch):
        self.net.eval()

        x, y = batch
        x = x.to(self.device, dtype=torch.float32)
        y = y.to(self.device, dtype=torch.long)

        image = torch.randn(10, 3, 224, 224).to(self.device)
        logits = self.net(x)

        self.result_1.set_step(image)
        self.result_2.set_step(image * 0.5)
        self.result_3.set_step(logits)
        self.result_4.set_step(logits + 20)


if __name__ == "__main__":
    valid_set = RandomClsDataset(n=300, seed=1)

    pred = TestPrediction(save_root="results", dataset=valid_set, batch_size=10)
    pred.predict()
