import torch
import torch.nn.functional as torch_funcs

def test_data_eval(model, data_loader, device):
  model = model.eval()
  
  review_texts = []
  predictions = []
  prediction_probs = []
  real_values = []

  with torch.no_grad():
    for d in data_loader:

      texts = d["review_txt"]
      input_ids = d["encoding"].to(device)
      attention_mask = d["attention_mask"].to(device)
      targets = d["label"].to(device)

      outputs = model(
        encodings=input_ids,
        attention_mask=attention_mask
      )
      _, preds = torch.max(outputs, dim=1)

      probs = torch_funcs.softmax(outputs, dim=1)

      review_texts.extend(texts)
      predictions.extend(preds)
      prediction_probs.extend(probs)
      real_values.extend(targets)

  predictions = torch.stack(predictions).cpu()
  prediction_probs = torch.stack(prediction_probs).cpu()
  real_values = torch.stack(real_values).cpu()
  return review_texts, predictions, prediction_probs, real_values

def predict(model, data_loader, device):
  model = model.eval()
  
  review_texts = []
  predictions = []
  prediction_probs = []

  with torch.no_grad():
    for id, d in enumerate(data_loader):

        texts = d["review_txt"]
        input_ids = d["encoding"].to(device)
        attention_mask = d["attention_mask"].to(device)

        outputs = model(
            encodings=input_ids,
            attention_mask=attention_mask
        )
        _, preds = torch.max(outputs, dim=1)

        probs = torch_funcs.softmax(outputs, dim=1)

        review_texts.extend(texts)
        predictions.extend(preds)
        prediction_probs.extend(probs)

  predictions = torch.stack(predictions).cpu()
  prediction_probs = torch.stack(prediction_probs).cpu()
  return review_texts, predictions, prediction_probs

def product_recomendation_score(predictions):
    return (predictions == 1).sum()/predictions.shape[0]