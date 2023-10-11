# Модель генерации парафразов

Этот репозиторий содержит код и инструкции для модели генерации парафразов. Модель предназначена для генерации парафразов для заданных входных текстов, сохраняя их первоначальное значение.

## Цель исследования
Изучение генерации парафраз на русском языке, с акцентом на синтаксические особенности.

## Основной инструмент
Модель *rut5-base-paraphraser* на основе T5-трансформера с интегрированной функцией потерь.

## Обучение 
Модель обучается с использованием специальной функции потерь, которая комбинирует потери от синтаксического сходства, семантического сходства и собственные потери модели.

### Функция потерь
```
def custom_loss(syntactic_similarities, target_syntactic_similarities,
                semantic_similarities, target_semantic_similarities,
                model_outputs, alpha=0.8, beta=0.5, gamma=0.3, device=device):
    # Calculate the syntactic similarity loss
    syntactic_loss = nn.MSELoss()(syntactic_similarities, target_syntactic_similarities.detach()).to(device)

    # Calculate the semantic similarity loss
    semantic_loss = nn.MSELoss()(semantic_similarities, target_semantic_similarities.detach()).to(device)

    # Extract the model's loss from the outputs
    model_loss = model_outputs.loss

    # Combine the losses using weights
    total_loss = alpha * syntactic_loss + beta * semantic_loss + gamma * model_loss

    return total_loss
```

## Валидация
Во время валидации модель генерирует парафразы для валидационного набора. Качество сгенерированных парафразов оценивается с помощью BLEU-оценки.

## Использование
Для того чтобы использовать модель для генерации парафразов, запустите следующий код:
```
from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch

MODEL_NAME = 'output/fine_tuned_model'
model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)
tokenizer_model = 'cointegrated/rut5-base-paraphraser'
tokenizer = T5Tokenizer.from_pretrained(tokenizer_model)
if torch.cuda.is_available():
    model.cuda();

model.eval();

def paraphrase_base(text, beams=5, grams=4, do_sample=False):
    x = tokenizer(text, return_tensors='pt', padding=True).to(model.device)
    max_size = int(x.input_ids.shape[1] * 1.5 + 10)
    out = model.generate(**x, encoder_no_repeat_ngram_size=grams, do_sample=do_sample, num_beams=beams, max_length=max_size, no_repeat_ngram_size=4,)
    return tokenizer.decode(out[0], skip_special_tokens=True)

#@title Парафразер { run: "auto", form-width: "50%", display-mode: "form" }
text = 'Каждый охотник желает знать, где сидит фазан' #@param {type:"string"}
beams = 5 #@param {type:"slider", min:1, max:10, step:1}
grams = 4 #@param {type:"slider", min:1, max:10, step:1}
randomize = True #@param {type:"boolean"}

paraphrase_base(text, beams=beams, grams=grams, do_sample=randomize)
```

## Результаты
- Созданная над моделью надстройка не сильно повлияла на результат.
- Разные уровни соответствия исходным и целевым текстам при генерации парафразов.

## Дальнейшие шаги
- Применение методов регуляризации, аугментации и тонкой настройки параметров.
- Рассмотрение других архитектур, например, синтаксических энкодеров или графовых нейросетей.

## Общий вывод
Учет синтаксических особенностей важен при генерации парафразов, и есть большой потенциал для дальнейших исследований в этой области.
