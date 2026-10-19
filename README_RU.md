# PCMN Text Core v0.6 — экспериментальная текстовая модель и проверяемое обучение

**Состояние на 15.10.2026:** исходники, совместимые checkpoint'ы v0.3–v0.6, обучение и тесты. Это **не** конкурентная LLM: знания, кодинг, tool use, длинный контекст в смысле понимания пока недостаточны. Сравнение с современными LLM по качеству **не выполнено**.

## Что нового

- `pcmn/quality_eval.py`: проход по **всем** токенам независимых validation-документов, с сохранением состояния внутри документа и сбросом между документами; средняя NLL взвешена по токенам. Случайные обучающие окна не выдаются за полную контрольную выборку.
- `pcmn/corpus_audit.py` и `scripts/audit_corpus_v06.py`: обнаружение полного совпадения текстов между train/valid, проверки пересечения источников и *приблизительный* поиск почти-дубликатов по словарным шинглам на ограниченной выборке.
- `scripts/curate_corpus.py --split-unit source`: опциональное разделение по `source`, чтобы один источник не попадал одновременно в train и valid. По умолчанию прежний режим `document` (сохранена совместимость старых вызовов).
- `scripts/train_streaming.py`: `best.pt` по наименьшей validation NLL, `--patience` / `--min-improvement` для ранней остановки, сохранение этих полей в checkpoint. `last.pt` также сохраняется. Поддержка возобновления старых v0.5 checkpoints.
- `pcmn/grounding.py`: явная JSONL-разметка источников извлечённых фрагментов, их экранирование; только дополнительный *инструмент поиска*, не гарантия соблюдения ссылок моделью. Опция `--grounded` в HTTP сервере.
- `scripts/hardware_preflight_v06.py`: проверяет реальные доступные CPU/CUDA и архитектуру GPU вместо предположений, что P100 уже протестирована.
- Расширенная регрессия: **53/53 теста** на CPU данной среды.

## Установка в Windows 11

Распакуйте архив в каталог, например `D:\AI\PCMN_text_v0.6`, затем в PowerShell:

```powershell
cd D:\AI\PCMN_text_v0.6
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python -m scripts.hardware_preflight_v06 --out benchmarks\hardware_local.json
```

**Важно:** `torch` из `requirements.txt` может установиться в CPU-сборке. Для работы на GPU используйте официальную инструкцию PyTorch по установке конкретной версии CUDA. Сборка должна поддерживать вашу карту: Tesla P100 — `sm_60`, RTX 4060 — `sm_89`. FP16 на P100 поддерживается аппаратно; BF16 нет. Никаких результатов CUDA на P100 эта версия проекта не заявляет.

В Linux: `python3 -m venv .venv && source .venv/bin/activate`.

## Повторить реально проведённый запуск CPU

```powershell
python -m scripts.evaluate_exhaustive_v06 --checkpoint runs/v05_mini_demo/last.pt --corpus data/technical_bin --context 96 --threads 2
python -m scripts.train_streaming --corpus data/technical_bin --resume runs/v05_mini_demo/last.pt --steps 36 --context 64 --segments 2 --batch-size 1 --grad-accum 1 --lr 0.00025 --min-lr 0.00003 --warmup 0 --eval-every 6 --save-every 12 --eval-batches 3 --num-threads 2 --device cpu --precision fp32 --out runs/v06_my_run
python -m scripts.evaluate_exhaustive_v06 --checkpoint runs/v06_my_run/last.pt --corpus data/technical_bin --context 96 --threads 2
python -m scripts.eval_tasks --checkpoint runs/v06_my_run/last.pt --count 8 --seed 999 --tokens 50
python -m scripts.bench_traced_cpu --checkpoint runs/v06_my_run/last.pt --tokens 64 --threads 2 --repeats 5
```

При возобновлении optimizer, RNG и sampler RNG восстанавливаются из checkpoint; **для точного повторения** оставьте те же версии PyTorch, архитектуру, корпус, устройства и порядок операций. Изменение `--steps` меняет график cosine LR, поэтому побитовое совпадение с альтернативным расписанием не гарантируется. Это всего лишь 36 шагов на корпусе из ~19 тыс. байтов суммарно, не крупное предобучение.

В этой среде получено на **всех 7 362 целевых токенах** одного независимого validation-документа: NLL 3.1340118 (checkpoint v0.5) и 2.5612756 (после продолжения); perplexity 22.96593 → 12.95233. У модели 2 273 480 параметров. Независимый синтетический задачник: 0/8.

## Подготовить реальный корпус

Входные данные: JSONL, каждый объект `{"text":"...","source":"...","license":"CC-BY-4.0","category":"code"}`. Добавьте собственные источники, для которых у вас есть права на использование; программа не проверяет действительность лицензий и не гарантирует удаление приватных данных.

```powershell
python -m scripts.curate_corpus --input D:\Datasets\raw_licensed.jsonl --out D:\Datasets\curated --allow-license CC-BY-4.0 --allow-license CC0-1.0 --valid-percent 10 --split-unit source
python -m scripts.audit_corpus_v06 --train D:\Datasets\curated\train.jsonl --valid D:\Datasets\curated\valid.jsonl --sample 200 --out benchmarks\audit_real.json
python -m pip install tokenizers
python -m scripts.prepare_corpus --train D:\Datasets\curated\train.jsonl --valid D:\Datasets\curated\valid.jsonl --out D:\Datasets\pcmn_bpe --tokenizer bpe --vocab-size 16000
```

Аудит «почти-дубликатов» ограничен выборкой и шинглами: не умеет находить все перефразирования. Если исходных источников очень мало, `--split-unit source` может дать пустой validation split; добавьте независимые источники. Входные лицензии должны быть оценены отдельно.

## Начальный рецепт для Tesla P100 (НЕ проверен здесь)

Сначала:

```powershell
python -m scripts.hardware_preflight_v06
python -m scripts.build_cuda --arch 6.0 --device cuda:0
```

CUDA-модуль содержит исследовательское ядро FP32 для рекурсии с прямым и обратным проходом. Если компиляция или тесты градиентов не прошли, **не включайте `--cuda-scan`**. Даже если тесты пройдут, измерьте скорость: кастомное ядро может работать медленнее PyTorch.

Для последующего обучения с собственным корпусом:

```powershell
python -m scripts.train_streaming --corpus D:\Datasets\pcmn_bpe --profile 30m --memory-slots 4 --context 128 --segments 2 --batch-size 1 --grad-accum 8 --steps 100000 --eval-every 200 --save-every 500 --precision fp16 --device cuda:0 --out runs\pcmn30m_real --patience 0
```

Это **шаблон запуска, не результат**. Проверьте VRAM, batch size, перегрев P100, скорость, стабильность FP16 и доступное место. Качественная LLM уровня современных крупных моделей потребует значительно больше качественных токенов и вычислений, чем демонстрация v0.6.

## Работа с памятью / локальный API

```powershell
python -m scripts.build_retrieval --input D:\Datasets\reference.jsonl --out data\my_memory.jsonl --chunk-words 180 --overlap 40
python -m scripts.serve_api --checkpoint runs/v06_continued_demo/last.pt --device cpu --host 127.0.0.1 --port 8765 --trace-cpu --memory-index data/my_memory.jsonl --grounded
```

`--grounded` подготавливает пронумерованные источники в JSONL внутри промпта. Это **не** защищает полностью от prompt injection и не доказывает, что модель правильно использует документы. Сохраняйте песочницу и подтверждения для агентных действий. API лишь частично совместим с OpenAI-style endpoint; прежде чем открывать внешний доступ, нужны отдельная аутентификация, TLS и лимиты.

## Результаты и ограничения

- Протоколы: `reports/STATUS_V06_RU.md`, `reports/v06_training.log`, `benchmarks/v06_exhaustive_after.json`, `benchmarks/v06_cpu_decode.json`, `benchmarks/v06_quality_tasks.json`, `benchmarks/hardware_v06.json`.
- CPU замер (2 потока, 64 decode токена, одна небольшая модель): eager 467.50 ток/с, traced 691.42 ток/с, 1.479×, численная разница логитов 0. Значения относятся к доступному CPU контейнера, **не** к EPYC 7352.
- CUDA `NOT_TESTED`: в этой среде нет устройства NVIDIA и компилятора `nvcc`.
- `torch.jit.trace` — устаревающий API; путь CPU требует дальнейшей миграции, замер не доказывает превосходство архитектуры.
- В обучении использованы демонстрационные данные, улучшение NLL не означает понимания или навыков программирования.
- Код PCMN не реализует полноценную мультимодальную LLM или гарантированно безопасного автономного агента.
- Файлы PyTorch `.pt` загружайте **только из доверенных источников**: они могут использовать небезопасную сериализацию Python.
