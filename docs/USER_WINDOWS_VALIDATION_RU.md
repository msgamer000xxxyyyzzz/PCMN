# Windows 11 — проверка PCMN v0.6 от пользователя (19.10.2026)

Источник: `reports/USER_LOCAL_WINDOWS_RESULTS_2026-10-19.txt`. Представленные ниже цифры получены на компьютере пользователя, а не в CI.

## Подтверждённые результаты

| Проверка | Результат |
|---|---|
| Unit tests | 53 запущено, 52 успешных, 1 пропущен: `test_native_scan_exact_forward` (нативная библиотека не собрана) |
| Python | 3.14.3 |
| PyTorch | 2.14.1+cpu |
| CUDA | недоступна, devices=[] |
| `nvcc` | найден, но CUDA-PyTorch не установлен |
| Число логических CPU / потоки PyTorch | 48 / 24 (по данным preflight) |
| Модель | 2 273 480 параметров, байтовый словарь 259 |
| Обучение | продолжено от шага 12 до шага 36, CPU fp32 |
| Исчерпывающая валидация | NLL 3.1340118 → 2.5612756, perplexity 22.9659304 → 12.9523292 |
| Задачи: арифметика, причинность, отслеживание состояния, Python-код | exact-match: 0/8 |
| Генерация на CPU, два потока | eager 135.0 токен/с; TorchScript trace 220.47 токен/с, ускорение 1.633× |

Данные относятся к очень маленькому техническому корпусу (train 11599 токенов, validation 7363), поэтому не показывают реального качества модели общего назначения. Метрики скорости получены без оценки качества больших моделей.

## Обнаруженные проблемы и действия

**`No module named 'numpy'`:** установка `python -m pip install -r requirements.txt` теперь включает `numpy`. Сообщение было предупреждением, а не ошибкой тестов.

**`torch 2.14.1+cpu`, `cuda_available: false`, `nvcc_found: true`:** установлен CPU-only PyTorch. Наличие компилятора CUDA не означает, что PyTorch умеет исполнять CUDA. Проверить драйвер и совместимую бинарную сборку PyTorch для Tesla P100 (`sm_60`) до GPU-бенчмарков.

**`torch.jit.trace` предупреждает о Python 3.14+:** рекомендуется отдельное окружение Python 3.12 для воспроизведения; в дальнейшем заменить устаревающую трассировку на поддерживаемый CPU-runtime только после проверки эквивалентности.

**`test_native_scan_exact_forward` — skipped:** отдельно собрать и загрузить C++ library и затем повторить тест; пометка skipped не равна успешному тестированию.

## Команды проверки

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python -m scripts.hardware_preflight_v06 --out benchmarks/hardware_local.json
python -m scripts.evaluate_exhaustive_v06 --checkpoint runs/v06_my_run/last.pt --corpus data/technical_bin --context 96 --threads 2
python -m scripts.eval_tasks --checkpoint runs/v06_my_run/last.pt --count 8 --seed 999 --tokens 50
python -m scripts.bench_traced_cpu --checkpoint runs/v06_my_run/last.pt --tokens 64 --threads 2 --repeats 5
```

Пути к весам и бинарному корпусу в командах относятся к локальным файлам и не входят в Git-репозиторий. Исходный текст отчёта сохранён без изменений.
