# whisper-gpu

Локальная расшифровка аудио и видео через **faster-whisper** на CUDA: txt + srt + json, без облака и без лимитов на минуты.

Обёртка над `faster-whisper` с нормальным CLI, Docker-образом и замером скорости — чтобы одна и та же команда работала и на ноутбуке, и на машине с GPU.

## Что умеет

- Модели `tiny`…`large-v3`, язык задаётся или определяется автоматически.
- Вывод: `.txt`, `.srt`, `.json` (сегменты с таймкодами).
- VAD-фильтр, чтобы не транскрибировать тишину.
- Docker-образ с CUDA 12.4 и ffmpeg — ничего не надо ставить в систему.
- `bench.py` — реальный замер RTF (real-time factor) на твоём железе.

## Быстрый старт

```bash
pip install faster-whisper
python transcribe.py lecture.mp4
# lecture.txt, lecture.srt, lecture.json рядом с файлом
```

Модель по умолчанию `medium`. Путь к локальной копии модели задаётся через `WHISPER_MODEL_PATH`, если HF недоступен:

```bash
WHISPER_MODEL_PATH=/models/medium python transcribe.py lecture.mp4
```

## Docker

```bash
docker compose run --rm whisper lecture.mp4
```

Модели кэшируются в томе `whisper-models`, поэтому скачиваются один раз.

## Бенчмарк

```bash
python bench.py sample.mp3 --model medium --device cuda
```

Скрипт печатает время, RTF и пропускную способность. RTF = время расшифровки / длительность аудио: меньше 1 — быстрее реального времени.

Результаты своего прогона складывай в `docs/BENCHMARKS.md` — таблица уже заготовлена.

## Почему faster-whisper

| | openai/whisper | faster-whisper |
|---|---|---|
| Скорость на GPU | базовая | в 4 раза быстрее при том же качестве |
| Память | больше | меньше за счёт int8/float16 |
| VAD | отдельно | встроен |

## Ограничения

- Нужен ffmpeg для не-WAV контейнеров (в Docker он уже есть).
- `large-v3` на 8 ГБ VRAM идёт только в `int8_float16`.
- Диаризация (кто говорит) не поддерживается — это отдельная задача.

## Лицензия

MIT
