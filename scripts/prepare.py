import argparse
import hashlib
import json
from pathlib import Path
import re
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
SOURCES = json.loads(Path(__file__).with_name('data_sources.json').read_text())


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]


def save(data, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')


def download(url, path, expected=None):
    if path.exists():
        if expected and hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f'Cached input hash mismatch: {path}. Remove this file and retry.')
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.part')
    for attempt in range(3):
        try:
            print(f'Downloading {url}', flush=True)
            with urllib.request.urlopen(url, timeout=120) as response, temporary.open('wb') as out:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    out.write(chunk)
            if expected and hashlib.sha256(temporary.read_bytes()).hexdigest() != expected:
                raise ValueError(f'Downloaded input hash mismatch: {path}')
            temporary.replace(path)
            return path
        except (OSError, ValueError):
            if attempt == 2:
                raise
            time.sleep(2 ** attempt)


def build_musique(path):

    questions, paragraphs = [], {}
    for row in read_jsonl(path):
        questions.append(dict(id=f"musique_{row['id']}", source='musique',
                              question=row['question'], answer=row['answer'],
                              question_type='', evidence=''))
        for p in row['paragraphs']:
            key = (p['title'], p['paragraph_text'])
            paragraphs.setdefault(key, f"{p['title']}\n{p['paragraph_text']}")
    words = []
    for _, paragraph in sorted(paragraphs.items()):
        text = paragraph.split('\n', 1)[1].lower()
        text = re.sub(r'([^\w\s])', r' \1 ', text)
        words.extend(re.sub(r'\s+', ' ', text).strip().split())
    chunks = [f"{i // 838}:" + ' '.join(words[i:i + 838]) for i in range(0, len(words), 838)]
    return {'questions.json': questions, 'chunks.json': chunks}


def split_wix(text):
    if len(text) <= 5000:
        return [text]
    parts, start = [], 0
    while start < len(text):
        if len(text) - start <= 5000:
            parts.append(text[start:])
            break
        end = start + 5000
        period = text.find('.', end, end + 300)
        cut = period + 1 if period != -1 else end
        parts.append(text[start:cut])
        start = cut
        while start < len(text) and text[start] in ' \n\t':
            start += 1
    return [part for part in parts if part.strip()]


def build_wix(paths):
    chunks, meta, questions = [], {}, []
    for article in read_jsonl(paths[0]):
        title, contents = article.get('title', '').strip(), article.get('contents', '').strip()
        text = contents if contents.startswith(title) else (f'{title}\n{contents}' if title else contents)
        for part in split_wix(text):
            meta[str(len(chunks))] = article['id']
            chunks.append(part)

    for source, path in zip(['expertwritten', 'simulated'], paths[1:]):
        for row in read_jsonl(path):
            questions.append(dict(question=row['question'], answer=row['answer'],
                                  article_ids=row.get('article_ids', []), source=source))
    return {'questions.json': questions, 'chunks.json': chunks, 'corpus_meta.json': meta}


def prepare_dataset(name, data_dir):
    spec = SOURCES[name]
    raw = data_dir / '_raw' / name
    paths = [download(f"https://huggingface.co/datasets/{spec['repo']}/resolve/{spec['revision']}/{f}",
                      raw / f, spec.get('raw_sha256', {}).get(f)) for f in spec['files']]
    if name == 'musique_full':
        output = build_musique(paths[0])
    elif name == 'wixqa':
        output = build_wix(paths)
    else:
        output = {path.name: json.loads(path.read_text(encoding='utf-8')) for path in paths}

    for filename, data in output.items():
        canonical = json.dumps(data, sort_keys=True, ensure_ascii=False, separators=(',', ':')).encode()
        expected = spec['expected'][filename]
        if len(data) != expected['count'] or hashlib.sha256(canonical).hexdigest() != expected['sha256']:
            raise ValueError(f'Data validation failed: {name}/{filename}')
    for filename, data in output.items():
        save(data, data_dir / name / filename)
        print(f'Verified {name}/{filename}: {len(data)} records', flush=True)
    return output


def prepare_models(model_dir):
    from huggingface_hub import snapshot_download
    models = [
        ('sentence-transformers/all-mpnet-base-v2', 'e8c3b32edf5434bc2275fc9bab85f82640a19130', 'all-mpnet-base-v2',
         ['config.json', 'config_sentence_transformers.json', 'model.safetensors', 'modules.json',
          'sentence_bert_config.json', 'special_tokens_map.json', 'tokenizer.json', 'tokenizer_config.json',
          'vocab.txt', '1_Pooling/config.json']),
        ('cross-encoder/nli-deberta-v3-large', 'bab4bc7178836f731dcfd18c06ca9def0a137712', 'nli-deberta-v3-large',
         ['config.json', 'model.safetensors', 'added_tokens.json', 'special_tokens_map.json',
          'spm.model', 'tokenizer_config.json']),
    ]
    for repo, revision, folder, files in models:
        for attempt in range(3):
            try:
                snapshot_download(repo, revision=revision, allow_patterns=files, local_dir=str(model_dir / folder),
                                  resume_download=True, max_workers=1, etag_timeout=60)
                print(f'Model ready: {model_dir / folder}', flush=True)
                break
            except (OSError, ValueError) as exc:
                if attempt == 2:
                    raise
                print(f'Model download interrupted ({type(exc).__name__}); retrying from cache.', flush=True)
                time.sleep(2 ** attempt)


def main():
    parser = argparse.ArgumentParser(description="Download pinned inputs and reproduce the paper's processed datasets.")
    parser.add_argument('--dataset', choices=[*SOURCES, 'all'])
    parser.add_argument('--models', action='store_true')
    parser.add_argument('--data-dir', type=Path, default=ROOT / 'dataset')
    parser.add_argument('--model-dir', type=Path, default=ROOT / 'model')
    args = parser.parse_args()
    if not args.dataset and not args.models:
        parser.error('Specify --dataset, --models, or both.')
    if args.dataset:
        for name in SOURCES if args.dataset == 'all' else [args.dataset]:
            prepare_dataset(name, args.data_dir)
    if args.models:
        prepare_models(args.model_dir)


if __name__ == '__main__':
    main()
