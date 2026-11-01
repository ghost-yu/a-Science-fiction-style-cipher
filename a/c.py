import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import base64, hashlib, heapq, hmac, json, math, sys
from pathlib import Path
import torch
from Crypto.Cipher import AES
from nacl import pwhash, utils
from d import Config, CharTransformer
import e as date_seal
ROOT = Path(__file__).resolve().parent
STARTERS = ['傍晚，我', '雨停后，我', '散步时，我', '我走进书店，', '有一天，', '一天，', '从前，']

class LetterEngine:
    minimum_body_bytes = 17

    def __init__(self, device='cuda', topk=8, gate=0.92, context='[短篇64字]', manifest_name='h', create_manifest=False):
        torch.set_num_threads(4)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        self.device = device
        self.topk = topk
        self.gate = gate
        self.context = context
        path = ROOT / 'f'
        self.model_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
        self.model = CharTransformer(Config(**checkpoint['config'])).to(device).eval()
        self.model.load_state_dict(checkpoint['model'], strict=True)
        self.vocab_path = ROOT / 'g'
        self.vocab = json.loads(self.vocab_path.read_text(encoding='utf-8'))['tokens']
        self.ids = {c: i for i, c in enumerate(self.vocab)}
        self.alphabet = self.vocab[4:8109]
        self.char_index = {c: i for i, c in enumerate(self.alphabet)}
        manifest = ROOT / manifest_name
        if not manifest.exists():
            if not create_manifest:
                raise ValueError('Missing public manifest. Restore it with the model; do not replace its salt.')
            manifest.write_text(json.dumps({'version': 'neural-letter-v1', 'public_salt': utils.random(16).hex(), 'model_sha256': self.model_hash, 'vocab_sha256': hashlib.sha256(self.vocab_path.read_bytes()).hexdigest(), 'context': context, 'topk': topk, 'forced_threshold': gate, 'precision': 'float32-math-SDPA'}, ensure_ascii=False, indent=2), encoding='utf-8')
        self.manifest = json.loads(manifest.read_text(encoding='utf-8'))
        if self.manifest['model_sha256'] != self.model_hash or self.manifest['topk'] != topk or self.manifest['forced_threshold'] != gate or (self.manifest['context'] != context):
            raise ValueError('Protocol/model settings differ from the frozen manifest.')
        if self.manifest['vocab_sha256'] != hashlib.sha256(self.vocab_path.read_bytes()).hexdigest():
            raise ValueError('Vocabulary differs from the frozen manifest.')
        self.cache = {}

    def keys(self, password):
        key = pwhash.argon2id.kdf(64, password.encode('utf-8'), bytes.fromhex(self.manifest['public_salt']), opslimit=2, memlimit=64 * 1024 * 1024)
        return (key, hmac.new(key, b'TL069/date-key/v1', hashlib.sha256).digest())

    def pack(self, message):
        if not 1 <= len(message) <= 10:
            raise ValueError('Expected one to ten characters.')
        if all((c in self.char_index for c in message)):
            bits = '0' + format(len(message), '04b') + ''.join((format(self.char_index[c], '013b') for c in message))
            bits += '0' * (-len(bits) % 8)
            return bytes((int(bits[i:i + 8], 2) for i in range(0, len(bits), 8)))
        return bytes([128, len(message)]) + message.encode('utf-8')

    def unpack(self, data):
        if not data:
            raise ValueError('Empty message.')
        if data[0] & 128:
            text = data[2:].decode('utf-8')
            if not 1 <= len(text) <= 10 or len(text) != data[1]:
                raise ValueError('Bad UTF-8 frame.')
            return text
        bits = ''.join((format(x, '08b') for x in data))
        n = int(bits[1:5], 2)
        if not 1 <= n <= 10 or len(bits) < 5 + 13 * n:
            raise ValueError('Bad character frame.')
        codes = [int(bits[5 + 13 * i:18 + 13 * i], 2) for i in range(n)]
        if any((x >= len(self.alphabet) for x in codes)) or '1' in bits[5 + 13 * n:]:
            raise ValueError('Bad packed character.')
        return ''.join((self.alphabet[i] for i in codes))

    def aad(self, received):
        received = date_seal._date(received).isoformat()
        return ('neural-letter-v1|Asia/Shanghai|' + received + '|' + self.model_hash).encode('ascii')

    @torch.no_grad()
    def probabilities(self, text):
        seq = [1] + [self.ids[c] for c in self.context + '\n' + text]
        seq = seq[-self.model.config.block_size:]
        if text in self.cache:
            return self.cache[text]
        x = torch.tensor([seq], device=self.device)
        with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
            logits, _ = self.model(x)
        scores = logits[0, -1].double().cpu()
        for i, c in enumerate(self.vocab):
            if i < 4 or len(c) != 1 or ord(c) < 128:
                scores[i] = -float('inf')
        p = torch.softmax(scores, dim=0).numpy()
        ranked = sorted(range(len(p)), key=lambda i: (-p[i], i))
        result = (p, ranked)
        self.cache[text] = result
        return result

    def choices(self, text):
        p, rank = self.probabilities(text)
        if p[rank[0]] >= self.gate:
            return {self.vocab[rank[0]]: ''}
        heap = []
        for serial, idx in enumerate(rank[:self.topk]):
            heapq.heappush(heap, (max(1, round(float(p[idx]) * 2 ** 24)), serial, idx))
        serial = self.topk
        while len(heap) > 1:
            a = heapq.heappop(heap)
            b = heapq.heappop(heap)
            heapq.heappush(heap, (a[0] + b[0], serial, (a[2], b[2])))
            serial += 1
        codes = {}

        def walk(node, path):
            if isinstance(node, int):
                codes[self.vocab[node]] = path
            else:
                walk(node[0], path + '0')
                walk(node[1], path + '1')
        walk(heap[0][2], '')
        return codes

    def hide(self, body, starter=0, max_characters=1200, include_length=True):
        frame = (bytes([len(body)]) if include_length else b'') + body
        bits = ''.join((format(x, '08b') for x in frame))
        text = STARTERS[starter]
        offset = 0
        while offset < len(bits):
            if len(text) >= max_characters:
                raise RuntimeError('Carrier budget exceeded.')
            codes = self.choices(text)
            if len(codes) == 1:
                ch = next(iter(codes))
            else:
                ch = None
                for candidate, code in codes.items():
                    remaining = bits[offset:offset + len(code)].ljust(len(code), '0')
                    if remaining == code:
                        ch = candidate
                        offset += len(code)
                        break
                if ch is None:
                    raise RuntimeError('Huffman partition failed.')
            text += ch
        for _ in range(48):
            if text[-1] in '。！？':
                break
            p, rank = self.probabilities(text)
            text += self.vocab[rank[0]]
        if text.count('“') > text.count('”'):
            text += '”' * (text.count('“') - text.count('”'))
        return text

    def extract(self, text, known_length=None):
        starts = [s for s in STARTERS if text.startswith(s)]
        if len(starts) != 1:
            raise ValueError('Unknown story opening.')
        prefix = starts[0]
        bits = ''
        total = None if known_length is None else 8 * known_length
        for char in text[len(prefix):]:
            codes = self.choices(prefix)
            if char not in codes:
                raise ValueError('Character outside deterministic candidate set.')
            bits += codes[char]
            prefix += char
            if total is None and len(bits) >= 8:
                n = int(bits[:8], 2)
                if not self.minimum_body_bytes <= n <= 80:
                    raise ValueError('Invalid encrypted frame length.')
                total = 8 + 8 * n
            if total is not None and len(bits) >= total:
                start = 8 if known_length is None else 0
                return bytes((int(bits[i:i + 8], 2) for i in range(start, total, 8)))
        raise ValueError('Truncated story.')
