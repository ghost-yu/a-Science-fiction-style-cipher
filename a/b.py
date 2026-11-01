import base64, hashlib, hmac
from Crypto.Cipher import AES
from nacl import utils
from c import LetterEngine, date_seal
ALPHABET = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_'

def b64(data):
    return base64.urlsafe_b64encode(data).decode('ascii').rstrip('=')

def unb64(text):
    if len(text) != 11:
        raise ValueError('Expected eleven nonce characters.')
    data = base64.urlsafe_b64decode(text + '=')
    if len(data) != 8 or b64(data) != text:
        raise ValueError('Non-canonical nonce.')
    return data

class CompactEngine(LetterEngine):

    def __init__(self, external_nonce=False, **kwargs):
        self.external_nonce = external_nonce
        self.profile = 'compact-ctr64-external-v2' if external_nonce else 'compact-ctr64-internal-v2'
        self.minimum_body_bytes = 1 if external_nonce else 9
        kwargs.setdefault('manifest_name', 'i' if external_nonce else 'h')
        super().__init__(**kwargs)

    def encryption_key(self, password, received):
        master, _ = self.keys(password)
        return hmac.new(master, b'encryption\x00' + self.profile.encode('ascii') + b'\x00' + self.aad(received), hashlib.sha256).digest()

    def date_key(self, password):
        master, _ = self.keys(password)
        return hmac.new(master, b'date-key\x00' + self.profile.encode('ascii') + b'\x00' + self.model_hash.encode('ascii'), hashlib.sha256).digest()

    def encode(self, password, received, written, event, message, starter=0):
        r = date_seal._date(received)
        a = (date_seal._date(written) - r).days
        b = (date_seal._date(event) - r).days
        if abs(a) > 3650 or abs(b) > 30:
            raise ValueError('Dates outside agreed search window.')
        nonce = utils.random(8)
        cipher = AES.new(self.encryption_key(password, received), AES.MODE_CTR, nonce=nonce)
        ciphertext = cipher.encrypt(self.pack(message))
        body = nonce + ciphertext
        payload = ciphertext if self.external_nonce else body
        story = self.hide(payload, starter, include_length=False)
        seal = date_seal.seal(self.date_key(password), received, written, event, body, story)
        code = ALPHABET[len(payload)] + (b64(nonce) + seal if self.external_nonce else seal)
        return {'profile': self.profile, 'story': story, 'public_code': code, 'story_characters': len(story), 'code_characters': len(code), 'story_payload_bytes': len(ciphertext) if self.external_nonce else len(body), 'starter': starter}

    def decode(self, password, received, story, code):
        if len(code) != (23 if self.external_nonce else 12):
            raise ValueError('Wrong public code length.')
        if code[0] not in ALPHABET:
            raise ValueError('Invalid length character.')
        n = ALPHABET.index(code[0])
        if not self.minimum_body_bytes <= n <= 50:
            raise ValueError('Invalid payload length.')
        extracted = self.extract(story, known_length=n)
        body = unb64(code[1:12]) + extracted if self.external_nonce else extracted
        seal = code[12:] if self.external_nonce else code[1:]
        dates = date_seal.recover(self.date_key(password), received, seal, body, story)
        cipher = AES.new(self.encryption_key(password, received), AES.MODE_CTR, nonce=body[:8])
        message = self.unpack(cipher.decrypt(body[8:]))
        return {'message': message, **dates}
